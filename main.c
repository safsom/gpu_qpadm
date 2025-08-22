/*
 * main.c - command line interface for gpu_qpadm
 *
 * This program exposes three subcommands:
 *   precompute <geno> <ind> <block_size> <cache>
 *     Read EIGENSTRAT genotype (.geno) and individual (.ind) files,
 *     compute per‑population allele frequencies and write blockwise
 *     Gram matrices to <cache> using the GPU.
 *
 *   qpadm <cache> <target> <base> <right_pops> <sources>
 *     Load a Gram cache and perform a qpAdm fit.  <target> and
 *     <base> are integer indices referring to populations.  The
 *     comma‑separated list <right_pops> defines the outgroups, and
 *     <sources> defines the candidate source populations.  Outputs
 *     mixture proportions and approximate standard errors.
 *
 * Usage examples:
 *   gpu_qpadm precompute data.geno data.ind 50000 data.cache
 *   gpu_qpadm qpadm data.cache 3 0 4,5,6 1,2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/* forward declarations from other translation units */
typedef struct GenoData GenoData;
GenoData *read_eigenstrat(const char *geno_filename, const char *ind_filename, int block_size);
void free_genodata(GenoData *gd);
int precompute_gram_gpu(const GenoData *gd, const char *cache_filename);

typedef struct {
    int num_pops;
    int num_blocks;
    double *grams;
} GramCache;
int load_gram_cache(const char *filename, GramCache *cache);
void free_gram_cache(GramCache *cache);
int build_F_b(const GramCache *cache,
              int target,
              int base,
              const int *right_pops,
              int k,
              const int *sources,
              int j,
              double *F,
              double *b);
int compute_b_covariance(const GramCache *cache,
                         int target,
                         int base,
                         const int *right_pops,
                         int k,
                         double *cov);
int solve_qpadm(int k, int j,
                const double *F,
                const double *b,
                const double *Winv,
                double *coeff,
                double *se);

/* invert a small symmetric matrix (row‑major) using Gauss‑Jordan */
static int invert_small_matrix(double *A, double *Ainv, int n) {
    /* reuse invert_matrix from qpadm_api by duplicating logic */
    int dim = 2 * n;
    double *aug = (double *)calloc(n * dim, sizeof(double));
    if (!aug) return -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * dim + j] = A[i * n + j];
        }
        for (int j = 0; j < n; j++) {
            aug[i * dim + (n + j)] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < n; i++) {
        double pivot = aug[i * dim + i];
        if (fabs(pivot) < 1e-12) {
            free(aug);
            return -1;
        }
        double inv_pivot = 1.0 / pivot;
        for (int j = i; j < dim; j++) {
            aug[i * dim + j] *= inv_pivot;
        }
        for (int row = 0; row < n; row++) {
            if (row == i) continue;
            double factor = aug[row * dim + i];
            for (int col = i; col < dim; col++) {
                aug[row * dim + col] -= factor * aug[i * dim + col];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ainv[i * n + j] = aug[i * dim + (n + j)];
        }
    }
    free(aug);
    return 0;
}

/* parse a comma‑separated list of integers; returns length via out_len */
static int *parse_int_list(const char *str, int *out_len) {
    if (!str || !out_len) return NULL;
    /* count commas */
    int count = 1;
    for (const char *p = str; *p; p++) {
        if (*p == ',') count++;
    }
    int *arr = (int *)malloc(sizeof(int) * count);
    int idx = 0;
    const char *start = str;
    char *end;
    while (1) {
        long val = strtol(start, &end, 10);
        if (start == end) {
            fprintf(stderr, "Invalid integer in list: %s\n", str);
            free(arr);
            *out_len = 0;
            return NULL;
        }
        arr[idx++] = (int)val;
        if (*end == ',') {
            start = end + 1;
        } else {
            break;
        }
    }
    *out_len = idx;
    return arr;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s precompute <geno> <ind> <block_size> <cache>\n", prog);
    fprintf(stderr, "  %s qpadm <cache> <target> <base> <right_pops> <sources>\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "precompute") == 0) {
        if (argc != 6) {
            usage(argv[0]);
            return 1;
        }
        const char *geno = argv[2];
        const char *ind  = argv[3];
        int block_size   = atoi(argv[4]);
        const char *cache = argv[5];
        if (block_size <= 0) {
            fprintf(stderr, "Invalid block size %s\n", argv[4]);
            return 1;
        }
        GenoData *gd = read_eigenstrat(geno, ind, block_size);
        if (!gd) {
            fprintf(stderr, "Failed to read EIGENSTRAT files\n");
            return 1;
        }
        if (precompute_gram_gpu(gd, cache) != 0) {
            fprintf(stderr, "Failed to precompute Gram matrices\n");
            free_genodata(gd);
            return 1;
        }
        int nb = gd->num_blocks;
        int np = gd->num_pops;
        free_genodata(gd);
        fprintf(stdout, "Precomputation complete. Wrote %d blocks for %d populations to %s\n", nb, np, cache);
        return 0;
    } else if (strcmp(argv[1], "qpadm") == 0) {
        if (argc != 7) {
            usage(argv[0]);
            return 1;
        }
        const char *cache_file = argv[2];
        int target = atoi(argv[3]);
        int base   = atoi(argv[4]);
        const char *right_list = argv[5];
        const char *source_list = argv[6];
        int k, j;
        int *right_pops = parse_int_list(right_list, &k);
        int *sources = parse_int_list(source_list, &j);
        if (!right_pops || !sources) {
            fprintf(stderr, "Failed to parse population lists\n");
            free(right_pops);
            free(sources);
            return 1;
        }
        GramCache cache;
        if (load_gram_cache(cache_file, &cache) != 0) {
            fprintf(stderr, "Failed to load Gram cache\n");
            free(right_pops);
            free(sources);
            return 1;
        }
        /* allocate F, b */
        double *F = (double *)calloc(k * j, sizeof(double));
        double *b = (double *)calloc(k, sizeof(double));
        if (!F || !b) {
            fprintf(stderr, "Memory allocation error\n");
            free(F); free(b);
            free(right_pops); free(sources);
            free_gram_cache(&cache);
            return 1;
        }
        if (build_F_b(&cache, target, base, right_pops, k, sources, j, F, b) != 0) {
            fprintf(stderr, "Failed to build F and b\n");
            free(F); free(b);
            free(right_pops); free(sources);
            free_gram_cache(&cache);
            return 1;
        }
        /* compute covariance of b and invert */
        double *cov = (double *)calloc(k * k, sizeof(double));
        double *Winv = (double *)calloc(k * k, sizeof(double));
        if (!cov || !Winv) {
            fprintf(stderr, "Memory allocation error\n");
            free(F); free(b);
            free(cov); free(Winv);
            free(right_pops); free(sources);
            free_gram_cache(&cache);
            return 1;
        }
        if (compute_b_covariance(&cache, target, base, right_pops, k, cov) != 0) {
            fprintf(stderr, "Failed to compute b covariance\n");
            free(F); free(b); free(cov); free(Winv);
            free(right_pops); free(sources);
            free_gram_cache(&cache);
            return 1;
        }
        if (invert_small_matrix(cov, Winv, k) != 0) {
            fprintf(stderr, "Failed to invert covariance matrix (singular?)\n");
            free(F); free(b); free(cov); free(Winv);
            free(right_pops); free(sources);
            free_gram_cache(&cache);
            return 1;
        }
        /* solve for mixture coefficients */
        double *coeff = (double *)calloc(j, sizeof(double));
        double *se    = (double *)calloc(j, sizeof(double));
        if (!coeff || !se) {
            fprintf(stderr, "Memory allocation error\n");
            free(F); free(b); free(cov); free(Winv);
            free(coeff); free(se);
            free(right_pops); free(sources);
            free_gram_cache(&cache);
            return 1;
        }
        if (solve_qpadm(k, j, F, b, Winv, coeff, se) != 0) {
            fprintf(stderr, "qpAdm solve failed\n");
            free(F); free(b); free(cov); free(Winv);
            free(coeff); free(se);
            free(right_pops); free(sources);
            free_gram_cache(&cache);
            return 1;
        }
        /* output results */
        printf("qpAdm result\n");
        for (int jj = 0; jj < j; jj++) {
            printf("source[%d] = %d, coeff = %.6f, se ≈ %.6f\n", jj, sources[jj], coeff[jj], se[jj]);
        }
        /* cleanup */
        free(F); free(b); free(cov); free(Winv);
        free(coeff); free(se);
        free(right_pops); free(sources);
        free_gram_cache(&cache);
        return 0;
    } else {
        usage(argv[0]);
        return 1;
    }
}