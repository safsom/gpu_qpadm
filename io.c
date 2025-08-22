/*
 * io.c - simplified I/O helper routines for gpu_qpadm
 *
 * This module implements a bare-bones reader for the EIGENSTRAT
 * genotype format used by the original ADMIXTOOLS suite.  It
 * aggregates per-population allele frequencies for each SNP and
 * partitions SNPs into blocks for downstream jackknife and
 * covariance estimation.  The goal here is to provide just
 * enough infrastructure to allow the rest of the codebase to
 * precompute Gram matrices on the GPU and perform qpWave/qpAdm
 * style analyses.  Many corner cases and format variations
 * supported by the legacy code (ancestrymap, transposed formats,
 * sex chromosome filtering, etc.) are intentionally omitted
 * for clarity.  Feel free to extend this module if you need
 * additional functionality.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Data structures
 */
typedef struct {
    int num_inds;         /* number of individuals */
    int num_pops;         /* number of distinct populations */
    int *ind_pop;         /* length num_inds; mapping from individual to population index */
    char **pop_names;     /* length num_pops; population names */
} PopInfo;

typedef struct {
    int num_snps;         /* number of SNPs */
    int num_inds;         /* number of individuals */
    int num_pops;         /* number of distinct populations */
    int num_blocks;       /* number of jackknife blocks */
    int *block_start;     /* length num_blocks; index of first SNP in block */
    int *block_end;       /* length num_blocks; index of one past last SNP in block */
    double **snp_means;   /* snp_means[p][s] = allele frequency for pop p at SNP s */
} GenoData;

/*
 * Utility to trim newline from a string in place.
 */
static void trim_newline(char *buf) {
    size_t n = strlen(buf);
    if (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) {
        buf[n-1] = '\0';
        if (n > 1 && buf[n-2] == '\r') {
            buf[n-2] = '\0';
        }
    }
}

/*
 * Parse the .ind file to build a PopInfo structure.  The .ind file
 * contains one individual per line with three whitespace separated
 * columns: individual ID, sex, population label.  Sex is ignored
 * here.  Duplicate population labels will be collapsed into a
 * single index.
 */
static PopInfo *parse_ind_file(const char *ind_filename) {
    FILE *fp = fopen(ind_filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open .ind file %s\n", ind_filename);
        return NULL;
    }
    int num_inds = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\0' || line[0] == '#') continue;
        num_inds++;
    }
    rewind(fp);
    PopInfo *info = (PopInfo *)calloc(1, sizeof(PopInfo));
    info->num_inds = num_inds;
    info->ind_pop = (int *)calloc(num_inds, sizeof(int));
    int pops_alloc = 8;
    int num_pops = 0;
    char **pops = (char **)malloc(pops_alloc * sizeof(char *));
    int idx = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\0' || line[0] == '#') continue;
        trim_newline(line);
        char *id = strtok(line, " \t");
        char *sex = strtok(NULL, " \t");
        char *pop = strtok(NULL, " \t");
        (void)id;
        (void)sex;
        if (!pop) {
            fprintf(stderr, "Malformed .ind line, expected population label\n");
            free(info->ind_pop);
            free(info);
            return NULL;
        }
        int p;
        int found = 0;
        for (p = 0; p < num_pops; p++) {
            if (strcmp(pop, pops[p]) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            if (num_pops >= pops_alloc) {
                pops_alloc *= 2;
                pops = (char **)realloc(pops, pops_alloc * sizeof(char *));
            }
            pops[num_pops] = strdup(pop);
            p = num_pops;
            num_pops++;
        }
        info->ind_pop[idx++] = p;
    }
    info->num_pops = num_pops;
    info->pop_names = pops;
    fclose(fp);
    return info;
}

/*
 * Count the number of SNPs in a .geno file.  This simply counts
 * lines; empty lines are ignored.  Returns zero on error.
 */
static int count_snps(const char *geno_filename) {
    FILE *fp = fopen(geno_filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open .geno file %s\n", geno_filename);
        return 0;
    }
    int count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\0') continue;
        count++;
    }
    fclose(fp);
    return count;
}

/*
 * Read the .geno file and compute per-population means per SNP.  The
 * resulting GenoData structure contains snp_means[p][s] for all
 * populations p and SNPs s.  The block_size parameter controls the
 * approximate number of SNPs per jackknife block.
 */
static GenoData *read_geno_means(const char *geno_filename, const PopInfo *info, int block_size) {
    int num_snps = count_snps(geno_filename);
    if (num_snps == 0) {
        return NULL;
    }
    FILE *fp = fopen(geno_filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open .geno file %s\n", geno_filename);
        return NULL;
    }
    GenoData *gd = (GenoData *)calloc(1, sizeof(GenoData));
    gd->num_snps = num_snps;
    gd->num_inds = info->num_inds;
    gd->num_pops = info->num_pops;
    gd->snp_means = (double **)malloc(gd->num_pops * sizeof(double *));
    for (int p = 0; p < gd->num_pops; p++) {
        gd->snp_means[p] = (double *)calloc(num_snps, sizeof(double));
    }
    int *pop_counts = (int *)calloc(gd->num_pops, sizeof(int));
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int snp_idx = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if (read <= 1) continue;
        memset(pop_counts, 0, sizeof(int) * gd->num_pops);
        for (int i = 0; i < info->num_inds; i++) {
            char c = line[i];
            if (c == '9' || c == '\n' || c == '\r') {
                continue;
            }
            int g = c - '0';
            int p = info->ind_pop[i];
            gd->snp_means[p][snp_idx] += g;
            pop_counts[p]++;
        }
        for (int p = 0; p < gd->num_pops; p++) {
            if (pop_counts[p] > 0) {
                gd->snp_means[p][snp_idx] /= (2.0 * pop_counts[p]);
            } else {
                gd->snp_means[p][snp_idx] = 0.0;
            }
        }
        snp_idx++;
    }
    if (line) free(line);
    free(pop_counts);
    fclose(fp);
    int num_blocks = (num_snps + block_size - 1) / block_size;
    gd->num_blocks = num_blocks;
    gd->block_start = (int *)malloc(num_blocks * sizeof(int));
    gd->block_end = (int *)malloc(num_blocks * sizeof(int));
    for (int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        int end = start + block_size;
        if (end > num_snps) end = num_snps;
        gd->block_start[b] = start;
        gd->block_end[b] = end;
    }
    return gd;
}

/*
 * Free a PopInfo structure and its internal allocations.
 */
static void free_popinfo(PopInfo *info) {
    if (!info) return;
    for (int i = 0; i < info->num_pops; i++) {
        free(info->pop_names[i]);
    }
    free(info->pop_names);
    free(info->ind_pop);
    free(info);
}

/*
 * Free a GenoData structure.
 */
static void free_genodata_internal(GenoData *gd) {
    if (!gd) return;
    if (gd->snp_means) {
        for (int p = 0; p < gd->num_pops; p++) {
            free(gd->snp_means[p]);
        }
        free(gd->snp_means);
    }
    free(gd->block_start);
    free(gd->block_end);
    free(gd);
}

/*
 * Public API: read EIGENSTRAT genotype and individual files.  This
 * orchestrates parsing the .ind file to obtain individual/population
 * mappings and then reading the .geno file to compute per-population
 * allele frequencies.  The block_size parameter controls the
 * approximate number of SNPs per block for jackknife/robust
 * covariance estimation.  Returns a pointer to a newly allocated
 * GenoData on success or NULL on failure.  Caller must free the
 * returned structure via free_genodata().
 */
GenoData *read_eigenstrat(const char *geno_filename, const char *ind_filename, int block_size) {
    PopInfo *info = parse_ind_file(ind_filename);
    if (!info) return NULL;
    GenoData *gd = read_geno_means(geno_filename, info, block_size);
    free_popinfo(info);
    return gd;
}

/*
 * Public API: free a GenoData allocated by read_eigenstrat().
 */
void free_genodata(GenoData *gd) {
    free_genodata_internal(gd);
}

/* Function declarations for external linkage. */
GenoData *read_eigenstrat(const char *geno_filename, const char *ind_filename, int block_size);
void free_genodata(GenoData *gd);