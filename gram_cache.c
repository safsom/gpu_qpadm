/*
 * gram_cache.c - Gram matrix cache loader and f4 utilities
 *
 * This module reads the binary Gram cache produced by precompute_gpu.cu
 * and provides helper routines to compute f4 statistics on a per‑block
 * and average basis.  It also builds the design matrix F and target
 * vector b required by qpAdm and estimates the covariance matrix of
 * b via a jackknife over blocks.  The Gram cache stores one P×P
 * symmetric matrix per block in column‑major order, preceded by a
 * small header (see precompute_gpu.cu for details).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* header struct mirrors that in precompute_gpu.cu */
typedef struct {
    int num_pops;
    int num_blocks;
} GramHeader;

/* GramCache holds all Gram matrices in memory.  Each block is
 * contiguous of size P*P doubles in column-major order. */
typedef struct {
    int num_pops;
    int num_blocks;
    double *grams;
} GramCache;

/*
 * Load a Gram cache from disk into memory.  On success returns 0 and
 * populates the provided GramCache structure.  Caller must call
 * free_gram_cache() to release the allocated memory.  On failure
 * returns -1.
 */
int load_gram_cache(const char *filename, GramCache *cache) {
    if (!filename || !cache) return -1;
    memset(cache, 0, sizeof(*cache));
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open Gram cache %s\n", filename);
        return -1;
    }
    GramHeader hdr;
    if (fread(&hdr, sizeof(GramHeader), 1, fp) != 1) {
        fprintf(stderr, "Failed to read header from %s\n", filename);
        fclose(fp);
        return -1;
    }
    cache->num_pops = hdr.num_pops;
    cache->num_blocks = hdr.num_blocks;
    size_t total = (size_t)hdr.num_pops * hdr.num_pops * hdr.num_blocks;
    cache->grams = (double *)malloc(sizeof(double) * total);
    if (!cache->grams) {
        fprintf(stderr, "Failed to allocate Gram cache memory (%zu doubles)\n", total);
        fclose(fp);
        return -1;
    }
    size_t nread = fread(cache->grams, sizeof(double), total, fp);
    if (nread != total) {
        fprintf(stderr, "Expected %zu doubles in cache but read %zu\n", total, nread);
        free(cache->grams);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

/*
 * Free resources associated with a GramCache.
 */
void free_gram_cache(GramCache *cache) {
    if (!cache) return;
    free(cache->grams);
    cache->grams = NULL;
}

/*
 * Compute f4(W,X;Y,Z) for a single block.  The Gram matrix is
 * accessed in column‑major order.  See Patterson et al. 2012 for
 * interpretation.  f4 = p(W,Y) - p(W,Z) - p(X,Y) + p(X,Z), where
 * p(A,B) is the covariance across SNPs in the block.
 */
static inline double f4_block(const GramCache *cache, int block, int A, int B, int C, int D) {
    int P = cache->num_pops;
    const double *G = cache->grams + (size_t)block * P * P;
    /* column‑major indexing: G[i + j*P] = p(i,j) */
    double G_AC = G[A + C * P];
    double G_AD = G[A + D * P];
    double G_BC = G[B + C * P];
    double G_BD = G[B + D * P];
    return G_AC - G_AD - G_BC + G_BD;
}

/*
 * Build the design matrix F (dim k x j) and target vector b (dim k)
 * for qpAdm, averaged across all blocks.  The basepop is assumed to
 * appear twice in the f4 definition as described in the commentary:
 * F(k,j) = mean_b f4(source_j, base, right_k, base)
 * b(k)   = mean_b f4(target,   base, right_k, base)
 *
 * Inputs:
 *   cache       - pointer to loaded GramCache
 *   target      - index of target population
 *   base        - index of base population (B and D in f4)
 *   right_pops  - array of length k containing indices of outgroup populations
 *   k           - number of right pops
 *   sources     - array of length j containing indices of source populations
 *   j           - number of sources
 * Outputs:
 *   F           - double array of size k*j (row‑major) filled with F values
 *   b           - double array of size k filled with b values
 *
 * Returns 0 on success, -1 on error.
 */
int build_F_b(const GramCache *cache,
              int target,
              int base,
              const int *right_pops,
              int k,
              const int *sources,
              int j,
              double *F,
              double *b) {
    if (!cache || !right_pops || !sources || !F || !b) {
        return -1;
    }
    int nblocks = cache->num_blocks;
    /* initialize sums to zero */
    for (int kk = 0; kk < k; kk++) {
        b[kk] = 0.0;
        for (int jj = 0; jj < j; jj++) {
            F[kk * j + jj] = 0.0;
        }
    }
    /* accumulate sums across blocks */
    for (int bidx = 0; bidx < nblocks; bidx++) {
        for (int kk = 0; kk < k; kk++) {
            int R = right_pops[kk];
            /* b contribution */
            double val_b = f4_block(cache, bidx, target, base, R, base);
            b[kk] += val_b;
            for (int jj = 0; jj < j; jj++) {
                int S = sources[jj];
                double val_f = f4_block(cache, bidx, S, base, R, base);
                F[kk * j + jj] += val_f;
            }
        }
    }
    /* divide by number of blocks to get means */
    double inv = 1.0 / (double)nblocks;
    for (int kk = 0; kk < k; kk++) {
        b[kk] *= inv;
        for (int jj = 0; jj < j; jj++) {
            F[kk * j + jj] *= inv;
        }
    }
    return 0;
}

/*
 * Compute the covariance matrix of b (k x k) using a leave‑one‑block‑out
 * jackknife.  Given b_bk for each block and the mean b, the
 * jackknife estimator is:
 *   cov(k,l) = (n-1)/n * sum_b ( (b_bk[k] - b_mean[k]) * (b_bk[l] - b_mean[l]) )
 *
 * Inputs:
 *   cache      - loaded GramCache
 *   target     - target population index
 *   base       - base population index
 *   right_pops - array of length k of right populations
 *   k          - number of right populations
 * Output:
 *   cov        - array of size k*k (row‑major) filled with covariance
 * Returns 0 on success or -1 on error.
 */
int compute_b_covariance(const GramCache *cache,
                         int target,
                         int base,
                         const int *right_pops,
                         int k,
                         double *cov) {
    if (!cache || !right_pops || !cov) return -1;
    int nblocks = cache->num_blocks;
    if (nblocks < 2) {
        /* not enough blocks for jackknife */
        for (int ii = 0; ii < k * k; ii++) cov[ii] = 0.0;
        return 0;
    }
    /* allocate b_bk matrix */
    double *b_bk = (double *)malloc(sizeof(double) * nblocks * k);
    double *b_mean = (double *)calloc(k, sizeof(double));
    if (!b_bk || !b_mean) {
        fprintf(stderr, "Failed to allocate scratch memory for covariance\n");
        free(b_bk);
        free(b_mean);
        return -1;
    }
    /* compute b_bk for each block */
    for (int bidx = 0; bidx < nblocks; bidx++) {
        for (int kk = 0; kk < k; kk++) {
            int R = right_pops[kk];
            double val = f4_block(cache, bidx, target, base, R, base);
            b_bk[bidx * k + kk] = val;
            b_mean[kk] += val;
        }
    }
    /* compute mean */
    double inv_n = 1.0 / (double)nblocks;
    for (int kk = 0; kk < k; kk++) {
        b_mean[kk] *= inv_n;
    }
    /* initialize cov matrix */
    for (int i = 0; i < k * k; i++) {
        cov[i] = 0.0;
    }
    /* accumulate covariance */
    double factor = (double)(nblocks - 1) / (double)nblocks;
    for (int bidx = 0; bidx < nblocks; bidx++) {
        for (int kk = 0; kk < k; kk++) {
            double diff_k = b_bk[bidx * k + kk] - b_mean[kk];
            for (int ll = 0; ll < k; ll++) {
                double diff_l = b_bk[bidx * k + ll] - b_mean[ll];
                cov[kk * k + ll] += diff_k * diff_l;
            }
        }
    }
    for (int kk = 0; kk < k; kk++) {
        for (int ll = 0; ll < k; ll++) {
            cov[kk * k + ll] *= factor;
        }
    }
    free(b_bk);
    free(b_mean);
    return 0;
}

/* Forward declare functions for external linkage */
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