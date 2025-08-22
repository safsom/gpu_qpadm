/*
 * precompute_gpu.cu - GPU precomputation of Gram matrices for qpAdm
 *
 * This module constructs blockwise population‑wise Gram matrices
 * G_b = X_b^T X_b from the per‑population allele frequency matrix
 * X_b (m_b x P).  Each Gram matrix is symmetric of size P x P and
 * accumulates the contributions of m_b SNPs in a jackknife block.
 *
 * The heavy lifting is offloaded to the GPU via cuBLAS DGEMM.  We
 * allocate device memory for each block, copy the column‑major
 * genotype means, perform a single DGEMM call and copy the result
 * back to the host.  All Gram matrices are appended to a binary
 * cache file along with a simple header encoding the dimensions.
 *
 * Note: error handling is minimal—production code should check
 * return codes more carefully and handle allocation failures.  This
 * implementation assumes the GPU has enough memory to handle
 * reasonably sized blocks (m_b × P).  If blocks are extremely
 * large, consider streaming the computation in tiles.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* bring in GenoData definition from io.c */
extern "C" {
    typedef struct GenoData GenoData;
    struct GenoData {
        int num_snps;
        int num_inds;
        int num_pops;
        int num_blocks;
        int *block_start;
        int *block_end;
        double **snp_means;
    };
}

/* simple header for the gram cache file */
typedef struct {
    int num_pops;
    int num_blocks;
} GramHeader;

/* CUDA error checking macro */
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

/* cuBLAS error checking macro */
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)stat, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

/*
 * Compute a single Gram matrix for block b and write it to the file
 * descriptor.  Returns 0 on success or -1 on error.
 */
static int process_block_gpu(const GenoData *gd, int b, FILE *fp, cublasHandle_t handle) {
    int start = gd->block_start[b];
    int end   = gd->block_end[b];
    int m = end - start;
    int P = gd->num_pops;
    /* allocate host column‑major matrix X (m x P) */
    double *hX = (double *)malloc(sizeof(double) * m * P);
    if (!hX) {
        fprintf(stderr, "Failed to allocate host X for block %d\n", b);
        return -1;
    }
    /* fill X such that X(row, col) = snp_means[col][start + row] */
    for (int col = 0; col < P; col++) {
        for (int row = 0; row < m; row++) {
            hX[row + col * m] = gd->snp_means[col][start + row];
        }
    }
    /* allocate device memory */
    double *dX = NULL;
    double *dG = NULL;
    CHECK_CUDA(cudaMalloc((void **)&dX, sizeof(double) * m * P));
    CHECK_CUDA(cudaMalloc((void **)&dG, sizeof(double) * P * P));
    /* copy to device */
    CHECK_CUDA(cudaMemcpy(dX, hX, sizeof(double) * m * P, cudaMemcpyHostToDevice));
    /* compute G = X^T * X using cuBLAS (G and X are column‑major) */
    const double alpha = 1.0;
    const double beta  = 0.0;
    /* A = X^T has dims P x m, B = X has dims m x P, C = G dims P x P */
    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             P, P, m,
                             &alpha,
                             dX, m,
                             dX, m,
                             &beta,
                             dG, P));
    /* copy back to host */
    double *hG = (double *)malloc(sizeof(double) * P * P);
    if (!hG) {
        fprintf(stderr, "Failed to allocate host G for block %d\n", b);
        cudaFree(dX);
        cudaFree(dG);
        free(hX);
        return -1;
    }
    CHECK_CUDA(cudaMemcpy(hG, dG, sizeof(double) * P * P, cudaMemcpyDeviceToHost));
    /* write G to file in binary, preserving column-major order */
    size_t written = fwrite(hG, sizeof(double), P * P, fp);
    if (written != (size_t)(P * P)) {
        fprintf(stderr, "Failed to write Gram matrix for block %d\n", b);
        cudaFree(dX);
        cudaFree(dG);
        free(hX);
        free(hG);
        return -1;
    }
    /* cleanup */
    cudaFree(dX);
    cudaFree(dG);
    free(hX);
    free(hG);
    return 0;
}

/*
 * Public API: precompute all Gram matrices for a given GenoData and
 * write them to a cache file.  The file is created/truncated and
 * contains a header followed by num_blocks consecutive Gram matrices.
 * Each Gram matrix is stored in column-major order as doubles.
 * Returns 0 on success or -1 on failure.
 */
int precompute_gram_gpu(const GenoData *gd, const char *cache_filename) {
    if (!gd || !cache_filename) {
        return -1;
    }
    FILE *fp = fopen(cache_filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open cache file %s for writing\n", cache_filename);
        return -1;
    }
    /* write header */
    GramHeader hdr;
    hdr.num_pops = gd->num_pops;
    hdr.num_blocks = gd->num_blocks;
    if (fwrite(&hdr, sizeof(GramHeader), 1, fp) != 1) {
        fprintf(stderr, "Failed to write header to %s\n", cache_filename);
        fclose(fp);
        return -1;
    }
    /* set up cuBLAS */
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    /* iterate over blocks */
    for (int b = 0; b < gd->num_blocks; b++) {
        if (process_block_gpu(gd, b, fp, handle) != 0) {
            fprintf(stderr, "Error processing block %d\n", b);
            cublasDestroy(handle);
            fclose(fp);
            return -1;
        }
    }
    cublasDestroy(handle);
    fclose(fp);
    return 0;
}

/* Function declaration for external linkage */
int precompute_gram_gpu(const GenoData *gd, const char *cache_filename);