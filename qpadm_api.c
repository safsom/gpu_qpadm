/*
 * qpadm_api.c - core qpAdm solving routines
 *
 * This module implements a simplified Generalised Least Squares
 * estimator for admixture proportions.  Given a design matrix F
 * (k x j), a target vector b (k), and an inverse covariance matrix
 * W_inv (k x k), we solve for mixture coefficients a subject to
 * sum(a)=1.  The estimator solves:
 *   minimise (b - F a)^T W_inv (b - F a)
 * subject to 1^T a = 1.
 *
 * We form the normal equations M a + lambda * 1 = f where
 * M = F^T W_inv F and f = F^T W_inv b.  The augmented system
 * [M  1; 1^T  0] [a] = [f; 1] is solved by Gaussian elimination.
 *
 * Standard errors are approximated from the inverse of M
 * (ignoring the constraint).  A more precise variance estimate
 * would require propagating the constraint through the covariance
 * structure of b and F; see Patterson et al. 2012 for details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* forward declarations */
static void multiply_matvec(const double *A, const double *x, double *y, int n);
static int invert_matrix(double *A, double *Ainv, int n);

/*
 * Solve the constrained GLS problem for admixture proportions.  F is
 * (k x j) row‑major, b is (k), and Winv is (k x k) row‑major
 * representing the inverse covariance of b.  On output, coeff
 * (length j) contains the estimated mixture proportions summing to 1.
 * se (length j) contains approximate standard errors.  Returns 0 on
 * success or -1 on failure.
 */
int solve_qpadm(int k, int j,
                const double *F,
                const double *b,
                const double *Winv,
                double *coeff,
                double *se) {
    if (!F || !b || !Winv || !coeff) return -1;
    /* allocate intermediate matrices */
    double *W_F  = (double *)calloc(k * j, sizeof(double));
    double *W_b  = (double *)calloc(k, sizeof(double));
    double *M    = (double *)calloc(j * j, sizeof(double));
    double *fvec = (double *)calloc(j, sizeof(double));
    if (!W_F || !W_b || !M || !fvec) {
        fprintf(stderr, "Memory allocation failure in solve_qpadm\n");
        free(W_F);
        free(W_b);
        free(M);
        free(fvec);
        return -1;
    }
    /* compute W_b = Winv * b and W_F = Winv * F */
    for (int r = 0; r < k; r++) {
        double sum_b = 0.0;
        for (int c = 0; c < k; c++) {
            sum_b += Winv[r * k + c] * b[c];
        }
        W_b[r] = sum_b;
        for (int jcol = 0; jcol < j; jcol++) {
            double sum_f = 0.0;
            for (int c = 0; c < k; c++) {
                sum_f += Winv[r * k + c] * F[c * j + jcol];
            }
            W_F[r * j + jcol] = sum_f;
        }
    }
    /* compute M = F^T * W_F and fvec = F^T * W_b */
    for (int i = 0; i < j; i++) {
        for (int jj = 0; jj < j; jj++) {
            double sum = 0.0;
            for (int r = 0; r < k; r++) {
                sum += F[r * j + i] * W_F[r * j + jj];
            }
            M[i * j + jj] = sum;
        }
        double sum_f = 0.0;
        for (int r = 0; r < k; r++) {
            sum_f += F[r * j + i] * W_b[r];
        }
        fvec[i] = sum_f;
    }
    /* build augmented system of size (j+1) */
    int dim = j + 1;
    double *Aaug = (double *)calloc(dim * dim, sizeof(double));
    double *yaug = (double *)calloc(dim, sizeof(double));
    if (!Aaug || !yaug) {
        fprintf(stderr, "Memory allocation failure in augmented system\n");
        free(W_F); free(W_b); free(M); free(fvec);
        free(Aaug); free(yaug);
        return -1;
    }
    /* fill top-left j×j block with M and top-right j×1 with ones */
    for (int i = 0; i < j; i++) {
        for (int jj = 0; jj < j; jj++) {
            Aaug[i * dim + jj] = M[i * j + jj];
        }
        Aaug[i * dim + j] = 1.0; /* lambda column */
        yaug[i] = fvec[i];
    }
    /* fill bottom row: ones transpose and zero */
    for (int jj = 0; jj < j; jj++) {
        Aaug[j * dim + jj] = 1.0;
    }
    Aaug[j * dim + j] = 0.0;
    yaug[j] = 1.0;
    /* solve augmented system by naive Gaussian elimination */
    /* convert to augmented matrix [A|y], dimension dim */
    int n = dim;
    /* forward elimination */
    for (int i = 0; i < n; i++) {
        /* pivot element */
        double pivot = Aaug[i * n + i];
        if (fabs(pivot) < 1e-12) {
            fprintf(stderr, "Singular matrix in qpAdm solve\n");
            free(W_F); free(W_b); free(M); free(fvec); free(Aaug); free(yaug);
            return -1;
        }
        /* normalise row */
        double inv_pivot = 1.0 / pivot;
        for (int col = i; col < n; col++) {
            Aaug[i * n + col] *= inv_pivot;
        }
        yaug[i] *= inv_pivot;
        /* eliminate below */
        for (int row = i + 1; row < n; row++) {
            double factor = Aaug[row * n + i];
            for (int col = i; col < n; col++) {
                Aaug[row * n + col] -= factor * Aaug[i * n + col];
            }
            yaug[row] -= factor * yaug[i];
        }
    }
    /* back substitution */
    double *xaug = (double *)calloc(n, sizeof(double));
    if (!xaug) {
        fprintf(stderr, "Memory allocation failure in back substitution\n");
        free(W_F); free(W_b); free(M); free(fvec); free(Aaug); free(yaug);
        return -1;
    }
    for (int row = n - 1; row >= 0; row--) {
        double sum = 0.0;
        for (int col = row + 1; col < n; col++) {
            sum += Aaug[row * n + col] * xaug[col];
        }
        xaug[row] = yaug[row] - sum;
    }
    /* extract coefficients (ignore lambda at index j) */
    for (int i = 0; i < j; i++) {
        coeff[i] = xaug[i];
    }
    /* approximate standard errors: invert M and take sqrt of diagonal */
    if (se) {
        double *Minv = (double *)calloc(j * j, sizeof(double));
        if (!Minv) {
            fprintf(stderr, "Memory allocation failure for Minv\n");
            /* but still return coeff */
        } else {
            if (invert_matrix(M, Minv, j) == 0) {
                for (int i = 0; i < j; i++) {
                    double var = Minv[i * j + i];
                    se[i] = (var > 0.0) ? sqrt(var) : 0.0;
                }
            } else {
                for (int i = 0; i < j; i++) se[i] = 0.0;
            }
            free(Minv);
        }
    }
    /* cleanup */
    free(W_F); free(W_b); free(M); free(fvec); free(Aaug); free(yaug); free(xaug);
    return 0;
}

/*
 * Multiply matrix A (n x n) by vector x (length n).  Row-major.
 */
static void multiply_matvec(const double *A, const double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

/*
 * Invert a symmetric positive definite matrix A (n x n) using
 * Gauss-Jordan elimination.  Both input A and output Ainv are
 * row-major.  Returns 0 on success or -1 on failure.  The input
 * matrix A is not modified.  This routine is not optimized for
 * speed; it is adequate for small j (<=10).
 */
static int invert_matrix(double *A, double *Ainv, int n) {
    /* create augmented matrix [A | I] of size n x 2n */
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
    /* forward elimination */
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
    /* extract inverse from augmented matrix */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ainv[i * n + j] = aug[i * dim + (n + j)];
        }
    }
    free(aug);
    return 0;
}

/* Function declaration for external linkage */
int solve_qpadm(int k, int j,
                const double *F,
                const double *b,
                const double *Winv,
                double *coeff,
                double *se);