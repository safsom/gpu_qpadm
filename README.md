# gpu_qpadm

GPU-accelerated precomputation and solving routines for qpAdm/qpWave-style analyses.
This implementation reads EIGENSTRAT-formatted genotype data, computes blockwise Gram matrices on the GPU (cuBLAS),
and solves a constrained GLS system for admixture coefficients.

---

## Repository layout

```
gpu_qpadm/gram_cache.c
gpu_qpadm/io.c
gpu_qpadm/main.c
gpu_qpadm/precompute_gpu.cu
gpu_qpadm/qpadm_api.c
```

- **gpu_qpadm/precompute_gpu.cu** — CUDA/cuBLAS precomputation of blockwise Gram matrices \(G_b = X_b^T X_b\).
- **gpu_qpadm/io.c** — Minimal EIGENSTRAT (.geno/.ind) reader and per-population allele-frequency aggregator; SNPs are partitioned into jackknife blocks.
- **gpu_qpadm/gram_cache.c** — Binary Gram-cache loader and utilities to derive f4 statistics, design matrix \(F\), target vector \(b\), and jackknife covariance.
- **gpu_qpadm/qpadm_api.c** — Constrained GLS solver for qpAdm (solves \(\min_a (b - Fa)^T W^{-1} (b - Fa)\) s.t. \(\sum a_i = 1\)).
- **gpu_qpadm/main.c** — CLI wrapper exposing `precompute` and `qpadm` subcommands.

---

## Build

**Requirements**
- A C/C++ toolchain (GCC/Clang) with C99 support
- **CUDA Toolkit** (for `nvcc`) and **cuBLAS**
- POSIX environment (Linux/macOS). Windows users: build under WSL or adapt the Makefile.

**Example (manual)**
```bash
# Compile CPU objects
cc -O3 -march=native -std=c99 -c gpu_qpadm/io.c -o io.o
cc -O3 -march=native -std=c99 -c gpu_qpadm/gram_cache.c -o gram_cache.o
cc -O3 -march=native -std=c99 -c gpu_qpadm/qpadm_api.c -o qpadm_api.o
cc -O3 -march=native -std=c99 -c gpu_qpadm/main.c -o main.o

# Compile CUDA object
nvcc -O3 -Xcompiler -fPIC -lcublas -c gpu_qpadm/precompute_gpu.cu -o precompute_gpu.o

# Link
cc -O3 -o gpu_qpadm_bin main.o io.o gram_cache.o qpadm_api.o precompute_gpu.o -lcublas -lcudart -lm
```

Adjust include/library paths if your CUDA is not in the default location (e.g., add `-I/opt/cuda/include -L/opt/cuda/lib64`).

---

## Usage

Two subcommands are exposed (see `gpu_qpadm/main.c`).

### 1) Precompute Gram matrices on GPU
```bash
./gpu_qpadm_bin precompute <data.geno> <data.ind> <block_size> <cache.bin>
```
- `block_size` \approx SNPs per jackknife block.
- The cache stores `num_blocks` symmetric \(P\times P\) Gram matrices in column-major order with a small header.

**Example**
```bash
./gpu_qpadm_bin precompute data.geno data.ind 50000 data.cache
```

### 2) Run qpAdm on a precomputed cache
```bash
./gpu_qpadm_bin qpadm <cache.bin> <target_idx> <base_idx> <right_pops_csv> <sources_csv>
```
- `<right_pops_csv>`: comma-separated indices of outgroups.
- `<sources_csv>`: comma-separated indices of candidate sources.
- Prints mixture coefficients and approximate standard errors.

**Example**
```bash
./gpu_qpadm_bin qpadm data.cache 3 0 4,5,6 1,2
```

Population indices correspond to the order induced by the `.ind` file parser in `io.c`.

---

## Data model & math (high level)

For each block \(b\) with \(m_b\) SNPs and \(P\) populations, the allele-frequency matrix is \(X_b \in \mathbb{R}^{m_b \times P}\).
The GPU computes \(G_b = X_b^T X_b\) (symmetric \(P\times P\)). From \{G_b\} we derive blockwise f4-statistics,
assemble the design matrix \(F\in\mathbb{R}^{k\times j}\) and the target vector \(b\in\mathbb{R}^k\) (where \(k\)=#right-pops, \(j\)=#sources),
and estimate a jackknife covariance \(W\in\mathbb{R}^{k\times k}\). qpAdm solves a constrained GLS for admixture weights \(a\in\mathbb{R}^j\) with \(\sum a_i = 1\).

---

## Time complexity (from code inspection)

Let
- \(M\) = total SNPs, \(B\) = number of blocks, \(m_b\) = SNPs in block \(b\), with \(\sum_b m_b = M\)
- \(N\) = individuals, \(P\) = populations, \(k\) = right pops, \(j\) = sources

**I/O & aggregation (`io.c`)**
- Reading `.ind` and mapping individuals→populations: \(\mathcal O(N)\).
- Reading `.geno` and computing per-population means per SNP: \(\mathcal O(M\cdot N)\).
- Block partitioning bookkeeping: \(\mathcal O(B)\).
**Total**: \(\mathcal O(MN)\).

**GPU Gram precompute (`precompute_gpu.cu`)**
- For each block, one DGEMM: \(X_b^T X_b\) multiplies \(P\times m_b\) by \(m_b\times P\).
  FLOPs per block: \(\Theta(P^2 m_b)\).
**Total**: \(\Theta(P^2 M)\).

**f4 / design / jackknife (`gram_cache.c`)**
- Build \(F\) and \(b\): scans all blocks with constant-time lookups per \((k,j)\) pair.  \(\mathcal O(B\cdot (k + kj)) = \mathcal O(Bkj)\).
- Jackknife covariance over \(k\) statistics across \(B\) blocks: \(\mathcal O(Bk^2)\).

**qpAdm solve (`qpadm_api.c`)**
- Form \(W^{-1}F\) and \(W^{-1}b\): \(\mathcal O(k^2 j + k^2)\).
- Form \(M = F^T W^{-1} F\): \(\mathcal O(k j^2)\).
- Solve augmented \((j+1)\times (j+1)\) system by Gaussian elimination: \(\mathcal O(j^3)\).

**Memory**
- Gram cache: \(B\cdot P^2\) doubles (8 bytes each).
- Working sets dominated by \(P^2\) for a single Gram and \(k^2, kj\) for qpAdm.

---

## Citation / provenance

This code is a compact reimplementation designed for clarity and GPU precompute; it mirrors qpAdm/qpWave workflows (f4, jackknife covariance, constrained GLS) while reducing legacy I/O surface area.
