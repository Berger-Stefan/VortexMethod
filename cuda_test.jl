using LinearAlgebra, CUDA, BenchmarkTools, Krylov, SparseArrays, CUDA.CUSPARSE

# CPU Arrays
A_cpu = sprand(20000, 20000, 0.3)
b_cpu = rand(20000)

@time A_cpu\b_cpu

# GPU Arrays
A_gpu = CuMatrix(A_cpu)
b_gpu = CuVector(b_cpu)

# Solve a rectangular and sparse system on GPU
# opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv!(y, P, x))
@time x, stats = cg(A_gpu, b_gpu)