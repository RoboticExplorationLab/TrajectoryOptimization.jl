using LinearAlgebra
using LDLFactorizations
using SparseArrays
using BenchmarkTools

n = 3
A = zeros(3,3)
A[1,1] = 9.0

B = 9.0*Matrix(I,n,n)

@benchmark cholesky($B)

function get_svd_sqrt(A::AbstractMatrix)
    x = svd(A)
    return Diagonal(sqrt.(x.S))*x.V'
end

function get_eig_sqrt(A::AbstractMatrix)
    x = eigen(A)
    return Diagonal(sqrt.(x.values))*x.vectors'
end

@benchmark get_svd_sqrt($B)
@benchmark get_eig_sqrt($B)
@benchmark ldlt(sparse($B); shift = 0.0, check = true, perm=Int[])




cholesky(B).U
C = SparseMatrixCSC(B)
D = sparse(B)
E = sparse(A)

F = ldlt(sparse(B); shift = 0.0, check = true, perm=Int[])


Array(sparse(cholesky(C)))

:
