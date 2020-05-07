using StaticArrays
using LinearAlgebra
using Test

n,m = 10,5
Qd = @SVector rand(n)
Rd = @SVector rand(m)
H = @SMatrix zeros(m,n)
q = @SVector rand(n)
r = @SVector rand(m)
c = randn()

# Test constructors
qc = DiagonalCost(Qd,Rd)
@test qc.q == zeros(n)
@test qc.r == zeros(m)
@test qc.Q isa Diagonal{Float64, SVector{n,Float64}}

qc = DiagonalCost(Diagonal(Qd), Diagonal(Rd))
@test qc.q == zeros(n)
@test qc.r == zeros(m)
@test qc.Q isa Diagonal{Float64, SVector{n,Float64}}

qc = DiagonalCost(Matrix(Diagonal(Qd)), Matrix(Diagonal(Rd)))
@test qc.q == zeros(n)
@test qc.r == zeros(m)
@test qc.Q isa Diagonal{Float64, SVector{n,Float64}}
@test qc.R isa Diagonal{Float64, SVector{m,Float64}}

qc = DiagonalCost(Qd, Rd, q, r, c)
@test qc.q == q
@test qc.r == r

qc = DiagonalCost(Diagonal(Qd), Diagonal(Rd), q, r, c)
@test qc.q == q
@test qc.r == r

qc = DiagonalCost(Matrix(Diagonal(Qd)), Matrix(Diagonal(Rd)), q, r, c)
@test qc.q == q
@test qc.r == r

qc = DiagonalCost(Matrix(Diagonal(Qd)), Matrix(Diagonal(Rd)), Vector(q), Vector(r), c)
@test qc.q == q
@test qc.r == r
@test qc.q isa SVector{n}
@test qc.r isa SVector{m}

qc = DiagonalCost(Matrix(Diagonal(Qd)), Matrix(Diagonal(Rd)), H, Vector(q), Vector(r), c)
@test qc.q == q
@test qc.r == r
@test qc.q isa SVector{n}
@test qc.r isa SVector{m}

qc = DiagonalCost(Matrix(Diagonal(Qd)), Matrix(Diagonal(Rd)), q=Vector(q))
@test qc.q == q
@test qc.r == zeros(m)
@test qc.c == 0

qc = DiagonalCost(Qd, Rd, q=q)
@test qc.q == q
@test qc.r == zeros(m)
@test qc.c == 0
