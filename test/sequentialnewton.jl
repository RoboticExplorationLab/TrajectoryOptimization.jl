
# Create solver
opts = ProjectedNewtonSolverOptions{Float64}()
solver = SequentialNewtonSolver(prob,opts)

# Test update functions
dynamics_constraints!(prob, solver)
dynamics_jacobian!(prob, solver)
update_constraints!(prob, solver)
constraint_jacobian!(prob, solver)
active_set!(prob, solver)
cost_expansion!(prob, solver)
invert_hessian!(prob, solver)

# Init solvers
solver = SequentialNewtonSolver(prob, opts)
update!(prob, solver)
solver0 = ProjectedNewtonSolver(prob)
update!(prob, solver0)

# Compare KKT solves
δV0 = solveKKT(prob, solver0)
Hinv = inv(Diagonal(solver0.H))
Y,y = active_constraints(prob, solver0)
# λ0, λ0_ = solve_cholesky(prob, solver0, Qinv, Rinv, A, B, C, D)
r0 = y - Y*Hinv*solver0.g
λ0 = (Y*Hinv*Y')\r0



p_active = sum.(solver.active_set)
y_part = ones(Int,2,N-1)*n
y_part[2,:] = p_active[1:end-1]
y_part = vec(y_part)
insert!(y_part,1,3)
push!(y_part, p[N])
NN = N*n + (N-1)*m

begin
    E = [zeros(n,n) for p in p_active]
    F = [zeros(n,p) for p in p_active]
    G = [cholesky(Matrix(I,n,n)) for p in p_active]

    K = [zeros(p,n) for p in p_active]
    L = [zeros(p,p) for p in p_active]
    M = [zeros(p,n) for p in p_active]
    H = [cholesky(Matrix(I,p,p)) for p in p_active]

    r = [zeros(y) for y in y_part]
    λ_ = deepcopy(r)
    λ = deepcopy(r)
    z = [zeros(n+m*(k<N)) for k = 1:N]
    x,u = [zeros(n) for k = 1:N], [zeros(m) for k = 1:N-1]

    vals = (E=E,F=F,G=G,K=K,L=L,M=M,H=H,
        r=r, λ_=λ_, λ=λ, z=z, x=x, u=u)
end


# Solve KKT system
solver0 = ProjectedNewtonSolver(prob, opts)
update!(prob, solver0)
δV0 = solveKKT(prob, solver0)
δV0[NN+1:end][solver0.a.duals] ≈ λ0

solver = SequentialNewtonSolver(prob, opts)
update!(prob, solver)
δx,δu,δλ = solveKKT(prob, solver, vals)
δz = [[x;u] for (x,u) in zip(δx[1:N-1], δu)]
push!(δz,δx[N])
vcat(δz...) ≈ -δV0[1:NN]
vcat(δλ...) ≈ δV0[NN+1:end][solver0.a.duals]

solver.V.X .+ δx


# Test projection
active_constraints!(prob, solver, vals.r)
vcat(vals.r...) ≈ y
λ,λ_,r = solve_cholesky(prob, solver, vals, vals.r)
vcat(λ...) ≈ (Y*Hinv*Y')\y

solver0 = ProjectedNewtonSolver(prob)
solver0.opts.feasibility_tolerance = 1e-10
update!(prob, solver0)
projection!(prob, solver0)
max_violation(solver0)

solver = SequentialNewtonSolver(prob, opts)
solver.opts.feasibility_tolerance = 1e-10
solver.opts.verbose = true
update!(prob, solver)
projection!(prob, solver, vals)
max_violation(solver) - max_violation(solver0) < 1e-16


# Residual
residual!(prob, solver, vals)
res_norm(prob, solver, vals)
norm([vals.x, vals.u])
norm(residual(prob, solver0))
@btime sqrt(norm($vals.x)^2 + norm($vals.u)^2)
@btime norm([norm($vals.x),norm($vals.u)])


# Multiplier projection
solver0 = ProjectedNewtonSolver(prob)
update!(prob, solver0)
Y,y = active_constraints(prob, solver0)
λ0 = duals(solver0.V)[solver0.a.duals]
res0 = solver0.g + Y'λ0
-(Y*Y')\(Y*res0)


solver = SequentialNewtonSolver(prob, opts)
update!(prob, solver)
vals.λ .*= 0
residual!(prob, solver, vals)
z = [[x;u] for (x,u) in zip(vals.x[1:N-1], vals.u)]
push!(z,vals.x[N])
vcat(z...) ≈ solver0.g + Y'λ0

jac_mult!(prob, solver, vals.x, vals.u, vals.r)
vcat(vals.r...) ≈ Y*(solver0.g + Y'λ0)

eyes = [I for k = 1:N]
δλ, = solve_cholesky(prob, solver, vals, vals.r, eyes, eyes)
vcat(δλ...) ≈ (Y*Y')\(Y*(solver0.g+Y'λ0))
vcat(δλ...)


# Compare Multiplier Projection
solver0 = ProjectedNewtonSolver(prob)
update!(prob, solver0)
δλ0 = multiplier_projection!(prob, solver0)
duals(solver0.V)

solver = SequentialNewtonSolver(prob, opts)
update!(prob, solver)
vals.λ .*= 0
δλ = multiplier_projection!(prob, solver, vals)
vcat(δλ...) ≈ -δλ0
duals(solver.V) ≈ duals(solver0.V)



# Test full newton step
solver0 = ProjectedNewtonSolver(prob,opts)
newton_step!(prob, solver0)

solver = SequentialNewtonSolver(prob, opts)
newton_step!(prob, solver, vals)









@btime res_norm($prob, $solver, $vals)
@btime norm([$vals.x; $vals.u])
@btime norm(residual($prob, $solver0))

jac_mult!(prob, solver, vals.λ, vals.x, vals.u)
solver.Qinv .* vals.x
solver.Rinv .* vals.u

V0 = PrimalDual(prob)
solver0.opts.verbose = false
solver.opts.verbose = false
@btime begin
    copyto!($solver0.V.V, $V0.V)
    projection!($prob, $solver0)
end
@btime begin
    copyto!($solver.V.V, $V0.V)
    projection!($prob, $solver, $vals)
end

solver0.Y.blocks
solver0.a.duals
solver0.Y.blocks[solver0.a.duals,:]
solver0.y.blocks[solver0.a.duals]

@btime projection!($prob, $solver, $vals)
@btime begin
    Y,y = active_constraints(prob, solver0)
    HinvY = $Hinv*Y'
    HinvY*((Y*HinvY)\y)
end
z = [[vals.x[k]; vals.u[k]] for k = 1:N-1]
push!(z,vals.x[N])
vcat(z...) ≈ Hinv*Y'*((Y*Hinv*Y')\y)


C = [solver.∇C[k][a[k],1:n] for k = 1:N]
D = [solver.∇C[k][a[k],4:5] for k = 1:N-1]
a = solver.active_set
part_z = (x=1:n,u=n+1:n+m)
ai = [collect(1:p) for p in num_constraints(prob)]
@btime C2 = [view($solver.∇C[k],$a[k],:) for k = 1:N]
for k = 1:N
    C2[k].indices[1] = findall(a[k])
end


struct KKTFactors{T}
    E::MatrixTrajectory{T}
    F::MatrixTrajectory{T}
    G::Vector{Cholesky{T,Matrix{T}}}
    K::MatrixTrajectory{T}
    L::MatrixTrajectory{T}
    M::MatrixTrajectory{T}
    H::Vector{Cholesky{T,Matrix{T}}}
    y_part::Vector{Int}
end

function KKTFactors(n::Int, m::Int, p::Vector{Int}, N::Int)
    E = [zeros(n,n) for p in p_active]
    F = [zeros(n,p) for p in p_active]
    G = [cholesky(Matrix(I,n,n)) for p in p_active]

    K = [zeros(p,n) for p in p_active]
    L = [zeros(p,p) for p in p_active]
    M = [zeros(p,n) for p in p_active]
    H = [cholesky(Matrix(I,p,p)) for p in p_active]

    y_part = ones(Int,2,N-1)*n
    y_part[2,:] = p[1:end-1]
    y_part = vec(y_part)
    insert!(y_part,1,3)
    push!(y_part, p[N])

    KKTFactors(E,F,G,K,L,M,H,y_part)
end

struct KKTJacobian{T}
    ∇F::Vector{PartedArray{T,2,Matrix{T},P}} where P
    ∇C::Vector{PartedArray{T,2,Matrix{T},P} where P}
    active_set::Vector{Vector{Bool}}
end

function KKTJacobian(prob::Problem)
    n,m,N = size(prob)
    part_f = create_partition2(prob.model)
    constraints = prob.constraints

    ∇F = [PartedMatrix(zeros(n,n+m+1),part_f) for k = 1:N]
    ∇C = [PartedMatrix(con,n,m) for con in constraints.C]
    ∇C[N] = PartedMatrix(constraints[N],n,m,:terminal)
    active_set = [ones(Bool,pk) for pk in p]

    KKTJacobian(∇F, ∇C, active_set)
end

function *(Y::KKTJacobian, r::AbstractVector{<:AbstractVector})
    N = length(Y.∇F)

    for k = 1:N
