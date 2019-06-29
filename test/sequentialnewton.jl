

solver0.C[N]
opts = ProjectedNewtonSolverOptions{Float64}()
solver = SequentialNewtonSolver(prob,opts)

dynamics_constraints!(prob, solver)
dynamics_jacobian!(prob, solver)
update_constraints!(prob, solver)
constraint_jacobian!(prob, solver)
active_set!(prob, solver)
cost_expansion!(prob, solver)
invert_hessian!(prob, solver)

solver = SequentialNewtonSolver(prob, opts)
update!(prob, solver)
solver0 = ProjectedNewtonSolver(prob)
update!(prob, solver0)

δV0 = solveKKT(prob, solver0)
Hinv = inv(Diagonal(solver0.H))
Y,y = active_constraints(prob, solver0)
λ0, λ0_ = solve_cholesky(prob, solver0, Qinv, Rinv, A, B, C, D)
r0 ≈ y - Y*inv(Diagonal(solver0.H))*solver0.g
r_ = Y*Hinv*solver0.g


p = sum.(solver.active_set)
y_part = ones(Int,2,N-1)*n
y_part[2,:] = p[1:end-1]
y_part = vec(y_part)
insert!(y_part,1,3)
push!(y_part, p[N])

δλ = S0\(y-Y*Hinv*g)

begin
    E = [@SMatrix zeros(n,n) for p in p_active]
    F = [@SMatrix zeros(n,p) for p in p_active]
    G = [cholesky(SMatrix{n,n}(1.0I)) for p in p_active]

    K = [@SMatrix zeros(p,n) for p in p_active]
    L = [@SMatrix zeros(p,p) for p in p_active]
    M = [@SMatrix zeros(p,n) for p in p_active]
    H = [cholesky(SMatrix{max(p,1),max(p,1)}(1.0I)) for p in p_active]

    r = [zeros(y) for y in y_part]
    λ_ = deepcopy(r)
    λ = deepcopy(r)

    a = [SVector{p_active[k],Int}(findall(solver.active_set[k])) for k = 1:N]
    vals = (E=E,F=F,G=G,K=K,L=L,M=M,H=H,
        r=r, λ_=λ_, λ=λ, a=a)
end
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

    vals = (E=E,F=F,G=G,K=K,L=L,M=M,H=H,
        r=r, λ_=λ_, λ=λ)
end
solver.∇C[1].x[vals.a[1],:]
solver.∇C[2].x[a[2],:]
λ0, λ0_,r0 = solve_cholesky(prob, solver0, Hinv, Qinv, Rinv, A, B, C, D)
λ,λ_,r = solve_cholesky(prob, solver, vals)
Profile.init(delay=1e-5)
@profiler solve_cholesky(prob, solver, vals)
vals.K[25]
size(vals.L[24])
vals.H[24]
p_active[24]

@btime solve_cholesky($prob, $solver0, $Hinv, $Qinv, $Rinv, $A, $B, $C, $D)
@btime solve_cholesky($prob, $solver, $vals)

vcat(λ_...) ≈ Array(λ0_)
vcat(λ...) ≈ Array(λ0)
vcat(r...) ≈ r0
solver.Q[2].u

a = [SVector{Int,p_active[k]}(findall(solver.active_set[k])) for k = 1:N]
p = num_constraints(prob)

vals.a[1]
A = rand(5,4)
A[vals.a[1],SVector(1,2,3)]

solver0.g
