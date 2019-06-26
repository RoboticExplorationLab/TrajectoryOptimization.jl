# Solve with ALTRO
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1,0]
N = 51
n,m = model.n, model.m
bnd = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf], u_min=[0.1,-2], u_max=1.5)
bnd1 = BoundConstraint(n,m, u_min=bnd.u_min)
bnd_x = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf])
goal = goal_constraint(xf)
obs = (([0.2, 0.6], 0.25),
       ([-0.5, 0.5], 0.4))
obs1 = planar_obstacle_constraint(n,m, obs[1]..., :obstacle1)
obs2 = planar_obstacle_constraint(n,m, obs[2]..., :obstacle2)
con = ProblemConstraints(N)
con[1] += bnd1
for k = 2:N-1
    con[k] += bnd  + obs1 + obs2
end
con[N] += goal
prob = Problem(rk4(model), Objective(costfun, N), constraints=con, tf=3)
initial_controls!(prob, ones(m,N-1))
ilqr = iLQRSolverOptions()
al = AugmentedLagrangianSolverOptions(opts_uncon=ilqr)
al.constraint_tolerance = 1e-2
al.constraint_tolerance_intermediate = 1e-1
al.verbose = true
solve!(prob, al)
plot()
plot_circle!(obs[1]...)
plot_circle!(obs[2]...)
plot_trajectory!(prob.X,markershape=:circle)
plot(prob.U)
max_violation(prob)

# Create PN Solver
solver = ProjectedNewtonSolver(prob)
NN = N*n + (N-1)*m
p = num_constraints(prob)
P = N*n + sum(p)

# Test functions
dynamics_constraints!(prob, solver)
update_constraints!(prob, solver)
active_set!(prob, solver)
@test all(solver.a.primals)
@test all(solver.a.ν)
@test all(solver.a.λ[end-n+1:end])
@test !all(solver.a.λ)
dynamics_jacobian!(prob, solver)
@test solver.∇F[1].xx == solver.Y[1:n,1:n]
@test solver.∇F[2].xx == solver.Y[n .+ (1:n),1:n]
constraint_jacobian!(prob, solver)
solver.∇C[1]
@test solver.∇C[1] == solver.Y[N*n .+ (1:p[1]), 1:n+m]

cost_expansion!(prob, solver)
# Y,y = Array(Y), Array(y)

# Test Constraint Violation
solver = ProjectedNewtonSolver(prob)
solver.opts.active_set_tolerance = 0.0
dynamics_constraints!(prob, solver)
update_constraints!(prob, solver)
dynamics_jacobian!(prob, solver)
constraint_jacobian!(prob, solver)
@btime constraint_jacobian!($prob, $solver)
active_set!(prob, solver)
Y,y = active_constraints(prob, solver)
viol = calc_violations(solver)
@test maximum(maximum.(viol)) == norm(y,Inf)
@test norm(y,Inf) == max_violation(prob)
@test max_violation(solver) == max_violation(prob)

# Test Projection
solver = ProjectedNewtonSolver(prob)
solver.opts.feasibility_tolerance = 1e-10
solver.opts.active_set_tolerance = 1e-3
projection!(prob, solver)
update!(prob, solver, solver.V)
max_violation(solver)
multiplier_projection!(prob, solver)

# Build KKT
V0 = copy(solver.V)
cost_expansion!(prob, solver)
J0 = cost(prob, V)
res0 = norm(residual(prob, solver))
viol0 = max_violation(solver)
δV = solveKKT(prob, solver)
V_ = line_search(prob, solver, δV)

# Test Newton Step
solver = ProjectedNewtonSolver(prob)
solver.opts.feasibility_tolerance = 1e-10
solver.opts.verbose = false
V_ = newton_step!(prob, solver)
@btime newton_step!(prob, solver)
copyto!(solver.V.V, V_.V)
V_ = newton_step!(prob, solver)

update!(prob, solver)
Y,y = active_constraints(prob, solver)
@test length(y) == num_active_constraints(solver)
sum.(solver.active_set)
solver.active_set
inv(prob.obj[1].Q)*(N-1)
solver.∇C[1]

num_active_constraints(solver)
update!(prob, solver)
inds = jacobian_permutation(prob, solver)
@test sort(inds) == 1:length(inds)
a_perm = solver.a.duals[inds]
Y_perm = solver.Y[inds,:]
Y = Array(Y_perm[a_perm,:])
Hinv = inv(Diagonal(solver.H))
Qinv = [begin
            off = (k-1)*(n+m) .+ (1:n);
            Hinv[off,off];
        end for k = 1:N]
Rinv = [begin
            off = (k-1)*(n+m) .+ (n+1:n+m);
            Hinv[off,off];
        end for k = 1:N-1]
A = [F.xx for F in solver.∇F[2:end]]
B = [F.xu for F in solver.∇F[2:end]]
C = [Array(F.x[a,:]) for (F,a) in zip(solver.∇C, solver.active_set)]
D = [Array(F.u[a,:]) for (F,a) in zip(solver.∇C, solver.active_set)]
S0 = Y*Hinv*Y'
HY = Array(Hinv*Y')
B[1]
Y[4:6,1:8]
HY[:,7:8]
Y'[:,7:8]

S,L = buildShurCompliment(prob, solver)
Array(S0)[4:6,7:8]
Array(S)[4:6,7:8]
Array{Int}(S .≈ S0)
S ≈ S0

L0 = cholesky(Array(S)).L
Array{Int}(L .≈ L0)
@test L ≈ L0
@test L*L' ≈ S

len = ones(Int,2,N-1)*n
p = sum.(solver.active_set)
len[2,:] = p[1:end-1]
len = append!([1,3], vec(len))
push!(len, p[N])
lcum = cumsum(len)
[lcum[k]:lcum[k+1]-1 for k = 1:length(lcum)-1]



len
issymmetric()

num_constraints(prob)
sum(solver.a.duals)
length(solver.a)
length(solver.a.ν)

using Juno
using Profile
using InteractiveUtils
Profile.init(delay=1e-4)
@profiler newton_step!(prob, solver)


plot()
plot_circle!(obs[1]...)
plot_circle!(obs[2]...)
plot_trajectory!(prob.X,markershape=:circle)
plot_trajectory!(V_.X,markershape=:circle)
plot(prob.U, color=:blue)
plot!(V_.U, color=:red)
