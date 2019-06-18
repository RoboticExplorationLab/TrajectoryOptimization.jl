using Test

model = Dynamics.doubleintegrator_model_uncertain
n = model.n; m = model.m; r = model.r

T = Float64

# costs
Q = 1.0*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0*Diagonal(I,m)

Qr = 1.0*Diagonal(I,n)
Qfr = 1.0*Diagonal(I,n)
Rr = 1.0*Diagonal(I,m)

x0 = [0; 0.]
xf = [1.0; 0]
D = Diagonal((0.2^2)*ones(r))
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,constraint_tolerance=1.0e-3)

N = 101
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:rk3_implicit,dt)

function bnd_func(c,x,u)
    for i = 1:m
        c[i] = u[i] - 5.0
        c[i+m] = -5.0 - u[i]
    end
    return nothing
end
bnd_con = Constraint{Inequality}(bnd_func,n,m,2m,:bound)

# bnd_con.c(zeros(2m),rand(n̄),rand(m+n^2))
# idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)),z=(1:n̄))
#
# f(ẋ,z) = prob.model.info[:fc](ẋ,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r))
# ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
# ∇f(x,u) = ∇f([x;u])
#
# function K(z,u)
#     x = z[idx.x]
#     s = z[idx.s]
#     P = reshape(s,n,n)*reshape(s,n,n)'
#     Bc = ∇f(x,u)[:,n .+ (1:m)]
#     R\(Bc'*P)
# end

con = ProblemConstraints([bnd_con],N)

prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), constraints=con, x0=x0, N=N, dt=dt)

# ilqr_solver_i = AbstractSolver(prob,opts_ilqr)
# jacobian!(prob,ilqr_solver_i)
# cost_expansion!(prob,ilqr_solver_i)
# ΔV = backwardpass!(prob,ilqr_solver_i)
# forwardpass!(prob,ilqr_solver_i,ΔV,J)
#
# model_d.f(zeros(n),rand(n),rand(m),rand(r),dt)
# model_d.∇f(zeros(n,n+m+r+1),rand(n),rand(m),rand(r),dt)
# F = PartedArray(zeros(T,n,length(prob.model)),create_partition2(prob.model))
# jacobian!(F,prob.model,zeros(n),zeros(m),1.0)

initial_controls!(prob,U0)
# rollout!(prob)
solve!(prob,opts_al)
plot(prob.X)
plot(prob.U)

prob_robust = robust_problem(prob,E1,H1,D,Qr,Rr,Qfr,Q,R,Qf,xf)

rollout!(prob_robust)

al_solver = AbstractSolver(prob_robust,opts_al)
solve!(prob_robust,al_solver)


uu = [prob_robust.U[k][1:m] for k = 1:N-1]
plot(prob.U)
plot!(uu)
max_violation(al_solver)
