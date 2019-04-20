using Test

## Pendulum
T = Float64

# model
model = TrajectoryOptimization.Dynamics.car_model
n = model.n; m = model.m
model_d = Model{Discrete}(model,rk4)

# cost
Q = Array(1e-3*Diagonal(I,n))
R = Array(1e-3*Diagonal(I,m))
Qf = Array(Diagonal(I,n)*100.0)
x0 = zeros(n)
xf = [0.0;1.0;0.0]
lqr_cost = LQRCost(Q,R,Qf,xf)

# problem
N = 31
U = [ones(m) for k = 1:N-1]
dt = 0.15
prob = Problem(model_d,ObjectiveNew(lqr_cost,N),U,dt=dt,x0=x0)
rollout!(prob)

## unconstrained
#bp
opts_ilqr = iLQRSolverOptions{T}()
ilqr_solver = AbstractSolver(prob,opts_ilqr)
jacobian!(prob,ilqr_solver)
ΔV = _backwardpass!(prob,ilqr_solver)

#sqrt bp
opts_ilqr_sqrt = iLQRSolverOptions{T}(square_root=true)
ilqr_solver_sqrt = AbstractSolver(prob,opts_ilqr)
jacobian!(prob,ilqr_solver_sqrt)
ΔV_sqrt = _backwardpass_sqrt!(prob,ilqr_solver_sqrt)
@test isapprox(ΔV_sqrt,ΔV)
@test all(isapprox(ilqr_solver.K,ilqr_solver_sqrt.K))
@test all(isapprox(ilqr_solver.d,ilqr_solver_sqrt.d))
for k = 1:N
    @test isapprox(ilqr_solver.S[k].xx,ilqr_solver_sqrt.S[k].xx'*ilqr_solver_sqrt.S[k].xx)
    @test isapprox(ilqr_solver.S[k].x,ilqr_solver_sqrt.S[k].x)
end

## constrained
opts_al = AugmentedLagrangianSolverOptions{T}(opts_uncon=opts_ilqr)
opts_altro = ALTROSolverOptions{T}(opts_al=opts_al)

# constraints
u_bnd = 5.
bnd = bound_constraint(n,m,u_min=-u_bnd,u_max=u_bnd,trim=true)
bnd
goal_con = goal_constraint(xf)
con = [bnd, goal_con]
prob = update_problem(prob,constraints=ProblemConstraints(con,N))
rollout!(prob)

#bp
solver_ilqr = AbstractSolver(prob,opts_ilqr)
solver_al = AbstractSolver(prob,opts_al)
prob_al = AugmentedLagrangianProblem(prob,solver_al)
update_constraints!(prob_al.obj.C,prob_al.obj.constraints, prob.X, prob.U)
update_active_set!(prob_al.obj)
jacobian!(prob_al,solver_ilqr)
ΔV = _backwardpass!(prob_al,solver_ilqr)

#bp sqrt
solver_ilqr_sqrt = AbstractSolver(prob,opts_ilqr)
solver_al_sqrt = AbstractSolver(prob,opts_al)
prob_al_sqrt = AugmentedLagrangianProblem(prob,solver_al_sqrt)
update_constraints!(prob_al_sqrt.obj.C,prob_al_sqrt.obj.constraints, prob_al_sqrt.X, prob_al_sqrt.U)
prob_al.obj.C[end]
update_active_set!(prob_al_sqrt.obj)
jacobian!(prob_al_sqrt,solver_ilqr_sqrt)
ΔV_sqrt = _backwardpass_sqrt!(prob_al_sqrt,solver_ilqr_sqrt)


@test isapprox(ΔV_sqrt,ΔV)
@test all(isapprox(solver_ilqr.K,solver_ilqr_sqrt.K))
@test all(isapprox(solver_ilqr.d,solver_ilqr_sqrt.d))
for k = 1:N
    @test isapprox(solver_ilqr.S[k].xx,solver_ilqr_sqrt.S[k].xx'*solver_ilqr_sqrt.S[k].xx)
    @test isapprox(solver_ilqr.S[k].x,solver_ilqr_sqrt.S[k].x)
end
