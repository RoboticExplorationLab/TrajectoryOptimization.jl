## Pendulum
T = Float64

# model
model = TrajectoryOptimization.Dynamics.car
n = model.n; m = model.m
model_d = TO.discretize_model(model,:rk4)

# cost
Q = Array(1e-3*Diagonal(I,n))
R = Array(1e-3*Diagonal(I,m))
Qf = Array(Diagonal(I,n)*100.0)
x0 = zeros(n)
xf = [0.0;1.0;0.0]

# problem
N = 31
U = [ones(m) for k = 1:N-1]
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)
dt = 0.15
prob = Problem(model_d,obj,U,dt=dt,x0=x0)
rollout!(prob)

## unconstrained
#bp
opts_ilqr = iLQRSolverOptions{T}()
ilqr_solver = iLQRSolver(prob,opts_ilqr)
jacobian!(prob,ilqr_solver)
TO.cost_expansion!(prob,ilqr_solver)
ΔV = TO._backwardpass!(prob,ilqr_solver)

#sqrt bp
opts_ilqr_sqrt = iLQRSolverOptions{T}(square_root=true)
ilqr_solver_sqrt = iLQRSolver(prob,opts_ilqr_sqrt)
jacobian!(prob,ilqr_solver_sqrt)
TO.cost_expansion!(prob,ilqr_solver_sqrt)
ΔV_sqrt = TO._backwardpass_sqrt!(prob,ilqr_solver_sqrt)
@test isapprox(ΔV_sqrt,ΔV)
@test all(isapprox(ilqr_solver.K,ilqr_solver_sqrt.K))
@test all(isapprox(ilqr_solver.d,ilqr_solver_sqrt.d))
for k = 1:N
    @test isapprox(ilqr_solver.S[k].xx,ilqr_solver_sqrt.S[k].xx'*ilqr_solver_sqrt.S[k].xx)
    @test isapprox(ilqr_solver.S[k].x,ilqr_solver_sqrt.S[k].x)
end

## constrained
opts_al = AugmentedLagrangianSolverOptions{T}(opts_uncon=opts_ilqr)
opts_al_sqrt = AugmentedLagrangianSolverOptions{T}(opts_uncon=opts_ilqr_sqrt)

# constraints
u_bnd = 5.
bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd,trim=true)
bnd
goal_con = goal_constraint(xf)
con = [bnd, goal_con]
prob = update_problem(prob,constraints=Constraints(con,N))
rollout!(prob)

#bp
solver_ilqr = iLQRSolver(prob,opts_ilqr)
solver_al = AugmentedLagrangianSolver(prob,opts_al)
prob_al = TO.AugmentedLagrangianProblem(prob,solver_al)
TO.update_constraints!(prob_al.obj.C,prob_al.obj.constraints, prob.X, prob.U)
TO.update_active_set!(prob_al.obj)
jacobian!(prob_al,solver_ilqr)
TO.cost_expansion!(prob_al,solver_ilqr)
ΔV = TO._backwardpass!(prob_al,solver_ilqr)

#bp sqrt
solver_al_sqrt = AugmentedLagrangianSolver(prob,opts_al_sqrt)
solver_ilqr_sqrt = solver_al_sqrt.solver_uncon
prob_al_sqrt = TO.AugmentedLagrangianProblem(prob,solver_al_sqrt)
TO.update_constraints!(prob_al_sqrt.obj.C,prob_al_sqrt.obj.constraints, prob_al_sqrt.X, prob_al_sqrt.U)
TO.update_active_set!(prob_al_sqrt.obj)
jacobian!(prob_al_sqrt,solver_ilqr_sqrt)
TO.cost_expansion!(prob_al_sqrt,solver_ilqr_sqrt)
ΔV_sqrt = TO._backwardpass_sqrt!(prob_al_sqrt,solver_ilqr_sqrt)

@test isapprox(ΔV_sqrt,ΔV)
@test all(isapprox(solver_ilqr.K,solver_ilqr_sqrt.K))
@test all(isapprox(solver_ilqr.d,solver_ilqr_sqrt.d))
for k = 1:N
    @test isapprox(solver_ilqr.S[k].xx,solver_ilqr_sqrt.S[k].xx'*solver_ilqr_sqrt.S[k].xx)
    @test isapprox(solver_ilqr.S[k].x,solver_ilqr_sqrt.S[k].x)
end
