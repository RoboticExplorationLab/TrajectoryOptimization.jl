# Set up problem
using TrajectoryOptimization: to_array, update_constraints!, calculate_jacobians!, outer_loop_update
using LinearAlgebra
using Plots
using Test
using Logging

u_bound = 5.
model, obj = TrajectoryOptimization.Dynamics.pendulum!
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false

obj.Q = 1e-3*Diagonal(I,2)
obj.R = 1e-2*Diagonal(I,1)
obj.tf = 2.
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective
obj_min = update_objective(obj_c,tf=:min,c=1e-2, Q = obj.Q*1, R = obj.R, Qf = obj.Qf*0.)
dt = 0.1
n,m = model.n, model.m

solver = Solver(model,obj_c,integration=:rk4,dt=dt)
U = ones(m,solver.N)
results,stats = solve(solver,U)
plot(to_array(results.X)')
plot(to_array(results.U)')

solver_min = Solver(model,obj_min,integration=:rk3_foh,N=41)
U = ones(m,solver_min.N)
solver_min.opts.verbose = true
solver_min.opts.use_static = false
solver_min.opts.max_dt = 0.2
solver_min.opts.constraint_tolerance = 1e-2
solver_min.opts.R_minimum_time = 1000
solver_min.opts.ρ_initial = 0
solver_min.opts.τ = .25
solver_min.opts.γ = 10.0
results_min,stats_min = solve(solver_min,U)
total_time(solver_min,results_min)
plot(to_array(results_min.X)[1:2,:]')
plot(to_array(results_min.U)[1:2,:]')
plot(stats_min["cost"])

TrajectoryOptimization.cost_mintime(solver_min,results_min)
max_violation(results_min)
results_min.CN
results_min.C[3]
results_min.U[1][2]
get_num_constraints(solver_min)

violations = map((x)->x.>0, results_min.Iμ) .* results_min.C
norm.(violations,Inf)

norm(to_array(violations),Inf)
max_violation(results_min)
argmax(abs.(to_array(violations)))




# Test dynamics change with dt
solver = Solver(model,obj_c,dt=dt)
solver_min = Solver(model,obj_min,N=solver.N)
@test get_num_controls(solver) == (1,1)
@test get_num_controls(solver_min) == (2,2)

x = rand(n)
u = rand(m)
S = [x;u;√dt]
xdot = zero(x)
Sdot = zero(S)

f = model.f
fd! = rk4(f,dt)
fd_aug = rk4(f_augmented!(f,n,m))
fd!(xdot,x,u)
fd_aug(Sdot,S)

fS = zeros(n+m+1,n+m+1)
ForwardDiff.jacobian!(fS,fd_aug,Sdot,S)

dft = zeros(n,1)
ft(dx,t) = fd!(dx,x,u,t[1])
ForwardDiff.jacobian!(dft,ft,xdot,[dt])

@test ~isapprox(dft,fS[1:n,end])  # make sure the augmented time jacobian is not the same as the not augmented, since S stores the square root of dt (for mintime problems)

u_dt = [u;sqrt(dt)]
xdot = zeros(n)
xdot0 = zeros(n)
solver_min.fd(xdot,x,u_dt,u_dt[end]^2)
solver.fd(xdot0,x,u)
fx,fu = solver_min.Fd(x,u_dt)
fx0,fu0 = solver.Fd(x,u)
@test xdot == xdot0
@test fx0 == fS[1:n,1:n]
@test fu0 ≈ fS[1:n,n+1]
@test fx == fx0
@test fu[:,1] ≈ fu0
@test fu ≈ fS[1:n,n+1:end]

# Change dt and make sure they are no longer equal
u_dt = [u;0.15]
solver_min.fd(xdot,x,u_dt,u_dt[end])
solver.fd(xdot0,x,u)
fx,fu = solver_min.Fd(x,u_dt)
fx0,fu0 = solver.Fd(x,u)
@test xdot ≉ xdot0
@test fx ≉ fx0
@test fu[:,1] ≉ fu0

# Run through parts of the iLQR algorithm
solver = Solver(model,obj_c,dt=dt)
solver_min = Solver(model,obj_min,N=solver.N)
n,m,N = get_sizes(solver)
p,pI,pE = TrajectoryOptimization.get_num_constraints(solver)
res_reg = ConstrainedVectorResults(n,m,p,N)
res_min = ConstrainedVectorResults(n,m+1,p+3,N)

res_reg.X[1] = obj.x0
res_min.X[1] = obj.x0

copyto!(res_reg.U, ones(m,N))
copyto!(res_min.U, [ones(m,N); ones(m,N)*sqrt(dt)])
@test to_array(res_min.U)[1,:] == vec(to_array(res_reg.U))

# Test that rollouts are the same
rollout!(res_reg, solver)
rollout!(res_min, solver_min)
@test maximum(norm.(res_reg.X .- res_min.X,Inf)) == 0

# Test constraints
res_min.U[2][2] = 0.15
res_min.U[N-1][2] = 0.2

update_constraints!(res_reg,solver)
update_constraints!(res_min,solver_min)

res_min.C[1]
res_reg.C[1]
res_reg.C[1] == res_min.C[1][[1,3]]
@test res_min.C[1][end] ≈ √dt-0.15
@test res_min.C[2][end] ≈ .15-√dt
@test res_min.C[N-2][end] == sqrt(dt)-0.2
@test res_min.C[N-1][end] == 0
res_min.Iμ[1][end,end] == 1
@test res_min.MU[1] == ones(5)
res_min.U[N-1][2] = √dt
res_min.U[2][2] = √dt

calculate_jacobians!(res_reg,solver)
calculate_jacobians!(res_min,solver_min)

@test size(res_min.fu[1]) == (n,m+1)
@test res_reg.fx[1] == res_min.fx[1]
@test res_reg.fu[1] ≈ res_min.fu[1][:,1]
@test res_reg.fu[N-1] ≈ res_min.fu[N-1][:,1]

res_min.Cu[1]
@test res_min.Cu[1] == [Matrix(I,2,2); -Matrix(I,2,2); 0 1]
@test res_min.Cu[N-1] == [Matrix(I,2,2); -Matrix(I,2,2); 0 0]


solver_min.obj
_cost(solver_min,res_min)
total_time(solver_min,res_min)

# Test cost is the same (when c=0 and R[m̄,m̄] = 0)
obj_min_c0 = update_objective(obj_min,c=0.,Q = obj_c.Q, R = obj_c.R, Qf = obj.Qf)
solver_min = Solver(model,obj_min_c0,N=solver.N)
solver_min.opts.min_time_regularization = 0
Xrand = rand(n,N)
copyto!(res_reg.X,Xrand)
copyto!(res_min.X,Xrand)

J_reg = _cost(solver,res_reg)
J_min = _cost(solver_min,res_min)
@test J_reg == J_min

# Make sure constrained cost is NOT the same
J_reg = cost(solver,res_reg)
J_min = cost(solver_min,res_min)
@test abs(J_reg-J_min) > 1e-2


# Test rollout with gains is the same
update_constraints!(res_reg,solver)
update_constraints!(res_min,solver_min)

calculate_jacobians!(res_reg,solver)
calculate_jacobians!(res_min,solver_min)

v_reg = backwardpass!(res_reg,solver)
for k = 1:N
    res_min.K[k][1,:] = res_reg.K[k]
    res_min.K[k][2,:] = zeros(2)
    res_min.d[k][1] = res_reg.d[k][1]
    res_min.d[k][2] = 0.05*0
end

rollout!(res_reg,solver,1.)
rollout!(res_min,solver_min,1.)


solver_min.opts.min_time_regularization = 0
J_reg = cost(solver, res_reg, res_reg.X_, res_reg.U_)
J_min = cost(solver_min, res_min, res_min.X_, res_min.U_)
@test J_reg == J_min
@test to_array(res_reg.X_) ≈ to_array(res_min.X_)


# Test infeasible
solver = Solver(model,obj_c,dt=dt,opts=opts); solver.opts.infeasible = true;
solver_min = Solver(model,obj_min,N=21,opts=opts); solver_min.opts.infeasible = true;
m̄,mm = get_num_controls(solver)
@test (m̄,mm) == (1,3)
@test get_num_controls(solver_min) == (2,4)
p,pI,pE = get_num_constraints(solver)
@test get_num_constraints(solver_min) == (7,4,3)

res_reg = ConstrainedVectorResults(n,m+n,p,N)
res_min = ConstrainedVectorResults(n,m+1+n,p+3,solver_min.N)
res_reg.ρ[1] = 0
res_min.ρ[1] = 0

X0 = line_trajectory(solver)
U0 = ones(m,N)
U0_min = [U0; ones(m,solver_min.N)*sqrt(dt)]
ui = infeasible_controls(solver, X0, U0)
ui_min = infeasible_controls(solver_min, X0, U0_min)
@test ui == ui_min
copyto!(res_reg.X, X0)
copyto!(res_min.X, X0)
copyto!(res_reg.U, [ones(m,N); ui])
copyto!(res_min.U, [ones(m,solver_min.N); ones(m,solver_min.N)*sqrt(dt); ui_min])

update_constraints!(res_reg,solver)
update_constraints!(res_min,solver_min)
@test res_reg.C[1][1:2] == res_min.C[1][[1,3]]
@test res_min.C[1][2] == sqrt(dt) - solver_min.opts.max_dt
@test res_reg.C[1][3:4] == res_min.C[1][5:6]

rollout!(res_reg, solver)
rollout!(res_min, solver_min)
@test to_array(res_reg.X) == to_array(res_min.X)
@test to_array(res_min.X) == X0
@test to_array(res_reg.X) == X0

J = cost(solver, res_reg)
J = cost(solver_min, res_min)

calculate_jacobians!(res_reg,solver)
calculate_jacobians!(res_min,solver_min)
@test res_reg.fx[1] == res_min.fx[1]
@test res_reg.fu[N-1] == res_min.fu[N-1][:,[1,3,4]]


function init_solve(solver)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    res = ConstrainedVectorResults(n,mm,p,N)
    dt = get_initial_dt(solver)

    U0 = ones(m,N)
    if is_min_time(solver)
        U0 = [U0; ones(m,solver.N)*sqrt(dt)]
    end
    if solver.opts.infeasible
        X0 = line_trajectory(solver)
        copyto!(res.X, X0)

        ui = infeasible_controls(solver, X0, U0)
        U0 = [U0; ui]
    end
    copyto!(res.U, U0)

    update_constraints!(res,solver)
    rollout!(res, solver)
    res
end

function inner_loop(res, solver)
    calculate_jacobians!(res,solver)
    v = backwardpass_mintime!(res,solver)
    with_logger(logger) do
        J = forwardpass!(res,solver,v)
    end

    res.X .= deepcopy(res.X_)
    res.U .= deepcopy(res.U_)

    print_header(logger,TrajectoryOptimization.InnerLoop)
    print_row(logger,TrajectoryOptimization.InnerLoop)

    display(plot(to_array(res_min.U)'))
    total_time(solver_min, res_min)
end

obj_min = update_objective(obj_c,tf=:min,c=10., Q = obj.Q*0, R = obj.R, Qf = obj.Qf*1)
solver_min = Solver(model,obj_min,N=21,opts=opts)
solver_min.opts.infeasible = true;
solver_min.opts.min_time_regularization = 100
solver_min.opts.infeasible_regularization = 10
solver_min.opts.iterations_linesearch = 20
solver_min.opts.max_dt = 0.2
res_min = init_solve(solver_min)
[mu[4] = 100 for mu in res_min.MU]
[mu[5] = 1000 for mu in res_min.MU]
inner_loop(res_min, solver_min)
outer_loop_update(res_min,solver_min)

outer_loop_update(res_reg,solver)

res_min.U[1]

display(plot(to_array(res_min.X)'))
display(plot(to_array(res_min.U)'))



# Test inner loop
logger = TrajectoryOptimization.SolverLogger(TrajectoryOptimization.InnerIters)
TrajectoryOptimization.add_level!(logger,TrajectoryOptimization.InnerLoop,[],[],print_color=:green)
TrajectoryOptimization.add_level!(logger,TrajectoryOptimization.InnerIters,[],[],print_color=:blue)

obj_min = update_objective(obj_c,tf=:min,c=10., Q = obj.Q*0, R = obj.R, Qf = obj.Qf*0)
opts = SolverOptions()
opts.outer_loop_update = :individual
opts.verbose = true
solver = Solver(model,obj_c,dt=dt,opts=opts)
solver_min = Solver(model,obj_min,N=31,opts=opts)
solver_min.opts.min_time_regularization  = 0
n,m,N = get_sizes(solver)
p,pI,pE = TrajectoryOptimization.get_num_constraints(solver)
res_reg = ConstrainedVectorResults(n,m,p,N)
res_min = ConstrainedVectorResults(n,m+1,p+3,solver_min.N)
res_reg.ρ[1] = 0
res_min.ρ[1] = 0


copyto!(res_reg.U, ones(m,N))
copyto!(res_min.U, [ones(m,solver_min.N); ones(m,solver_min.N)*sqrt(dt)])
# copyto!(res_min.U, U_guess)

[mu[end] = 100 for mu in res_min.MU]
update_constraints!(res_reg,solver)
update_constraints!(res_min,solver_min)


rollout!(res_reg, solver)
rollout!(res_min, solver_min)

J = cost(solver, res_reg)
J = cost(solver_min, res_min)


iter = 1

calculate_jacobians!(res_reg,solver)
calculate_jacobians!(res_min,solver_min)

# Test that the fwp rollouts are the same given the same bwp gains
v_reg = backwardpass!(res_reg,solver)
v_min = backwardpass_mintime!(res_reg,solver)
v_reg == v_min

v_min = backwardpass_mintime!(res_min,solver_min)
# v_min = backwardpass_mintime2!(res_min,solver_min)

with_logger(logger) do

    J_reg = forwardpass!(res_reg,solver,v_reg)
    J_min = forwardpass!(res_min,solver_min,v_min)

end

print_header(logger,TrajectoryOptimization.InnerLoop)
print_row(logger,TrajectoryOptimization.InnerLoop)

res_reg.X .= deepcopy(res_reg.X_)
res_reg.U .= deepcopy(res_reg.U_)

res_min.X .= deepcopy(res_min.X_)
res_min.U .= deepcopy(res_min.U_)


display(plot(to_array(res_min.U)'))
# display(plot(to_array(res_min.X)'))
total_time(solver_min, res_min)

outer_loop_update(res_reg,solver)
outer_loop_update(res_min,solver_min)

norm(to_array(res_min.C)[end,:],Inf)
res_min.CN
res_min.IμN
res_min.μN

U_guess = to_array(res_min.U)

res_min.U[2].^2
res_reg.K[1]

res_min.K[1]


##################
#    CARTPOLE    #
##################

# Set up problem
model, obj0 = Dynamics.cartpole_analytical
n,m = model.n, model.m

obj = copy(obj0)
obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.0
u_bnd = 20
x_bnd = [0.6,Inf,Inf,Inf]
obj_c = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)
obj_min = update_objective(obj_c,tf=:min,c=1.,Q = obj.Q*0., Qf = obj.Qf*1)
dt = 0.1

solver = Solver(model,obj_c,dt=dt)
U0 = ones(1,solver.N)
res0,stat0 = solve(solver,U0)
plot(to_array(res0.X)[1:2,:]')
plot(to_array(res0.U)[1:1,:]')


solver_min = Solver(model,obj_min,N=solver.N)
solver_min.opts.use_static = false
solver_min.opts.max_dt = 0.25
solver_min.opts.verbose = true
solver_min.opts.cost_tolerance = 1e-4
solver_min.opts.cost_intermediate_tolerance = 1e-2
solver_min.opts.constraint_tolerance = 0.05
solver_min.opts.outer_loop_update = :default
solver_min.opts.min_time_regularization = 100
solver_min.opts.μ1 = 1
U0 = ones(1,solver_min.N)
U0[:,solver.N÷2:end] *= -1
res,stats = solve(solver_min,U0)
plot(to_array(res.X)[1:2,:]')
plot(to_array(res.U)[1:2,:]',ylim=[0,0.5],markershape=:auto)

X0 = line_trajectory(solver_min)
res,stats = solve(solver_min,X0,U0)
