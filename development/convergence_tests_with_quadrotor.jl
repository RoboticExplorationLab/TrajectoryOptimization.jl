using Plots
using Random

Random.seed!(123)

# Solver options
tf = 5.0
N = 101
integration = :rk4
opts = SolverOptions()
opts.verbose = true
opts.cost_tolerance = 1e-8
opts.cost_tolerance_intermediate = 1e-8
opts.constraint_tolerance = 1e-8
opts.constraint_tolerance_coarse = 1e-8
opts.gradient_tolerance = 1e-12
opts.gradient_tolerance_intermediate = 1e-12
opts.use_gradient_aula = true
opts.active_constraint_tolerance = 0.0
opts.penalty_scaling = 10.0
opts.penalty_initial = 1e-3
opts.penalty_update_frequency = 1
opts.constraint_decrease_ratio = .25
opts.iterations = 1000
opts.iterations_outerloop = 50
opts.iterations_innerloop = 300
opts.outer_loop_update_type = :individual

# Set up model, objective, solver
model, = TrajectoryOptimization.Dynamics.quadrotor
n = model.n
m = model.m

# -initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; 0.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;40.;0.] # xyz position
xf[4:7] = q0

# -control limits
u_min = 0.0
u_max = 10.0

# Q = (1e-1)*Matrix(I,n,n)
# Q[4,4] = 1.0
# Q[5,5] = 1.0
# Q[6,6] = 1.0
# Q[7,7] = 1.0
# R = (1e-1)*Matrix(I,m,m)
Q = (1e-1)*Matrix(I,n,n)
R = (1e-1)*Matrix(I,m,m)
Qf = (1000.0)*Matrix(I,n,n)

# obstacles constraint
r_quad = 3.0
r_sphere = 3.0
spheres = ((0.,10.,0.,r_sphere),(0.,20.,0.,r_sphere),(0.,30.,0.,r_sphere))
n_spheres = 3

function cI(c,x,u)
    for i = 1:n_spheres
        c[i] = sphere_constraint(x,spheres[i][1],spheres[i][2],spheres[i][3],spheres[i][4]+r_quad)
    end
    c
end

# unit quaternion constraint
function cE(c,x,u)
    c = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
end

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,cI=cI,cE=cE)

solver_uncon = Solver(model,obj_uncon,integration=integration,N=N,opts=opts)
solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)

U0 = zeros(solver_uncon.model.m, solver_uncon.N-1)
@time results_uncon, stats_uncon = solve(solver_uncon,U0)

plot(to_array(results_uncon.X)[1:3,:]')

solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)

results_con.μ

@time results_con, stats_con = solve(solver_con,U0)

plot(to_array(results_con.U)[:,1:solver_con.N-1]')
plot(to_array(results_con.X)[1:3,:]')
plot(to_array(results_con.λ[1:N-1])')

max_violation(results_con)
total_time(solver_con,results_con)

plot!(log.(stats_con["max_condition_number"]))
plot(log.(stats_con["c_max"]).+15.5)
plot(log.(stats_con["cost"]))
