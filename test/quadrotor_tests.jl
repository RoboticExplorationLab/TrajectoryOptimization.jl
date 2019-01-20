Random.seed!(123)

# Solver options
N = 101
integration = :rk4
opts = SolverOptions()
opts.verbose = false
opts.square_root = true
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5

# Set up model, objective, solver
model, obj = TrajectoryOptimization.Dynamics.quadrotor
obj_c = TrajectoryOptimization.Dynamics.quadrotor_constrained[2]
n = model.n
m = model.m

# Unconstrained solve
solver_uncon = Solver(model,obj,integration=integration,N=N,opts=opts)
U0 = ones(solver_uncon.model.m, solver_uncon.N-1)
results_uncon, stats_uncon = solve(solver_uncon,U0)
@test norm(results_uncon.X[end]-obj.xf) < 5e-3

# Constrained
r_quad = 3.0
r_sphere = 3.0
spheres = ((0.,10.,0.,r_sphere),(0.,20.,0.,r_sphere),(0.,30.,0.,r_sphere))
n_spheres = 3

# -control limits
u_min = 0.0
u_max = 100.0

# 3 sphere obstacles
function cI_3obs_quad(c,x,u)
    for i = 1:n_spheres
        c[i] = TrajectoryOptimization.sphere_constraint(x,spheres[i][1],spheres[i][2],spheres[i][3],spheres[i][4]+r_quad)
    end
    c
end

# unit quaternion constraint
function unit_quaternion(c,x,u)
    c = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
end

obj_uq = TrajectoryOptimization.ConstrainedObjective(obj,cE=unit_quaternion)

solver = Solver(model,obj_uq,integration=integration,N=N,opts=opts)
results, stats = solve(solver,U0)

@test norm(results.X[end]-obj_uq.xf) < 1e-5
@test norm(max_violation(results)) < 1e-5

# obj_obs = TrajectoryOptimization.ConstrainedObjective(obj,cE=unit_quaternion,cI=cI_3obs_quad)
# solver = Solver(model,obj_obs,integration=integration,N=N,opts=opts)
# results, stats = solve(solver,U0)
#
# @test norm(results.X[end]-obj_obs.xf) < 1e-5
# @test norm(max_violation(results)) < 1e-5
