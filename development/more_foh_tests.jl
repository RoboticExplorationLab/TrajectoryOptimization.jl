# Set random seed
Random.seed!(7)

### Solver Options ###
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = false
# opts.c1 = 1e-4
# opts.c2 = 2.0
opts.constraint_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.cost_tolerance = 1e-5
# opts.iterations_outerloop = 100
# opts.iterations = 1000
# opts.iterations_linesearch = 50
opts.infeasible_regularization = 1e6
opts.outer_loop_update_type = :individual
opts.constraint_decrease_ratio = 0.1
######################

### Simple Pendulum ###
obj_uncon_p = TrajectoryOptimization.Dynamics.pendulum![2]
model_p = Dynamics.pendulum![1]

## Infeasible start with constraints pendulum (foh)
u_min = -2
u_max = 2
x_min = [-10;-10]
x_max = [10; 10]

obj_con_p = ConstrainedObjective(obj_uncon_p, x_min=x_min, x_max=x_max)
solver_con = Solver(model_p,obj_con_p,integration=:rk3_foh,dt=dt,opts=opts)

# -Linear interpolation for state trajectory
X_interp = line_trajectory(solver_con.obj.x0,solver_con.obj.xf,solver_con.N)
U = ones(solver_con.model.m,solver_con.N)

results_inf, = solve(solver_con,X_interp,U)

# Test final state from foh solve
@test norm(results_inf.X[:,end] - solver_con.obj.xf) < 1e-3

# plot(results_inf.X',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
# plot(results_inf.U',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")
# println("Final state: $(results_inf.X[:,end])")
# println("Final cost: $(results_inf.cost[end])")

results_inf

TrajectoryOptimization.λ_update_second_order(results_inf,solver_con)

c_aug = f_augmented(solver_con.c_fun,2,1)
c_aug
c_aug(ones(3))
ForwardDiff.jacobian(c_aug,ones(5))

#######
m = 1
n = 2

obj = obj_con_p

pI_obj, pE_obj = count_constraints(obj)
p = obj.p # number of constraints
pI, pI_c, pI_N, pI_N_c = pI_obj
pE, pE_c, pE_N, pE_N_c = pE_obj

u_min_active = isfinite.(obj.u_min)
u_max_active = isfinite.(obj.u_max)
x_min_active = isfinite.(obj.x_min)
x_max_active = isfinite.(obj.x_max)

# Inequality on control
pI_u_max = count(u_max_active)
pI_u_min = count(u_min_active)
pI_u = pI_u_max + pI_u_min

function c_control!(c,x,u)
    c[1:pI_u_max] = (u - obj.u_max)[u_max_active]
    c[pI_u_max+1:pI_u_max+pI_u_min] = (obj.u_min - u)[u_min_active]
end
cdot = zeros(pI_u)

x = X_interp[:,end]
u = U[:,5]

c_control!(cdot,x,u)
cdot

# Inequality on state
pI_x_max = count(x_max_active)
pI_x_min = count(x_min_active)
pI_x = pI_x_max + pI_x_min
function c_state!(c,x,u)
    c[1:pI_x_max] = (x - obj.x_max )[x_max_active]
    c[pI_x_max+1:pI_x_max+pI_x_min] = (obj.x_min - x)[x_min_active]
end
cdot1 = zeros(pI_x)
c_state!(cdot1,x,u)
cdot1

# Custom constraints
pI_c = pI - pI_x - pI_u
# TODO add custom constraints

# Form inequality constraint
function cI!(c,x,u)
    c_control!(view(c,1:pI_u),x,u)
    c_state!(view(c,(1:pI_x).+pI_u),x,u)
    if pI_c > 0
        obj.cI(view(c,(1:pI_c).+pI_u.+pI_x),x,u)
    end
end

cdot2 = zeros(pI)
cI!(cdot2,x,u)
cdot2

# Augment functions together
function c_fun!(c,x,u)
    infeasible = length(u) != m
    cI!(view(c,1:pI),x,u[1:m])
    if pE_c > 0
        obj.cE(view(c,(1:pE_c).+pI),x,u[1:m])
    end
    if infeasible
        c[pI+pE_c+1:pI+pE_c+n] = u[m+1:m+n]
    end
end

cdot3 = zeros(p)
c_fun!(cdot3,x,u)
cdot3

# Terminal Constraint
# TODO make this more general
function c_fun!(c,x)
    c[1:n] = x - obj.xf
end
