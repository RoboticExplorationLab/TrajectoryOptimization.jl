using LinearAlgebra
using Plots

########################
## Obstacle Avoidance ##
########################
model = Dynamics.dubinscar[1]
n,m = model.n, model.m

x0 = [0.0;0.0;0.]
xf = [10.0;10.0;0.]
tf =  5.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

u_bnd = 4.
u_min = [-u_bnd; -u_bnd]
u_max = [u_bnd; u_bnd]
x_min = [-Inf; -Inf; -Inf]
x_max = [Inf; Inf; Inf]

r = 1.0
circles = ((1.,2.,r),(5.,5.,r),(8.,9,r),(1.5,5,r),(7.5,5,r))
n_circles = length(circles)

function cI(c,x,u)
    for i = 1:n_circles
        c[i] = circle_constraint(x,circles[i][1],circles[i][2],circles[i][3])
    end
    c
end

obj_con = ConstrainedObjective(obj,u_min=u_min, u_max=u_max,x_min=x_min,x_max=x_max,cI=cI)

opts = SolverOptions()
opts.verbose = true
opts.use_gradient_aula = true
opts.constraint_tolerance = 1e-7
opts.cost_tolerance = 1e-7
opts.penalty_initial = 0.01
opts.penalty_scaling = 10
opts.cost_tolerance_intermediate = 1e-2
opts.outer_loop_update_type = :default


solver = Solver(model, obj_con, integration=:rk3, N=301, opts=opts)
U0 = rand(m,solver.N)
X0 = line_trajectory(solver)
X0_rollout = rollout(solver,U0)

solver.opts.use_nesterov = true
res, stats = solve(solver,U0)
stats["iterations"]
maximum(norm.(res.d,Inf))
plot(res.U)

solver2 = Solver(solver)
solver2.opts.square_root = true
solver2.opts.use_gradient_aula = false
solver2.opts.use_nesterov = false
res2, stats2 = solve(solver2,U0)

solver2.opts.use_nesterov = true
res3, stats3 = solve(solver2,U0)

plot(stats["c_max"],yscale=:log10)
plot!(stats2["c_max"],yscale=:log10)
plot!(stats3["c_max"])

# Obstacle Avoidance
plt = plot(title="Obstacle Avoidance",aspect_ratio=:equal)
plot_obstacles(circles)
plot_trajectory!(to_array(res.X),width=2,color=:green,label="Constrained")
plot_trajectory!(to_array(res2.X),width=2,color=:blue,label="Constrained")
plot_trajectory!(to_array(res3.X),width=2,color=:yellow,label="Constrained")

display(plt)
