using CoordinateTransformations

load_dims = (0.5, 0.5, 0.2)
load_params = let (l,w,h) = load_dims
    (mass=0.1, inertia=Diagonal(1.0I,3),
        gravity=(@SVector [0,0,-9.81]),
        r_cables=[(@SVector [ l/2,    0, h/2])*0,
                  (@SVector [-l/2,  w/2, h/2])*0,
                  (@SVector [-l/2, -w/2, h/2])*0])
end
info = Dict{Symbol,Any}(:quat0=>4:7, :dims=>load_dims)

n_load = 13
m_load = 3*num_lift
load_model = Model(load_dynamics!, n_load, m_load, load_params, info)
load_model.info[:radius] = 0.2
load_model.info[:rope_length] = 1.55
load_model_d = midpoint(load_model)
dt = prob_load.dt
N = 101

# Initial condition
x0_prev = prob_load.x0
x0 = zeros(n_load)
x0[1:3] = x0_prev[1:3]
x0[4] = 1

# Final condition
xf_prev = prob_load.xf
xf = zeros(n_load)
xf[1:3] = xf_prev[1:3]
xf[4:7] = SVector(Quaternion(RotZ(0*pi/2)))
# xf[4] = 1

# Objective
q_load = ones(n_load)*1e-6
q_load[8:10] .= 1e-2
q_load[4:7] .= 1e-2
q_load[11:13] .= 1e-2
Q_load = Diagonal(q_load)
qf_load = copy(q_load)*1e6
qf_load[4:7] .= 10.0
Qf_load = Diagonal(qf_load)
R_load = copy(prob_load.obj[1].R)
obj = LQRObjective(Q_load, R_load, Qf_load, xf, prob_load.N)

prob_load2 = Problem(midpoint(load_model), obj, x0=x0, xf=xf, N=prob_load.N, dt=prob_load.dt)
prob_load2.constraints[101] += goal_constraint(xf)
u0 = Float64[0,0,0,
             0,0,0,
             0,0,0]
U0 = [copy(u0) for k = 1:prob_load2.N-1]
initial_controls!(prob_load2,U0)
sol_load, solver = solve(prob_load2, opts_al)
visualize_rigidbody(vis["load"], sol_load)

# Solve quadrotor
probs, = init_quad_ADMM(x0[1:3], xf[1:3], distributed=false, num_lift=num_lift, obstacles=false, quat=true, infeasible=false, doors=false, rigidbody=true);
X_cache, U_cache, X_lift, U_lift = init_cache(probs);
prob = probs[1]
solve!(prob, opts_al)
visualize_rigidbody(vis["lift1"], prob)
for k = 1:N-1
    prob.obj[k].R .= I(7)*1e-2
end

X_cache[2][1] .= sol_load.X
U_cache[2][1] .= sol_load.U
update_lift_problem(prob, X_cache[2], U_cache[2], 1, prob_load2.model.info[:rope_length], prob.model.info[:radius])
sol_quad1, = solve(prob, opts_al)
findmax_violation(sol_quad1)
visualize_rigidbody(vis["lift1"], sol_quad1)
prob.constraints[1]
sol_load.U[2][1:3]

sol_quad1.U[2][5:7]

x0 = [0., 0., 0.3]
xf = [6., 0., 0.3]
probs, prob_load2 = init_quad_ADMM(x0, xf, distributed=false, num_lift=num_lift, obstacles=false, quat=true, infeasible=false, doors=false, rigidbody=true);
sol, solvers = solve_admm(prob_load2, probs, opts_al, true)

solve_aula!(sol[2], solvers[2])
solve_aula!(sol0[2], solvers0[2])

plot(sol[1].U, 1:9)
plot(sol0[1].U, 1:9)
plot(sol[2].U, 5:7)
plot(sol0[2].U, 5:7)

findmax_violation(sol[1])
sol[1].X[end]
findmax_violation(sol0[1])


sol[1] = sol_load
sol[2] = sol_quad1
anim = visualize_quadrotor_lift_system(vis, sol, door=:false)

sol[1].X[end]
sol[1].X[2]
sol[4].X[end]
plot(sol[1].U,1:3)
plot(sol[1].U,1:3)
plot(sol[2].U,5:7)

c = zeros(12)
k = 6
evaluate!(c, sol[1].constraints[k][2], sol[1].X[k], sol[1].U[k])
norm(sol[1].X[k] - sol[2].X[k])
sol[1].model.info[:rope_length]

visualize_rigidbody(vis["load"], sol[1])
plot(sol[1].U)
plot_quad_scene(vis, 6, sol)

k = 1
setobject!(vis["load"], HyperRectangle(Vec(load_dims ./ (-2)), Vec(load_dims)))
settransform!(vis, Translation(0,0,0))
load_dims ./ -2
settransform!(vis["load"], compose(Translation(sol[1].X[k][1:3]),LinearMap(Quat(sol[1].X[k][4:7]...))))
function visualize_rigidbody(vis, prob)
    anim = MeshCat.Animation(convert(Int,floor(1.0/prob.dt)))
    for k = 1:prob.N
        MeshCat.atframe(anim,vis,k) do frame
            settransform!(frame, compose(Translation(prob.X[k][1:3]),LinearMap(Quat(prob.X[k][4:7]...))))
        end
    end
    MeshCat.setanimation!(vis,anim)
end
