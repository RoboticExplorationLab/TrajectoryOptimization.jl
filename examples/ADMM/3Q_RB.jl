using CoordinateTransformations

load_dims = (0.5, 0.5, 0.2)
load_params = let (l,w,h) = load_dims
    (mass=0.1, inertia=Diagonal(1.0I,3),
        gravity=(@SVector [0,0,-9.81]),
        r_cables=[(@SVector [ l/2,    0, h/2])*0.8,
                  (@SVector [-l/2,  w/2, h/2])*0.8,
                  (@SVector [-l/2, -w/2, h/2])*0.8])
end
info = Dict{Symbol,Any}(:quat0=>4:7, :dims=>load_dims, :r_cables=>load_params.r_cables)

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
xf[4:7] = SVector(Quaternion(RotZ(pi/4)))
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

# Solve
lift_model = quadrotor_lift
lift_model = doubleintegrator3D_lift
probs, prob_load = init_lift_ADMM(lift_model, x0, xf,
	distributed=false, num_lift=num_lift, obstacles=false,quat=true, infeasible=false, doors=false, rigidbody=true);
sol, solvers = solve_admm(prob_load, probs, opts_al, parallel=true, max_iter=2);
anim = visualize_lift_system(vis, sol, door=:false)

r1 = [sol[4].X[k][1:3] - sol[1].X[k][1:3] for k = 1:100]
f1 = [sol[4].U[k][4:6] for k = 1:100]
[normalize(r)'normalize(f) for (r,f) in zip(r1,f1)]
plot([r'f for (r,f) in zip(r1,f1)])
plot([solvers[2].λ[k].cable_length for k = 1:101])
plot(sol[2].U, 4:6)


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


# Solve the whole system
x0 = [0., 0., 0.3]
xf = [6., 0., 0.3]
probs, prob_load2 = init_quad_ADMM(x0[1:3], xf[1:3], distributed=false, num_lift=num_lift, obstacles=false, quat=true, infeasible=false, doors=false, rigidbody=true);
sol, solvers, X_cache = solve_admm(prob_load2, probs, opts_al, true)

anim = visualize_quadrotor_lift_system(vis, sol, door=:false)
sol[2].xf
sol[2].X[end]

function dist(x)
    q = Quaternion(x[4:7])
    r = load_params.r_cables[1]
    [norm(sol[2].X[1][1:3] - (q*r + x[1:3]))^2 - 1.55^2]
end

function ∇dist(x)
    q = Quaternion(x[4:7])
    r = load_params.r_cables[1]
    diff = sol[2].X[1][1:3] - (q*r + x[1:3])
    C = zeros(13)
    C[1:3] = - 2diff
    C[4:7] = -2diff'grad_rotation(q,r)
    C
end
dist(x0)

ForwardDiff.jacobian(dist,xf) ≈ ∇dist(xf)'

plot(sol[2].X,3:3)
plot(sol[1].U,1:3)
plot(sol0[1].U,1:3)
plot(sol[1].U,5:7)
plot(sol0[1].U,5:7)

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
