
load_dims = (0.5, 0.5, 0.2)
load_params = let (l,w,h) = load_dims
    (mass=1.0, inertia=Diagonal(1.0I,3),
        r_cables=[(@SVector [ l/2,    0, h/2]),
                  (@SVector [-l/2,  w/2, h/2]),
                  (@SVector [-l/2, -w/2, h/2])])
end
info = Dict{Symbol,Any}(:quat0=>4:7)

n_load = 13
m_load = 3*num_lift
load_model = Model(Dynamics.load_dynamics!, n_load, m_load, load_params, info)
load_model_d = midpoint(load_model)
dt = prob_load.dt
Z = PartedMatrix(load_model_d)
jacobian!(Z, load_model_d, x0, u0, dt)

# Initial condition
x0_prev = prob_load.x0
x0 = zeros(n_load)
x0[1:3] = x0_prev[1:3]
x0[4] = 1

# Final condition
xf_prev = prob_load.xf
xf = zeros(n_load)
xf[1:3] = xf_prev[1:3]
xf[4] = 1


# Objective
Q_load = Diagonal(I,n_load)*0.0
qf_load = zeros(n_load)
qf_load[4:7] .= 1.0
Qf_load = Diagonal(qf_load)
R_load = copy(prob_load.obj[1].R)
obj = LQRObjective(Q_load, R_load, Qf_load, xf, prob_load.N)

prob_load2 = Problem(midpoint(load_model), obj, x0=x0, xf=xf, N=prob_load.N, dt=prob_load.dt)
u0 = Float64[0,0,0,
             0,0,0,
             0,0,0]
U0 = [copy(u0) for k = 1:prob_load2.N-1]
initial_controls!(prob_load2,U0)

x0 = [0., 0., 0.3]
xf = [6., 0., 0.3]
probs, prob_load2 = init_quad_ADMM(x0, xf, distributed=false, num_lift=num_lift, obstacles=false, quat=true, infeasible=false, doors=false, rigidbody=true);
sol, solvers = solve_admm(prob_load2, probs, opts_al, false)

z = [sol[1].X[k]; sol[1].U[k]]

sol[1].obj[90]
sol[1].X[end]
findmax_violation(sol[1])
sol[1].X[end]
sol[1].xf
solvers[1].C[6].cable_load
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
