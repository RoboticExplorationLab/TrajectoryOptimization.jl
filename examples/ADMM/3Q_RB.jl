
load_dims = (0.5, 0.5, 0.2)
load_params = let (l,w,h) = load_dims
    (mass=1.0, inertia=Diagonal(1.0I,3),
        r_cables=[(@SVector [ l/2,    0, h/2]),
                  (@SVector [-l/2,  w/2, h/2]),
                  (@SVector [-l/2, -w/2, h/2])])
end

n_load = 13
m_load = 3*num_lift
load_model = Model(Dynamics.load_dynamics!, n_load, m_load, load_params)
load_model_d = midpoint(load_model)
dt = prob_load.dt

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
Qf_load = Diagonal(I,n_load)*0.0
R_load = copy(prob_load.obj[1].R)
obj = LQRObjective(Q_load, R_load, Qf_load, xf, prob_load.N)

prob_load2 = Problem(midpoint(load_model), obj, x0=x0, xf=xf, N=prob_load.N, dt=prob_load.dt)
u0 = Float64[0,0,0,
             0,0,0,
             0,0,0]
U0 = [copy(u0) for k = 1:prob_load2.N-1]
initial_controls!(prob_load2,U0)



function visualize_rigidbody(vis, prob)
    anim = MeshCat.Animation(convert(Int,floor(1.0/prob.dt)))
    for k = 1:prob.N
        MeshCat.atframe(anim,vis,k) do frame
            settransform!(frame, compose(Translation(prob.X[k][1:3]),LinearMap(Quat(prob.X[k][4:7]...))))
        end
    end
    MeshCat.setanimation!(vis,anim)
end
