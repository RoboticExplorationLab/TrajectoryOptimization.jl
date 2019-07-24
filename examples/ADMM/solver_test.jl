doubleintegrator3D = rk3(Dynamics.doubleintegrator3D)
n = doubleintegrator3D.n
m = doubleintegrator3D.m
actuated_models = [doubleintegrator3D,doubleintegrator3D,doubleintegrator3D]
num_act_models = length(actuated_models)
load_model = doubleintegrator3D

Q = 1.0*Diagonal(I,n)
Qf = 1.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)


N = 51
dt = 0.1

obj = LQRObjective(Q,R,Qf,xf,N)
x0 = zeros(n)
xf = ones(n)

probs = []
for i = 1:num_act_models
    prob = TrajectoryOptimization.Problem(actuated_models[i], obj, x0=x0, xf=xf, N=N, dt=dt)
    initial_controls!(prob, rand(m,N-1))
    rollout!(prob)
    push!(probs,prob)
end

probs
