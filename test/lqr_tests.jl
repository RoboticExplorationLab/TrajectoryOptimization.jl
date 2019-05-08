using Interpolations, OrdinaryDiffEq

# model
T = Float64
model = TrajectoryOptimization.Dynamics.doubleintegrator_model
n = model.n; m = model.m; r = model.r

# costs
Q = 1.0*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0*Diagonal(I,m)
x0 = [0; 0.]
xf = [pi; 0]

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-8)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-5)

N = 101
tf = 5.0
U0 = [zeros(m) for k = 1:N-1]

prob = TrajectoryOptimization.Problem(model, TrajectoryOptimization.Objective(costfun,N), integration=:rk4, x0=x0, N=N, tf=tf)
dt = prob.dt
TrajectoryOptimization.initial_controls!(prob, U0)

solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
TrajectoryOptimization.solve!(prob, solver_ilqr)
@test norm(prob.X[N] - xf) < 1e-3

U_ = [prob.U..., prob.U[end]]
K,P = tvlqr_dis(prob,Q,R,Qf)

K_vec = [reshape(K[k],n*m) for k = 1:N-1]
P_vec = [reshape(P[k],n^2) for k = 1:N]
plot(K_vec)
plot(P_vec)

function gen_cubic_interp(X::VectorTrajectory{T},dt::T)
    N = length(X); n = length(X[1])

    function interp(t::T) where T
        j = t/dt + 1.
        x = zeros(n)
        for i = 1:n
            x[i] = interpolate([X[k][i] for k = 1:N],BSpline(Cubic(Line(OnGrid()))))(j)
        end
        return x
    end
end

function gen_zoh_interp(X::VectorTrajectory{T},dt::T)
    N = length(X); n = length(X[1])

    function interp(t::T) where T
        if (t/dt + 1.0)%floor(t/dt + 1.0) < 0.99999
            j = convert(Int64,floor(t/dt)) + 1
        else
            j = convert(Int64,ceil(t/dt)) + 1
        end

        interpolate(X,BSpline(Constant()))(j)

    end
end


X_interp = gen_cubic_interp(prob.X,prob.dt)
U_interp = gen_zoh_interp([prob.U...,prob.U[end]],prob.dt)

# # x interp
plot(range(0,stop=total_time(prob),length=N),to_array(prob.X)[1,:])
plot!(range(0,stop=total_time(prob),length=N),to_array(prob.X)[2,:])

t_ = range(0,stop=total_time(prob),length=2*N)
scatter!(t_,to_array(X_interp.(t_))[1,:])
scatter!(t_,to_array(X_interp.(t_))[2,:])

# u interp
t_ = range(0,stop=total_time(prob)-prob.dt,length=N-1)
plot(t_,to_array(prob.U)[1,:],linetype=:stairs)

t_ = range(0,stop=total_time(prob),length=N)
scatter!(t_,to_array(U_interp.(t_))[1,:])

Z = zeros(n,n+m+r)

function Ac(t)
    x = X_interp(t)
    u = U_interp(t)
    # model.∇f(Z,x,u,zeros(r))
    model.∇f(Z,x,u)

    Z[:,1:n]
end

function Bc(t)
    x = X_interp(t)
    u = U_interp(t)
    # model.∇f(Z,x,u,zeros(r))
    model.∇f(Z,x,u)

    Z[:,n .+ (1:m)]
end

function riccati_con!(Ṗ::AbstractVector{T},P::AbstractVector{T},t::T) where T
    n = convert(Int64,sqrt(length(P)))
    Ṗ[1:n^2] = -1.0*reshape(Ac(t)'*reshape(P,n,n) + reshape(P,n,n)*Ac(t) - reshape(P,n,n)*Bc(t)*(R\(Bc(t)'*reshape(P,n,n))) + Q, n^2)
end

# function riccati_con_ode(u,p,t)
#     uu = zero(u)
#     riccati_con!(uu,u,t)
#     uu
# end

u0=vec(Qf)
tspan = (tf,0.0)
pro = ODEProblem(riccati_con_ode,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,Tsit5(),reltol=1e-12,abstol=1e-12)
p = plot()
for i = 1:4
    p = plot!(sol.t[end:-1:1],to_array(sol.u)[i,end:-1:1])
end
for i = 1:4
    p = plot!(range(0,stop=tf,length=N),to_array(P_vec)[i,:],style=:dash,legend=:topleft)
end
display(p)

P1 = sol.u[end]
P[1]

u0=vec(P1)
tspan = (0,tf)
pro = ODEProblem(riccati_con_ode,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,Tsit5(),reltol=1e-12,abstol=1e-12)
sol.u
p = plot()
for i = 1:4
    p = plot!(sol.t,to_array(sol.u)[i,:])
end
for i = 1:4
    p = plot!(range(0,stop=tf,length=N),to_array(P_vec)[i,:],style=:dash,legend=:topleft)
end
display(p)
sol.u[end]

function rk4_con(f!::Function, dt::Float64)
    # Runge-Kutta 4
    fd!(xdot,x,t,dt=dt) = begin
        k1 = k2 = k3 = k4 = zero(xdot)
        f!(k1, x, t);         k1 *= dt;
        f!(k2, x + k1/2, t + dt/2); k2 *= dt;
        f!(k3, x + k2/2, t + dt/2); k3 *= dt;
        f!(k4, x + k3, t + dt);    k4 *= dt;
        copyto!(xdot, x + (k1 + 2*k2 + 2*k3 + k4)/6)
    end
end

riccati_con_rk4 = rk4_con(riccati_con!,dt)
t_ = range(0.,stop=tf,length=N)
t_ = range(tf,stop=0.,length=N)

pp = [zeros(n^2) for k = 1:N]
pp[1] = P1
pp[1] = vec(Qf)


for k = 1:N-1
    riccati_con_rk4(pp[k+1],pp[k],t_[k],-dt)
end
