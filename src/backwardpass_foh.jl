using TrajectoryOptimization
using Plots

function backwardpass_foh!(res::SolverIterResults,solver::Solver)
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    W = solver.obj.Q
    Wf = solver.obj.Qf
    xf = solver.obj.xf
    K = res.K
    b = res.b
    d = res.d

    dt = solver.dt

    X = res.X
    U = res.U

    R = getR(solver)

    S = zeros(n+m,n+m)
    s = zeros(n+m)

    v1 = 0.0
    v2 = 0.0

    # precompute once
    G = zeros(3*n+3*m,3*n+3*m)
    G[1:n,1:n] = W
    G[n+m+1:n+m+n,n+m+1:n+m+n] = 4.0*W
    G[2*n+2*m+1:3*n+2*m,2*n+2*m+1:3*n+2*m] = W

    G[n+1:n+m,n+1:n+m,] = R
    G[2*n+1:2*n+m,2*n+1:2*n+m] = 4.0*R
    G[3*n+1:3*n+m,3*n+1:3*n+m] = R

    g = zeros(3*n+3*m)

    # precompute most of matrix once
    E = zeros(3*n+3*m,2*n+2*m)
    E[1:n,1:n] = eye(n)
    E[n+1:n+m,n+1:n+m] = eye(m)
    #E[n+m+1:n+m+n,:] = M
    E[2*n+m+1:2*n+m+m,n+1:n+m] = 0.5*eye(m)
    E[2*n+m+1:2*n+m+m,2*n+m+1:2*n+m+m] = 0.5*eye(m)
    E[2*n+2*m+1:2*n+2*m+n,n+m+1:n+m+n] = eye(n)
    E[3*n+2*m+1:end,2*n+m+1:end] = eye(m)

    # Boundary conditions
    S[1:n,1:n] = Wf
    s[1:n] = Wf*(X[:,N]-xf)

    k = N-1
    # Loop backwards
    while k >= 1
        # println("Jacobians")

        # Calculate the L(x,u,y,v)
        Ac, Bc = solver.Fc(X[:,k],U[:,k])
        Ad, Bd, Cd = solver.Fd(X[:,k],U[:,k],U[:,k+1])

        # println("Ac: $(Ac)")
        # println("Bc: $(Bc)")
        #
        #
        # Ac, Bc = Jacobian(fc,X[:,k],U[:,k])
        # Ad, Bd, Cd = Jacobian(fd,X[:,k],U[:,k],U[:,k+1])
        #
        # println("Ac: $(Ac)")
        # println("Bc: $(Bc)")
        #
        # println("Functions:")
        # println(solver.fc(zeros(3),X[:,k],U[:,k]))
        # println(fc(X[:,k],U[:,k]))

        M = 0.25*[3*eye(n)+dt*Ac dt*Bc eye(n) zeros(n,m)]
        E[n+m+1:n+m+n,:] = M

        xm = M*[X[:,k];U[:,k];X[:,k+1];U[:,k+1]]
        # function xm_func(x,u,y)
        #     tmp = zeros(x)
        #     solver.fc(tmp,x,u)
        #     0.75*x + 0.25*solver.dt*tmp + 0.25*y
        # end
        # xm = xm_func(X[:,k],U[:,k],X[:,k+1])
        res.Xm[:,k] .= xm
        um = (U[:,k] + U[:,k+1])/2.0

        # # analytic solution
        # M1 = 0.25*3*eye(n)+dt*Ac
        # M2 = 0.25*dt*Bc
        #
        # Hxx = 4*M1'*W*M1 + W
        # Hxu = 2*M1'*zeros(n,m) + 4*M1*W*M2 + zeros(n,m)
        # Hxy = M1*W
        # Hxv= 2*M1*zeros(n,m)
        # Huu = R + 2*(M2'*zeros(n,m) + zeros(m,n)*M2) + 4*M2'*W*M2 + R
        # Huy = 0.5*zeros(m,n) + M2'*W
        # Huv = R + 2*M2'*zeros(n,m)
        # Hyy = 0.25*W + W
        # Hyv = 0.5*zeros(n,m) + zeros(n,m)
        # Hvv = R + R
        #
        # Hx = M1*W*(xm - xf) + W*(X[:,k] - xf)
        # Hu = 0.5*R*um + M2'*W*(xm -xf) + R*U[:,k+1]
        # Hy = 0.25*W*(xm - xf) + W*(X[:,k] - xf)
        # Hv = 0.5*R*um + R*U[:,k]

        g[1:n,1] = W*(X[:,k] - xf)
        g[n+1:n+m] = R*U[:,k]
        g[n+m+1:n+m+n] = 4.0*W*(xm - xf)
        g[n+m+n+1:n+m+n+m]= 4.0*R*(U[:,k] + U[:,k+1])./2
        g[n+m+n+m+1:n+m+n+m+n] = W*(X[:,k+1] - xf)
        g[n+m+n+m+n+1:end] = R*U[:,k+1]

        H = dt/6*E'*G*E
        h = dt/6*g'*E

        # parse L(...) into H blocks
        Hx = h[1:n]
        Hu = h[n+1:n+m]
        Hy = h[n+m+1:n+m+n]
        Hv = h[2*n+m+1:2*n+2*m]

        Hxx = H[1:n,1:n]
        Huu = H[n+1:n+m,n+1:n+m]
        Hyy = H[n+m+1:n+m+n,n+m+1:n+m+n]
        Hvv = H[2*n+m+1:2*n+2*m,2*n+m+1:2*n+2*m]
        Hxu = H[1:n,n+1:n+m]
        Hxy = H[1:n,n+m+1:n+m+n]
        Hxv = H[1:n,2*n+m+1:2*n+2*m]
        Huy = H[n+1:n+m,n+m+1:n+m+n]
        Huv = H[n+1:n+m,2*n+m+1:2*n+2*m]
        Hyv = H[n+m+1:n+m+n,2*n+m+1:2*n+2*m]

        # substitute in dynamics dx = Addx + Bddu1 + Cddu2
        Hx_ = Hx + Hy*Ad
        Hu_ = Hu + Hy*Bd
        Hv_ = Hv + Hy*Cd

        Hxx_ = Hxx + Hxy*Ad + Ad'*Hxy' + Ad'*Hyy*Ad
        Huu_ = Huu + Huy*Bd + Bd'*Huy' + Bd'*Hyy*Bd
        Hvv_ = Hvv + Hyv'*Cd + Cd'*Hyv + Cd'*Hyy*Cd
        Hxu_ = Hxu + Hxy*Bd + Ad'*Huy' + Ad'*Hyy*Bd
        Hxv_ = Hxv + Hxy*Cd + Ad'*Hyv + Ad'*Hyy*Cd
        Huv_ = Huv + Huy*Cd + Bd'*Hyv + Bd'*Hyy*Cd

        # parse (approximate) cost-to-go P
        Sy = s[1:n]
        Sv = s[n+1:n+m]
        Syy = S[1:n,1:n]
        Svv = S[n+1:n+m,n+1:n+m]
        Syv = S[1:n,n+1:n+m]

        Sx_ = Sy'*Ad
        Su_ = Sy'*Bd
        Sv_ = Sy'*Cd + Sv' # TODO come back and sort out transpose business

        Sxx_ = Ad'*Syy*Ad
        Suu_ = Bd'*Syy*Bd
        Svv_ = Svv + Cd'*Syy*Cd + Cd'*Syv + Syv'*Cd
        Sxu_ = Ad'*Syy*Bd
        Sxv_ = Ad'*Syy*Cd + Ad'*Syv
        Suv_ = Bd'*Syy*Cd + Bd'*Syv

        # collect terms to form Q
        Qx = Hx_ + Sx_
        Qu = Hu_ + Su_
        Qv = Hv_ + Sv_

        Qxx = Hxx_ + Sxx_
        Quu = Huu_ + Suu_
        Qvv = Hvv_ + Svv_

        Qxu = Hxu_ + Sxu_
        Qxv = Hxv_ + Sxv_
        Quv = Huv_ + Suv_

        K[:,:,k+1] .= -Qvv\Qxv'
        b[:,:,k+1] .= -Qvv\Quv'
        d[:,k+1] .= -Qvv\Qv'

        Qx_ = Qx + Qv*K[:,:,k+1] + d[:,k+1]'*Qxv' + d[:,k+1]'*Qvv*K[:,:,k+1]
        Qu_ = Qu + Qv*b[:,:,k+1] + d[:,k+1]'*Quv' + d[:,k+1]'*Qvv*b[:,:,k+1]
        Qxx_ = Qxx + Qxv*K[:,:,k+1] + K[:,:,k+1]'*Qxv' + K[:,:,k+1]'*Qvv*K[:,:,k+1]
        Quu_ = Quu + Quv*b[:,:,k+1] + b[:,:,k+1]'*Quv' + b[:,:,k+1]'*Qvv*b[:,:,k+1]
        Qxu_ = Qxu + K[:,:,k+1]'*Quv' + Qxv*b[:,:,k+1] + K[:,:,k+1]'*Qvv*b[:,:,k+1] # note, I'm using this instead of Qux'

        # generate (approximate) cost-to-go at timestep k
        s[1:n] = Qx_
        s[n+1:n+m] = Qu_
        S[1:n,1:n] = Qxx_
        S[n+1:n+m,n+1:n+m] = Quu_
        S[1:n,n+1:n+m] = Qxu_
        S[n+1:n+m,1:n] = Qxu_'

        v1 += d[:,k]'*Qv[:,1]
        v2 += d[:,k]'*Qvv*d[:,k]

        # at last time step, optimize over final control
        if k == 1
            K[:,:,k] .= -Quu_\Qxu_'
            b[:,:,k] .= zeros(m,m)
            d[:,k] .= -Quu_\Qu_'

            v1 += d[:,k]'*Qu_[:,1] #TODO confirm that this is correct
            v2 += d[:,k]'*Quu_*d[:,k]
        end

        k = k - 1;
    end

    return v1, v2
end
#
# dt = 0.1
# function rk3_foh1(f::Function, dt::Float64)
#     # Runge-Kutta 3 with first order hold on controls
#     fd(x,u1,u2) = begin
#         k1 = k2 = k3 = zeros(x)
#         k1 = f(x, u1);         k1 *= dt;
#         k2 = f(x + k1/2., (u1 + u2)./2); k2 *= dt;
#         k3 = f(x - k1 + 2.*k2, u2); k3 *= dt;
#         x + (k1 + 4.*k2 + k3)/6.
#     end
# end
# #
# function Jacobian(f::Function,x::Array{Float64,1},u::Array{Float64,1})
#     f1 = a -> f(a,u)
#     f2 = b -> f(x,b)
#     fx = ForwardDiff.jacobian(f1,x)
#     fu = ForwardDiff.jacobian(f2,u)
#     return fx, fu
# end
#
# function Jacobian(f::Function,x::Array{Float64,1},u1::Array{Float64,1},u2::Array{Float64,1})
#     f1 = a -> f(a,u1,u2)
#     f2 = b -> f(x,b,u2)
#     f3 = c -> f(x,u1,c)
#     fx = ForwardDiff.jacobian(f1,x)
#     fu1 = ForwardDiff.jacobian(f2,u1)
#     fu2 = ForwardDiff.jacobian(f3,u2)
#     return fx, fu1, fu2
# end

function rollout_foh!(res::SolverResults,solver::Solver)
    infeasible = solver.model.m != size(res.U,1)
    X = res.X; U = res.U

    X[:,1] = solver.obj.x0
    for k = 1:solver.N-1

        if solver.control_integration == :foh
            solver.fd(view(X,:,k+1), X[:,k], U[1:solver.model.m,k], U[1:solver.model.m,k+1])
        else
            solver.fd(view(X,:,k+1), X[:,k], U[1:solver.model.m,k])
        end
        if infeasible
            X[:,k+1] .+= U[solver.model.m+1:end,k]
        end
    end
end

function rollout_foh!(res::SolverResults,solver::Solver,alpha::Float64)
    infeasible = solver.model.m != size(res.U,1)
    N = solver.N
    X = res.X; U = res.U; K = res.K; b = res.b; d = res.d; X_ = res.X_; U_ = res.U_
    du = zeros(solver.model.m)
    dv = zeros(solver.model.m)

    X_[:,1] .= solver.obj.x0;
    du .= d[:,1]
    U_[:,1] .= U[:,1] + du

    for k = 2:N
        delta = X_[:,k-1] - X[:,k-1]

        dv .= (K[:,:,k]*delta + alpha*(b[:,:,k]*du + d[:,k]))
        # println("delta: $delta")
        # # println("dv: $dv")
        # println("u: $(U[:,k])")
        U_[:,k] .= U[:,k] + dv
        # println("u_: $(U_[:,k])")
        solver.fd(view(X_,:,k), X_[:,k-1], U_[1:solver.model.m,k-1], U_[1:solver.model.m,k])
        du .= dv

        if ~all(isfinite, X_[:,k]) || ~all(isfinite, U_[:,k-1])
            return false
        end
    end
    return true
end

function cost_foh(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    # pull out solver/objective values

    N = solver.N; Q = solver.obj.Q;xf = solver.obj.xf; Qf = solver.obj.Qf
    R = getR(solver)

    function stage_cost(x,u,W,R,xf)
        0.5*(x - xf)'*Q*(x - xf) + 0.5*u'*R*u
    end

    function xm_func(x,u,y,fc!)
        tmp = zeros(x)
        fc!(tmp,x,u)
        0.75*x + 0.25*solver.dt*tmp + 0.25*y
    end

    J = 0.0
    for k = 1:N-1
        J += solver.dt/6*(stage_cost(X[:,k],U[:,k],Q,R,xf) + 4*stage_cost(xm_func(X[:,k],U[:,k],X[:,k+1],solver.fc),(U[:,k] + U[:,k+1])/2,Q,R,xf) + stage_cost(X[:,k+1],U[:,k+1],Q,R,xf)) # rk3 foh stage cost (integral approximation)
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

# fd = rk3_foh1(Dynamics.pendulum_dynamics, dt)
# fc = Dynamics.pendulum_dynamics
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache=false
opts.c1=1e-4
opts.c2=5.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-3
opts.iterations_outerloop = 250
opts.iterations = 100

obj_uncon = TrajectoryOptimization.Dynamics.pendulum![2]

###
solver_foh = TrajectoryOptimization.Solver(Dynamics.pendulum![1], obj_uncon, dt=dt,integration=:rk3_foh, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(Dynamics.pendulum![1], obj_uncon, dt=dt,integration=:rk4, opts=opts)

U = ones(solver_foh.model.m, solver_foh.N)

results_foh = TrajectoryOptimization.UnconstrainedResults(solver_foh.model.n,solver_foh.model.m,solver_foh.N)
results_zoh = TrajectoryOptimization.UnconstrainedResults(solver_zoh.model.n,solver_zoh.model.m,solver_zoh.N)

results_foh.U[:,:] .= U
results_zoh.U[:,:] .= U

rollout_foh!(results_foh,solver_foh)
rollout!(results_zoh,solver_zoh)

cost_foh(solver_foh,results_foh.X,results_foh.U)
cost(solver_zoh,results_zoh.X,results_zoh.U)

plot(results_foh.X')
plot!(results_zoh.X')

backwardpass_foh!(results_foh,solver_foh)

calc_jacobians(results_zoh,solver_zoh)
backwardpass!(results_zoh,solver_zoh)

plot(results_foh.K[1,1,:])
plot!(results_zoh.K[1,2,:])
rollout_foh!(results_foh,solver_foh,0.5)
rollout!(results_zoh,solver_zoh,0.5)

cost_foh(solver_foh,results_foh.X_,results_foh.U_)
cost(solver_zoh,results_zoh.X_,results_zoh.U_)

results_foh.X .= results_foh.X_
results_foh.U .= results_foh.U_

results_zoh.X .= results_zoh.X_
results_zoh.U .= results_zoh.U_

backwardpass_foh!(results_foh,solver_foh)

calc_jacobians(results_zoh,solver_zoh)
backwardpass!(results_zoh,solver_zoh)

rollout_foh!(results_foh,solver_foh,0.5)
rollout!(results_zoh,solver_zoh,0.5)

cost_foh(solver_foh,results_foh.X_,results_foh.U_)
cost(solver_zoh,results_zoh.X_,results_zoh.U_)

# sol = solve(solver_zoh,U)
























#
# results_foh.X .= results_foh.X_
# results_foh.U .= results_foh.U_
# backwardpass_foh!(results_foh,solver)
# rollout_foh!(results_foh,solver,1.0)
# cost_foh(solver,results_foh.X_,results_foh.U_)

# results_foh.X .= results_foh.X_
# results_foh.U .= results_foh.U_
# backwardpass_foh!(results_foh,solver)
# rollout!(results_foh,solver,1.0)
# cost_foh(solver,results_foh.X_,results_foh.U_,results_foh.Xm)
#
#
#
#
#

## Sorting out Jacobians for Dubin's Car
# x1 = rand(3)
# u1 = rand(2)
#
# Ac, Bc = solver.Fc(x1,u1) # Jacobian1(fc,X[:,k],U[:,k])
# Ad, Bd, Cd = solver.Fd(x1,u1,u1) #Jacobian1(fd,X[:,k],U[:,k],U[:,k+1])
#
# Ac1, Bc1 = Jacobian(fc,x1,u1)
# Ad1, Bd1, Cd1 = Jacobian(fd,x1,u1,u1)
#
# isapprox.(Ac,Ac1)
# isapprox.(Bc,Bc1)
# isapprox.(Ad,Ad1)
# isapprox.(Bd,Bd1)
# isapprox.(Cd,Cd1)
#
# x2 = rand(3)
# u2 = rand(2)
# norm(solver.fd(zeros(3),x2,u2,u2) - fd(x2,u2,u2))
# norm(Dynamics.dubins_dynamics(x1,u1) -Dynamics.dubins_dynamics!(zeros(3),x1,u1))
# norm(Dynamics.dubins_dynamics(x1,u1) -solver.fc(ones(3),x1,u1))
# solver.fc(zeros(3),x1,u1)
#
# Bc
#
# Bc1
#
# function test1(x,u)
#
#     return [cos(x[1])*u[1];sin(x[2])*u[2]]
# end
#
# x = [pi/2.; pi/2.]
# u = [1.;1.]
# typeof(u)
# test1(x,u)
# Jacobian(test1,x,u)
#
# fc = Dynamics.dubinscar![1].f
#
#
# fa = f_augmented!(fc,3,2)
#
# F!(a,b,c) = ForwardDiff.jacobian!(a,fa,b,c)
# AA = zeros(5,5)
# x1 = ones(5)
# u1 = rand(5)
# F!(AA,x1,u1)
#
# AA
#
# h = Jacobian(Dynamics.dubinscar[1].f,x1,u1)
