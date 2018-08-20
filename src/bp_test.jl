using TrajectoryOptimization

function backwardpass_foh!(X,U,solver::Solver)
    N = solver.N; n = solver.model.n; m = solver.model.m;
    W = solver.obj.Q; Wf = solver.obj.Qf; xf = solver.obj.xf;
    dt = solver.dt
    R = getR(solver)

    XM = zeros(size(X))

    alpha = zeros(m,n,N)
    beta = zeros(m,m,N)
    gamma = zeros(m,N)

    S = zeros(n+m,n+m,N)
    s = zeros(n+m,N)

    xm = zeros(n)

    # precompute once
    G = zeros(3*n+3*m,3*n+3*m)
    for i = 1:n+m:3*n+3m
        G[i:i+n-1,i:i+n-1] = W
    end
    for i = n+1:n+m:3*n+3*m
        G[i:i+m-1,i:i+m-1] = R
    end

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
    S[1:n,1:n,N] = Wf
    s[1:n,N] = Wf*(X[:,N]-xf)

    H = []
    k = N-1
    # Loop backwards
    while k >= 1

        # Calculate the L(x,u,y,v)
        Ac, Bc = Jacobian(fc,X[:,k],U[:,k])
        Ad, Bd, Cd = Jacobian(fd,X[:,k],U[:,k],U[:,k+1])

        M = 0.25*[3*eye(n)+dt*Ac Bc*eye(m) eye(n) zeros(n,m)]
        E[n+m+1:n+m+n,:] = M

        xm = M*[X[:,k];U[:,k];X[:,k+1];U[:,k+1]]
        XM[:,k] = xm
        g[1:n,1] = W*(X[:,k] - xf)
        g[n+1:n+m] = R*U[:,k]
        g[n+m+1:n+m+n] = W*(xm - xf)
        g[n+m+n+1:n+m+n+m]= R*(U[:,k] + U[:,k+1])./2
        g[n+m+n+m+1:n+m+n+m+n] = W*(X[:,k+1] - xf)
        g[n+m+n+m+n+1:end] = R*U[:,k+1]

        H = dt/6.*E'*G*E
        h = g'*E

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

        # substitute in dynamics dx = Adx + Bdu1 + Cdu2
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
        Sy = s[1:n,k]
        Sv = s[n+1:n+m,k]
        Syy = S[1:n,1:n,k]
        Svv = S[n+1:n+m,n+1:n+m,k]
        Syv = S[1:n,n+1:n+m,k]

        Sx_ = Sy'*Ad
        Su_ = Sy'*Bd
        Sv_ = Sy'*Cd + Sv' # TODO come back and sort out transpose business
        #
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

        alpha[:,:,k+1] = -Qvv\Qxv'
        beta[:,:,k+1] = -Qvv\Quv'
        gamma[:,k+1] = -Qvv\Qv'

        Qx_ = Qx + Qv*alpha[:,:,k+1] + gamma[:,k+1]'*Qxv' + gamma[:,k+1]'*Qvv*alpha[:,:,k+1]
        Qu_ = Qu + Qv*beta[:,:,k+1] + gamma[:,k+1]'*Quv' + gamma[:,k+1]'*Qvv*beta[:,:,k+1]
        Qxx_ = Qxx + Qxv*alpha[:,:,k+1] + alpha[:,:,k+1]'*Qvv*alpha[:,:,k+1]
        Quu_ = Quu + Quv*beta[:,:,k+1] + beta[:,:,k+1]*Quv' + beta[:,:,k+1]'*Qvv*beta[:,:,k+1]
        Qxu_ = Qxu + alpha[:,:,k+1]'*Quv' + Qxv*beta[:,:,k+1] + alpha[:,:,k+1]'*Qvv*beta[:,:,k+1]

        # generate (approximate) cost-to-go at timestep k
        s[1:n,k] = Qx_
        s[n+1:n+m,k] = Qu_
        S[1:n,1:n,k] = Qxx_
        S[n+1:n+m,n+1:n+m] = Quu_
        S[1:n,n+1:n+m] = Qxu_
        S[n+1:n+m,1:n] = Qxu_'

        # if at last time step, optimize over final control
        if k == 1
            alpha[:,:,k] = -Quu_\Qxu_'
            beta[:,:,k] = zeros(m,m)
            gamma[:,k] = -Quu_\Qu_'
        end

        k = k - 1;
    end

    return alpha, beta, gamma, XM
end

function rk3_foh(f::Function, dt::Float64)
    # Runge-Kutta 3 with first order hold on controls
    fd(x,u1,u2) = begin
        k1 = k2 = k3 = zeros(x)
        k1 = f(x, u1);         k1 *= dt;
        k2 = f(x + k1/2., (u1 + u2)./2); k2 *= dt;
        k3 = f(x - k1 + 2.*k2, u2); k3 *= dt;
        x + (k1 + 4.*k2 + k3)/6.
    end
end

function Jacobian(f::Function,x::Array{Float64,1},u::Array{Float64,1})
    f1 = a -> f(a,u)
    f2 = b -> f(x,b)
    fx = ForwardDiff.jacobian(f1,x)
    fu = ForwardDiff.jacobian(f2,u)
    return fx, fu
end

function Jacobian(f::Function,x::Array{Float64,1},u1::Array{Float64,1},u2::Array{Float64,1})
    f1 = a -> f(a,u1,u2)
    f2 = b -> f(x,b,u2)
    f3 = c -> f(x,u1,c)
    fx = ForwardDiff.jacobian(f1,x)
    fu1 = ForwardDiff.jacobian(f2,u1)
    fu2 = ForwardDiff.jacobian(f3,u2)
    return fx, fu1, fu2
end

function rollout_foh!(res::SolverResults,solver::Solver,alpha::Float64,k1,k2,k3,dyna)
    # infeasible = solver.model.m != size(res.U,1)
    N = solver.N
    X = res.X; U = res.U; X_ = res.X_; U_ = res.U_; K = res.K; d = res.d;
    du1 = []
    X_[:,1] = solver.obj.x0;
    for k = 2:N
        delta = X_[:,k-1] - X[:,k-1]
        if k == 2
            du1 = k1[:,:,1]*delta + alpha*k3[:,1]
            U_[:,1] = U[:,1] + du1
        end
        du2 = k1[:,:,k]*delta + alpha*(k2[:,:,k]*du1 + k3[:,1]) # TODO line search term goes on bias term, I think this is corrent then
        U_[:, k] = U[:, k-1] + du2
        X[:,k] .= dyna(X_[:,k-1], U_[1:solver.model.m,k-1], U_[1:solver.model.m,k])

        du1 = du2

        if ~all(isfinite, X_[:,k]) || ~all(isfinite, U_[:,k-1])
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)
Compute the unconstrained cost for foh (rk3) # TODO add better description
"""
function cost_foh(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},XM)
    # pull out solver/objective values

    N = solver.N; Q = solver.obj.Q;xf = solver.obj.xf; Qf = solver.obj.Qf
    R = getR(solver)

    function l(x,u)
        0.5*(x - xf)'*Q*(x - xf) + 0.5*u'*R*u
    end

    J = 0.0
    for k = 1:N-1
        J += solver.dt/6*(l(X[:,k],U[:,k]) + 4*l(XM[:,k],(U[:,k] + U[:,k+1])/2) + l(X[:,k+1],U[:,k+1])) # rk3 foh stage cost (integral approximation)
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

dt = 0.1
fd = rk3_foh(Dynamics.dubins_dynamics, dt)
fc = Dynamics.dubins_dynamics


opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache=true
opts.c1=1e-4
opts.c2=2.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-5
opts.iterations_outerloop = 250
opts.iterations = 1000

obj_uncon = TrajectoryOptimization.Dynamics.dubinscar![2]
obj_uncon.R[:] = 1e-2*eye(2) # control needs to be properly regularized for infeasible start to produce a good warm-start control output
solver = TrajectoryOptimization.Solver(Dynamics.dubinscar![1], obj_uncon, dt=0.1, opts=opts)
U = rand(solver.model.m, solver.N)

alpha = zeros(solver.model.m,solver.model.n,solver.N)
beta = zeros(solver.model.m,solver.model.m,solver.N)
gamma = zeros(solver.model.m,solver.N)

results_foh = TrajectoryOptimization.UnconstrainedResults(solver.model.n,solver.model.m,solver.N)
results_foh.U[:,:] = U
rollout_foh!(results_foh,solver,1.0,alpha,beta,gamma,fd)
results_foh.X

# initial cost
alpha, beta, gamma, XM = backwardpass_foh!(results_foh.X,results_foh.U,solver)
rollout_foh!(results_foh,solver,1.0,alpha,beta,gamma,fd)
cost_foh(solver,results_foh.X_,results_foh.U_,XM)

# cost after 1 update
results_foh.X .= results_foh.X_
results_foh.U .= results_foh.U_
alpha, beta, gamma, XM = backwardpass_foh!(results_foh.X,results_foh.U,solver)
rollout_foh!(results_foh,solver,0.5,alpha,beta,gamma,fd)
cost_foh(solver,results_foh.X,results_foh.U,XM)

alpha, beta, gamma, XM = backwardpass_foh!(results_foh.X,results_foh.U,solver)
rollout_foh!(results_foh,solver,0.5,alpha,beta,gamma,fd)
cost_foh(solver,results_foh.X,results_foh.U,XM)
