using TrajectoryOptimization

function backwardpass_foh_alt!(res::SolverIterResults,solver::Solver)
    N = solver.N
    n = solver.model.n
    m = solver.model.m

    Q = solver.obj.Q
    Qf = solver.obj.Qf
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

    # line search stuff
    v1 = 0.0
    v2 = 0.0


    # Boundary conditions
    S[1:n,1:n] = Qf
    s[1:n] = Qf*(X[:,N]-xf)

    k = N-1
    while k >= 1
        ## Calculate the L(x,u,y,v)

        # unpack Jacobians
        Ac1,Bc1 = res.Ac[:,:,k], res.Bc[:,:,k]
        Ac2,Bc2 = res.Ac[:,:,k+1], res.Bc[:,:,k+1]
        Ad, Bd, Cd = res.fx[:,:,k], res.fu[:,:,k], res.fv[:,:,k]

        # calculate xm, um using cubic and linear splines
        xdot1 = zeros(n)
        xdot2 = zeros(n)
        solver.model.f(xdot1,X[:,k],U[1:solver.model.m,k])
        solver.model.f(xdot2,X[:,k+1],U[1:solver.model.m,k+1])
        xm = 0.5*X[:,k] + dt/8*xdot1 + 0.5*X[:,k+1] - dt/8*xdot2
        um = (U[:,k] + U[:,k+1])/2.0

        println("got here")
        # Expansion of stage cost L(x,u,y,v) -> dL(dx,du,dy,dv)
        Lx = dt/6*Q*(X[:,k] - xf) + (0.5*eye(n) + dt/8*Ac1)'*Q*(xm - xf)
        Lu = dt/6*R*U[:,k] + (dt/8*Bc1)'*Q*(xm - xf) + 0.5*R*um
        Ly = dt/6*Q*(X[:,k+1] - xf) + (0.5*eye(n) - dt/8*Ac2)'*Q*(xm - xf)
        Lv = dt/6*R*U[:,k+1] + (-dt/8*Bc2)'*Q*(xm - xf) + 0.5*R*um

        Lxx = dt/6*Q + (0.5*eye(n) + dt/8*Ac1)'*Q*(0.5*eye(n) + dt/8*Ac1)
        Luu = dt/6*R + (dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5
        Lyy = dt/6*Q + (0.5*eye(n) - dt/8*Ac2)'*Q*(0.5*eye(n) - dt/8*Ac2)
        Lvv = dt/6*R + (-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5

        Lxu = (0.5*eye(n) + dt/8*Ac1)'*Q*(dt/8*Bc1)
        Lxy = (0.5*eye(n) + dt/8*Ac1)'*Q*(0.5*eye(n) - dt/8*Ac2)
        Lxv = (0.5*eye(n) + dt/8*Ac1)'*Q*(-dt/8*Bc2)
        Luy = (dt/8*Bc1)'*Q*(0.5*eye(n) - dt/8*Ac2)
        Luv = (dt/8*Bc1)'*Q*(-dt/8*Bc2)
        Lyv = (0.5*eye(n) - dt/8*Ac2)'*Q*(-dt/8*Bc2)

        # Unpack cost-to-go P, then add L + P
        Sy = s[1:n]
        Sv = s[n+1:n+m]
        Syy = S[1:n,1:n]
        Svv = S[n+1:n+m,n+1:n+m]
        Syv = S[1:n,n+1:n+m]

        Ly += Sy
        Lv += Sv
        Lyy += Syy
        Lvv += Svv
        Lyv += Syv

        # Substitute in discrete dynamics dx = (Ad)dx + (Bd)du1 + (Cd)du2

        println("size Lx: $(size(vec(Lx)))")
        println("size Ly: $(size(Ly))")
        println("size Ad: $(size(Ad))")
        println("size Bd: $(size(Bd))")

        Qx = vec(Lx) + Ad'*vec(Ly)
        Qu = vec(Lu) + Bd'*vec(Ly)
        Qv = vec(Lv) + Cd'*vec(Ly)

        Qxx = Lxx + Lxy*Ad + Ad'*Lxy' + Ad'*Lyy*Ad
        Quu = Luu + Luy*Bd + Bd'*Luy' + Bd'*Lyy*Bd
        Qvv = Lvv + Lyv'*Cd + Cd'*Lyv + Cd'*Lyy*Cd
        Qxu = Lxu + Lxy*Bd + Ad'*Luy' + Ad'*Lyy*Bd
        Qxv = Lxv + Lxy*Cd + Ad'*Lyv + Ad'*Lyy*Cd
        Quv = Luv + Luy*Cd + Bd'*Lyv + Bd'*Lyy*Cd

        #TODO add regularization
        # regularization
        # if !isposdef(Qvv)
        #     mu[1] += solver.opts.mu_reg_update
        #     k = N-1
        #     if solver.opts.verbose
        #         println("regularized")
        #     end
        #     break
        # end

        K[:,:,k+1] .= -Qvv\Qxv'
        b[:,:,k+1] .= -Qvv\Quv'
        d[:,k+1] .= -Qvv\vec(Qv)

        Qx_ = vec(Qx) + K[:,:,k+1]'*vec(Qv) + Qxv*vec(d[:,k+1]) + K[:,:,k+1]'Qvv*d[:,k+1]
        Qu_ = vec(Qu) + b[:,:,k+1]'*vec(Qv) + Quv*vec(d[:,k+1]) + b[:,:,k+1]'*Qvv*d[:,k+1]
        Qxx_ = Qxx + Qxv*K[:,:,k+1] + K[:,:,k+1]'*Qxv' + K[:,:,k+1]'*Qvv*K[:,:,k+1]
        Quu_ = Quu + Quv*b[:,:,k+1] + b[:,:,k+1]'*Quv' + b[:,:,k+1]'*Qvv*b[:,:,k+1]
        Qxu_ = Qxu + K[:,:,k+1]'*Quv' + Qxv*b[:,:,k+1] + K[:,:,k+1]'*Qvv*b[:,:,k+1]

        # cache (approximate) cost-to-go at timestep k
        s[1:n] = Qx_
        s[n+1:n+m] = Qu_
        S[1:n,1:n] = Qxx_
        S[n+1:n+m,n+1:n+m] = Quu_
        S[1:n,n+1:n+m] = Qxu_
        S[n+1:n+m,1:n] = Qxu_'

        # line search terms
        v1 += -d[:,k+1]'*vec(Qv)
        v2 += d[:,k+1]'*Qvv*d[:,k+1]

        # at last time step, optimize over final control
        if k == 1
            K[:,:,1] .= -Quu_\Qxu_'
            b[:,:,1] .= zeros(m,m)
            d[:,1] .= -Quu_\vec(Qu_)

            v1 += -d[:,1]'*vec(Qu_)
            v2 += d[:,1]'*Quu_*d[:,1]
        end

        k = k - 1;
    end

    return v1, v2
end

opts = TrajectoryOptimization.SolverOptions()
opts.verbose=true
opts.cache=true

obj_uncon = TrajectoryOptimization.Dynamics.dubinscar![2]
obj_uncon.R[:] = 1e-2*eye(2) # control needs to be properly regularized for infeasible start to produce a good warm-start control output
solver = TrajectoryOptimization.Solver(Dynamics.dubinscar![1], obj_uncon,integration=:rk3_foh, dt=0.1, opts=opts)
U = rand(solver.model.m, solver.N)
#
# alpha = zeros(solver.model.m,solver.model.n,solver.N)
# beta = zeros(solver.model.m,solver.model.m,solver.N)
# gamma = zeros(solver.model.m,solver.N)
#
results_foh = TrajectoryOptimization.UnconstrainedResults(solver.model.n,solver.model.m,solver.N)
results_foh.U[:,:] = U
rollout!(results_foh,solver)
results_foh.X
cost(solver,results_foh.X,results_foh.U)

#
# # initial cost
calc_jacobians(results_foh,solver)
backwardpass_foh_alt!(results_foh,solver)
rollout!(results_foh,solver,1.0/64)
cost(solver,results_foh.X_,results_foh.U_)

a = 1
#
# # cost after 1 update
# alpha, beta, gamma, XM = backwardpass_foh!(results_foh.X,results_foh.U,solver)
# rollout_foh!(results_foh,solver,0.5,alpha,beta,gamma,fd)
# cost_foh(solver,results_foh.X,results_foh.U,XM)
# #
# alpha, beta, gamma, XM = backwardpass_foh!(results_foh.X,results_foh.U,solver)
# rollout_foh!(results_foh,solver,0.25,alpha,beta,gamma,fd)
# cost_foh(solver,results_foh.X,results_foh.U,XM)


###
solver = TrajectoryOptimization.Solver(Dynamics.dubinscar![1], obj_uncon, dt=0.1,integration=:rk3_foh, opts=opts)
U = ones(solver.model.m, solver.N)
results_foh = TrajectoryOptimization.UnconstrainedResults(solver.model.n,solver.model.m,solver.N)
results_foh.U[:,:] = U
results_foh.K
rollout!(results_foh,solver)
results_foh.K
backwardpass_foh!(results_foh,solver)
results_foh.X
rollout!(results_foh,solver,1.0)
results_foh.Xm
cost_foh(solver,results_foh.X_,results_foh.U_,results_foh.Xm)
results_foh.X .= results_foh.X_
results_foh.U .= results_foh.U_
backwardpass_foh!(results_foh,solver)
rollout!(results_foh,solver,0.5)
cost_foh(solver,results_foh.X_,results_foh.U_,results_foh.Xm)

#
# # solver.fd(rand(3),rand(3),rand(2),rand(2))
# # solver.fc(rand(3),rand(3),rand(2))
#
# solver.Fc(rand(3),rand(2))
# solver.Fd(rand(3),rand(2),rand(2))
# #
# solver.fd(rand(3),rand(3),rand(2),rand(2))
# solver.Fd(rand(3),rand(2))
# solver.Fc(rand(3),rand(2))
#
# fd! = rk3_foh(Dynamics.dubinscar![1].f, dt)
# tmp = TrajectoryOptimization.f_augmented_foh!(fd!,3,2)
#
#
# tmp(zeros(8),ones(8))
