using TrajectoryOptimization

function backwardpass_foh!(X,U,solver::Solver)
    N = solver.N; n = solver.model.n; m = solver.model.m;
    W = solver.obj.Q; Wf = solver.obj.Qf; xf = solver.obj.xf;
    R = getR(solver)

    S = zeros(n,n,N)
    s = zeros(n,N)
    K = zeros(m,n,N)
    d = zeros(m,N)

    # Terminal Values
    S[:,:,N] = Wf
    s[:,N] = Wf*(X[:,N] - xf)

    # v1 = 0.0
    # v2 = 0.0
    #
    k = N-1
    # Loop backwards
    while k >= 1

        #fx,fu = res.fx[:,:,k], res.fu[:,:,k]
        fx = ones(n,n)
        fu = ones(n,m)
        fu_ = copy(fu) # FIX Compute gradients of the dynamics


        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = W*(X[:,k] - xf) + fx'*s[:,k+1]
        Qu = R*U[:,k] + fu'*s[:,k+1]
        Qu_ = R*U[:,k+1] + fu_'*s[:,k+1]
        Qxx = W + fx'*S[:,:,k+1]*fx
        Quu = R + fu'*S[:,:,k+1]*fu
        Qu_u_ = R + fu_'*S[:,:,k+1]*fu_
        Qxu = fx'*S[:,:,k+1]*fu
        Qxu_ = fx'*S[:,:,k+1]*fu_
        Qux = fu'*S[:,:,k+1]*fx
        Quu_ = fu'*S[:,:,k+1]*fu_
        Qu_x = fu_'*S[:,:,k+1]*fx
        Qu_u = fu_'*S[:,:,k+1]*fu

        # Compute gains
        alpha = -Qu_u_\Qu_x
        beta = -Qu_u_\Qu_u
        gamma = -Qu_u_\Qu_

        K[:,:,k] = -(Quu + beta'*Qu_u_*beta + Quu_*beta + beta'*Qu_u)\(beta'*Qu_u_*alpha + Qux + beta'*Qu_x + Quu_*alpha)
        d[:,k] = -(Quu + beta'*Qu_u_*beta + Quu_*beta + beta'*Qu_u)\(Qu + Qu_*beta + beta'*Qu_u_*gamma + Quu_*gamma)
        if k == N-1
            K[:,:,k+1] = (alpha + beta*K[:,:,k])
            d[:,k+1] = (beta*d[:,k] + gamma)
        end
        S[:,:,k] = (Qxx + K[:,:,k]'*Quu*K[:,:,k] + K[:,:,k+1]'*Qu_u_*K[:,:,k+1] + Qxu*K[:,:,k] + Qxu_*K[:,:,k+1] + K[:,:,k]'*Qux + K[:,:,k+1]'*Qu_x + K[:,:,k]'*Quu_*K[:,:,k+1] + K[:,:,k+1]'*Qu_u*K[:,:,k])
        s[:,k] = (Qx' + Qu*K[:,:,k] + Qu_*K[:,:,k+1] + d[:,k]'*Quu*K[:,:,k] + d[:,k+1]'*Qu_u_*K[:,:,k+1] + d[:,k]'*Qux + d[:,k+1]'*Qu_x + d[:,k]'*Quu_*K[:,:,k] + d[:,k+1]'*Qu_u*K[:,:,k])''

        k = k - 1;
    end

    return K, d
end

opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache=false
opts.c1=1e-4
opts.c2=2.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-5
opts.iterations_outerloop = 250
opts.iterations = 1000

obj_uncon = TrajectoryOptimization.Dynamics.pendulum[2]
obj_uncon.R[:] = [1e-2] # control needs to be properly regularized for infeasible start to produce a good warm-start control output
solver = TrajectoryOptimization.Solver(Dynamics.pendulum![1], obj_uncon, dt=0.1, opts=opts)
U = rand(solver.model.m, solver.N-1)

tmp = TrajectoryOptimization.UnconstrainedResults(solver.model.n,solver.model.m,solver.N)
tmp.U[:,:] = U # store infeasible control output

rollout!(tmp,solver)
tmp.X
calc_jacobians(tmp,solver)
K,d = backwardpass_foh!(tmp.X,ones(solver.model.m,solver.N),solver)

tmp.S
tmp.d

plot(U')
