# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Forward and Backward passes for iLQR algorithm
#
#     backwardpass!: iLQR backward pass
#     backwardpass_sqrt: iLQR backward pass with Cholesky Factorization of
#        Cost-to-Go
#     backwardpass_foh!: iLQR backward pass for first order hold on controls
#     chol_minus: Calculate sqrt(A-B)
#     forwardpass!: iLQR forward pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
$(SIGNATURES)
Solve the dynamic programming problem, starting from the terminal time step
Computes the gain matrices K and d by applying the principle of optimality at
each time step, solving for the gradient (s) and Hessian (S) of the cost-to-go
function. Also returns parameters `v1` and `v2` (see Eq. 25a in Yuval Tassa Thesis)
"""
function backwardpass!(res::SolverIterResults,solver::Solver)
    N = solver.N; n = solver.model.n; m = solver.model.m;
    Q = solver.obj.Q; Qf = solver.obj.Qf; xf = solver.obj.xf;
    R = getR(solver)
    dt = solver.dt

    if solver.model.m != size(res.U,1)
        m += n
    end

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s

    # Boundary Conditions
    S[:,:,N] = Qf
    s[:,N] = Qf*(X[:,N] - xf)

    # Initialize expected change in cost-to-go
    Δv = 0.

    # Terminal constraints
    if res isa ConstrainedResults
        C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA
        CxN = res.Cx_N
        S[:,:,N] += CxN'*res.IμN*CxN
        s[:,N] += CxN'*res.IμN*res.CN + CxN'*res.λN
    end

    mu = res.mu_reg
    k = N-1

    # Backward pass
    while k >= 1
        lx = dt*Q*(X[:,k] - xf)
        lu = dt*R*(U[:,k])
        lxx = dt*Q
        luu = dt*R

        # Compute gradients of the dynamics
        fx, fu = res.fx[:,:,k], res.fu[:,:,k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = lx + fx'*s[:,k+1]
        Qu = lu + fu'*s[:,k+1]
        Qxx = lxx + fx'*S[:,:,k+1]*fx

        Quu = Hermitian(luu + fu'*S[:,:,k+1]*fu + mu[1]*I)
        Qux = fu'*S[:,:,k+1]*fx

        # Constraints
        if res isa ConstrainedResults
            Cx, Cu = res.Cx[:,:,k], res.Cu[:,:,k]
            Qx += (Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k])
            Qu += (Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k])
            Qxx += Cx'*Iμ[:,:,k]*Cx
            Quu += Cu'*Iμ[:,:,k]*Cu
            Qux += Cu'*Iμ[:,:,k]*Cx
        end

        # regularization
        # if !isposdef(Quu)
        #     if size(Quu,1) == 1
        #         mu[1] += -2.0*Quu[1]
        #     else
        #         mu[1] += -2.0*minimum(eigvals(Quu))
        #     end
        #     # mu[1] += solver.opts.mu_reg_update
        #     k = N-1
        #     if solver.opts.verbose
        #         println("regularized (normal bp)")
        #     end
        #     continue
        # end

        # Compute gains
        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s[:,k] = Qx - Qux'd[:,k]
        S[:,:,k] = Qxx - Qux'K[:,:,k]

        Δv += 1.5*vec(Qu)'*vec(d[:,k])

        k = k - 1;
    end

    return Δv
end

function chol_plus(A,B)
    n1,m = size(A)
    n2 = size(B,1)
    P = zeros(n1+n2,m)
    P[1:n1,:] = A
    P[n1+1:end,:] = B
    qr(P)
end

"""
$(SIGNATURES)
Perform a backwards pass with Cholesky Factorizations of the Cost-to-Go to
avoid ill-conditioning.
"""
function backwardpass_sqrt!(res::SolverResults,solver::Solver)

    N = solver.N
    n = solver.model.n
    m = solver.model.m

    if solver.model.m != size(res.U,1)
        m += n
    end

    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf
    dt = solver.dt

    Uq = cholesky(Q).U
    Ur = cholesky(R).U

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; Su = res.S; s = res.s

    # Terminal Cost-to-go
    if isa(solver.obj, ConstrainedObjective)
        Cx = res.Cx_N
        Su[:,:,N] = cholesky(Qf + Cx'*res.IμN*Cx).U
        s[:,N] = Qf*(X[:,N] - xf) + Cx'*res.IμN*res.CN + Cx'*res.λN
    else
        Su[:,:,N] = cholesky(Qf).U
        s[:,N] = Qf*(X[:,N] - xf)
    end

    # Initialization of expected change in cost-to-go
    Δv = 0.

    mu = res.mu_reg
    k = N-1

    # Backward pass
    while k >= 1
        lx = dt*Q*(X[:,k] - xf)
        lu = dt*R*(U[:,k])
        lxx = dt*Q
        luu = dt*R

        fx, fu = res.fx[:,:,k], res.fu[:,:,k]

        Qx = lx + fx'*s[:,k+1]
        Qu = lu + fu'*s[:,k+1]

        Wxx = chol_plus(Su[:,:,k+1]*fx, cholesky(lxx).U)
        Wuu = chol_plus(Su[:,:,k+1]*fu, cholesky(luu).U + mu[1]*I)
        # Wxx = qrfact!([Su[:,:,k+1]*fx; chol(lxx)])
        # Wuu = qrfact!([Su[:,:,k+1]*fu; chol(luu) + mu[1]*I])
        Qxu = (fx'*Su[:,:,k+1]')*(Su[:,:,k+1]*fu)

        if isa(solver.obj, ConstrainedObjective)
            # Constraints
            Iμ = res.Iμ; C = res.C; LAMBDA = res.LAMBDA;
            Cx, Cu = res.Cx[:,:,k], res.Cu[:,:,k]
            Iμ2 = sqrt.(Iμ[:,:,k])
            Qx += (Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k])
            Qu += (Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k])
            Qxu += Cx'*Iμ[:,:,k]*Cu

            Wxx = chol_plus(Wxx.R, Iμ2*Cx)
            Wuu = chol_plus(Wxx.R, Iμ2*Cu)
        end

        K[:,:,k] = Wuu.R\(Wuu.R'\Qxu')
        d[:,k] = Wuu.R\(Wuu.R'\Qu)

        s[:,k] = Qx - Qxu*(Wuu.R\(Wuu.R'\Qu))

        try  # Regularization
            Su[:,:,k] = chol_minus(Wxx.R,(Wuu.R')\(Qxu'))
        catch ex
            # if ex isa LinAlg.PosDefException
            #     mu[1] += -2.0*minimum(eigvals(Wxx[:R]))
            #     k = N-1
            #     println("*regularization not implemented") #TODO fix regularization
            #     continue
            # end
            error("(sqrt bp) regularization not implemented")
        end

        # Expected change in cost-to-go
        Δv += 1.5*vec(Qu)'*vec(d[:,k])

        k = k - 1;
    end

    return Δv
end

function backwardpass_foh!(res::SolverIterResults,solver::Solver)
    N = solver.N
    n = solver.model.n
    m = solver.model.m

    if solver.model.m != size(res.U,1)
        m += n
    end

    dt = solver.dt

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

    # Initialization of expected change in cost-to-go
    Δv = 0.

    mu = res.mu_reg

    # Boundary conditions
    S[1:n,1:n] = Qf
    s[1:n] = Qf*(X[:,N]-xf)

    # Terminal constraints
    if res isa ConstrainedResults
        C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA
        CxN = res.Cx_N
        S[1:n,1:n] += CxN'*res.IμN*CxN
        s[1:n] += CxN'*res.IμN*res.CN + CxN'*res.λN
    end

    k = N-1
    while k >= 1
        ## Calculate the L(x,u,y,v) second order expansion

        # Unpack Jacobians, ̇x
        Ac1, Bc1 = res.Ac[:,:,k], res.Bc[:,:,k]
        Ac2, Bc2 = res.Ac[:,:,k+1], res.Bc[:,:,k+1]
        Ad, Bd, Cd = res.fx[:,:,k], res.fu[:,:,k], res.fv[:,:,k]

        xdot1 = res.xdot[:,k]
        xdot2 = res.xdot[:,k+1]

        xm = 0.5*X[:,k] + dt/8.0*xdot1 + 0.5*X[:,k+1] - dt/8.0*xdot2
        um = (U[:,k] + U[:,k+1])/2.0

        # Expansion of stage cost L(x,u,y,v) -> dL(dx,du,dy,dv)
        Lx = dt/6*Q*(X[:,k] - xf) + 4*dt/6*(0.5*eye(n) + dt/8*Ac1)'*Q*(xm - xf)
        Lu = dt/6*R*U[:,k] + 4*dt/6*((dt/8*Bc1)'*Q*(xm - xf) + 0.5*R*um)
        Ly = dt/6*Q*(X[:,k+1] - xf) + 4*dt/6*(0.5*eye(n) - dt/8*Ac2)'*Q*(xm - xf)
        Lv = dt/6*R*U[:,k+1] + 4*dt/6*((-dt/8*Bc2)'*Q*(xm - xf) + 0.5*R*um)

        Lxx = dt/6*Q + 4*dt/6*(0.5*eye(n) + dt/8*Ac1)'*Q*(0.5*eye(n) + dt/8*Ac1)
        Luu = dt/6*R + 4*dt/6*((dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5)
        Lyy = dt/6*Q + 4*dt/6*(0.5*eye(n) - dt/8*Ac2)'*Q*(0.5*eye(n) - dt/8*Ac2)
        Lvv = dt/6*R + 4*dt/6*((-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)

        Lxu = 4*dt/6*(0.5*eye(n) + dt/8*Ac1)'*Q*(dt/8*Bc1)
        Lxy = 4*dt/6*(0.5*eye(n) + dt/8*Ac1)'*Q*(0.5*eye(n) - dt/8*Ac2)
        Lxv = 4*dt/6*(0.5*eye(n) + dt/8*Ac1)'*Q*(-dt/8*Bc2)
        Luy = 4*dt/6*(dt/8*Bc1)'*Q*(0.5*eye(n) - dt/8*Ac2)
        Luv = 4*dt/6*((dt/8*Bc1)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)
        Lyv = 4*dt/6*(0.5*eye(n) - dt/8*Ac2)'*Q*(-dt/8*Bc2)

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

        if res isa ConstrainedResults
            if k == N-1
                Cy, Cv = res.Cx[:,:,k+1], res.Cu[:,:,k+1]
                Ly += (Cy'*Iμ[:,:,k+1]*C[:,k+1] + Cy'*LAMBDA[:,k+1])
                Lv += (Cv'*Iμ[:,:,k+1]*C[:,k+1] + Cv'*LAMBDA[:,k+1])
                Lyy += Cy'*Iμ[:,:,k+1]*Cy
                Lvv += Cv'*Iμ[:,:,k+1]*Cv
                Lyv += Cy'*Iμ[:,:,k+1]*Cv
            end
            Cx, Cu = res.Cx[:,:,k], res.Cu[:,:,k]
            Lx += (Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k])
            Lu += (Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k])
            Lxx += Cx'*Iμ[:,:,k]*Cx
            Luu += Cu'*Iμ[:,:,k]*Cu
            Lxu += Cx'*Iμ[:,:,k]*Cu
        end

        # Substitute in discrete dynamics dx = (Ad)dx + (Bd)du1 + (Cd)du2
        Qx = vec(Lx) + Ad'*vec(Ly)
        Qu = vec(Lu) + Bd'*vec(Ly)
        Qv = vec(Lv) + Cd'*vec(Ly)

        Qxx = Lxx + Lxy*Ad + Ad'*Lxy' + Ad'*Lyy*Ad
        Quu = Luu + Luy*Bd + Bd'*Luy' + Bd'*Lyy*Bd
        Qvv = Hermitian(Lvv + Lyv'*Cd + Cd'*Lyv + Cd'*Lyy*Cd + mu[1]*eye(m))
        Qxu = Lxu + Lxy*Bd + Ad'*Luy' + Ad'*Lyy*Bd
        Qxv = Lxv + Lxy*Cd + Ad'*Lyv + Ad'*Lyy*Cd
        Quv = Luv + Luy*Cd + Bd'*Lyv + Bd'*Lyy*Cd

        #TODO check regularization
        # regularization
        # if !isposdef(Qvv)
        #     # if size(Qvv,1) > 1
        #     #     mu[1] += -2.0*minimum(eigvals(Qvv))
        #     # else
        #     #     mu[1] += -2.0*Qvv[1]
        #     # end
        #     # # mu[1] += solver.opts.mu_reg_update
        #     # k = N-1
        #     if solver.opts.verbose
        #         println("regularized needed")
        #         # println("Qvv: $(Qvv)")
        #     end
        #     continue
        # end

        K[:,:,k+1] = -Qvv\Qxv'
        b[:,:,k+1] = -Qvv\Quv'
        d[:,k+1] = -Qvv\vec(Qv)

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
        Δv += -1.5*vec(Qv)'*vec(d[:,k+1])

        # at last time step, optimize over final control
        if k == 1
            K[:,:,1] = -Quu_\Qxu_'
            b[:,:,1] = zeros(m,m)
            d[:,1] = -Quu_\vec(Qu_)

            res.s[:,1] = Qx_ - Qxu_*vec(d[:,1])

            Δv += -1.5*vec(Qu_)'*vec(d[:,1])
        end

        k = k - 1;
    end

    return Δv
end

"""
$(SIGNATURES)
Perform the operation sqrt(A-B), where A and B are Symmetric Matrices
"""
function chol_minus(A,B::Matrix)
    AmB = Cholesky(A,:U,0)
    for i = 1:size(B,1)
        lowrankdowndate!(AmB,B[i,:])
    end
    U = AmB[:U]
end


"""
$(SIGNATURES)
Propagate dynamics with a line search (in-place)
"""
function forwardpass!(res::SolverIterResults, solver::Solver, Δv::Float64)

    # Pull out values from results
    X = res.X
    U = res.U
    K = res.K
    d = res.d
    X_ = res.X_
    U_ = res.U_

    # Compute original cost
    if res isa ConstrainedResults
        update_constraints!(res,solver,X,U)
    end
    J_prev = cost(solver, res, X, U)

    J = Inf
    alpha = 1.0
    iter = 0
    z = 0.

    while z ≤ solver.opts.c1 || z > solver.opts.c2

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            # set trajectories to original trajectory
            X_ .= X
            U_ .= U

            if res isa ConstrainedResults
                update_constraints!(res,solver,X_,U_)
            end
            J = copy(J_prev)
            z = 0.

            if solver.opts.verbose
                println("Max iterations (forward pass)\n -No improvement made")
            end
            alpha = 0.0
            break
        end

        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout!(res,solver,alpha)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            if solver.opts.verbose
                println("Non-finite values in rollout")
            end
            iter += 1
            alpha /= 2.0
            continue
        end

        # Calcuate cost
        if res isa ConstrainedResults
            update_constraints!(res,solver,X_,U_)
        end
        J = cost(solver, res, X_, U_)
        z = (J_prev - J)/(alpha*Δv)

        alpha /= 2.0

        iter += 1
    end

    if solver.opts.verbose
        println("New cost: $J")
        if res isa ConstrainedResults
            println("- state+control cost: $(cost(solver,X_,U_))")
            max_c = max_violation(res)
            println("- Max constraint violation: $max_c")
        end
        println("- Expected improvement (Δv): $(Δv)")
        println("- Actual improvement : $(J_prev-J)")
        println("- (z = $z, α = $(2.0*alpha)")
    end

    return J

end
