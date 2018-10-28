# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Forward and Backward passes for iLQR algorithm
#
#     _backwardpass!: iLQR backward pass
#     _backwardpass_sqrt: iLQR backward pass with Cholesky Factorization of
#        Cost-to-Go
#     _backwardpass_foh!: iLQR backward pass for first order hold on controls
#     chol_minus: Calculate sqrt(A-B)
#     forwardpass!: iLQR forward pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
using Test
"""
$(SIGNATURES)
    Solve the dynamic programming problem, starting from the terminal time step.
    Computes the gain matrices K and d by applying the principle of optimality at
    each time step, solving for the gradient (s) and Hessian (S) of the approximated quadratic cost-to-go
    function. Also returns parameters Δv for line search (see Synthesis and Stabilization of Complex Behaviors through
    Online Trajectory Optimization)
"""
function backwardpass!(results::SolverVectorResults,solver::Solver)
    if solver.control_integration == :foh
        Δv = _backwardpass_foh_min_time!(results,solver)
    elseif solver.opts.square_root
        Δv = _backwardpass_sqrt!(results, solver)
    else
        Δv = _backwardpass!(results, solver)
    end
    return Δv
end

function _backwardpass!(res::SolverVectorResults,solver::Solver)
    # Problem dimensions
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    # Objective parameters
    Q = solver.obj.Q; Qf = solver.obj.Qf; xf = solver.obj.xf; c = solver.obj.c;
    R = getR(solver)
    dt = solver.dt

    # Check for minimum time solve
    min_time = is_min_time(solver)

    # Pull out results
    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s

    # Boundary Conditions
    S[N] = Qf
    s[N] = Qf*(X[N] - xf)

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ
        CxN = res.Cx_N
        S[N] += CxN'*res.IμN*CxN
        s[N] += CxN'*res.IμN*res.CN + CxN'*res.λN
    end

    # Backward pass
    Δv = [0.0 0.0]
    k = N-1
    while k >= 1
        # Check for minimum time solve
        min_time ? dt = U[k][m̄]^2 : nothing

        # Calculate stage costs
        lx = dt*Q*vec(X[k] - xf)
        lu = dt*R*vec(U[k])
        lxx = dt*Q
        luu = dt*R

        # get discrete dynamics Jacobians
        fx, fu = res.fdx[k], res.fdu[k]

        # Form gradients and Hessians of Taylor Series Expansion of Q
        Qx = lx + fx'*vec(s[k+1])
        Qu = lu + fu'*vec(s[k+1])
        Qxx = lxx + fx'*S[k+1]*fx

        Quu = luu + fu'*S[k+1]*fu
        Qux = fu'*S[k+1]*fx

        # Constraints
        if res isa ConstrainedIterResults
            Cx, Cu = res.Cx[k], res.Cu[k]
            Qx += Cx'*Iμ[k]*C[k] + Cx'*λ[k]
            Qu += Cu'*Iμ[k]*C[k] + Cu'*λ[k]
            Qxx += Cx'*Iμ[k]*Cx
            Quu += Cu'*Iμ[k]*Cu
            Qux += Cu'*Iμ[k]*Cx

            if min_time
                h = U[k][m̄]
                Qu[m̄] += 2*h*stage_cost(X[k],U[k],Q,R,xf,c)
                Qux[m̄,1:n] += vec(2h*(X[k]-xf)'Q)
                tmp = zero(Quu)
                tmp[:,m̄] = R*U[k]
                Quu += 2h*(tmp+tmp')
                Quu[m̄,m̄] += 2*stage_cost(X[k],U[k],Q,R,xf,c)

                if k > 1
                    Qu[m̄] -= C[k-1][end]*Iμ[k-1][end,end] + λ[k-1][end]
                    Quu[m̄,m̄] += Iμ[k-1][end,end]
                end
            end
        end

        # Regularization
            # Note: it is critical to have a separate, regularized Quu, Qux for the gains and unregularized versions for S,s to propagate backward
        if solver.opts.regularization_type == :state
            Quu_reg = Quu + res.ρ[1]*fu'*fu
            Qux_reg = Qux + res.ρ[1]*fu'*fx
        elseif solver.opts.regularization_type == :control
            Quu_reg = Quu + res.ρ[1]*I
            Qux_reg = Qux
        end

        if rank(Quu_reg) != mm  # TODO determine if rank or PD is best check
            @logmsg InnerLoop "Regularized"
            # if solver.opts.verbose # TODO: switch to logger
            #     println("regularized (normal bp)")
            #     println("-condition number: $(cond(Array(Quu_reg)))")
            #     println("Quu_reg: $(eigvals(Quu_reg))")
            #     @show Quu_reg
            # end

            # Increase regularization
            regularization_update!(res,solver,:increase)

            # Reset backward pass
            k = N-1
            Δv = [0.0 0.0]
            continue
        end

        # Compute gains
        K[k] = -Quu_reg\Qux_reg
        d[k] = -Quu_reg\Qu

        # Calculate cost-to-go
        s[k] = vec(Qx) + K[k]'*Quu*vec(d[k]) + K[k]'*vec(Qu) + Qux'*vec(d[k])
        S[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # Calculated change is cost-to-go over entire trajectory
        Δv += [vec(d[k])'*vec(Qu) 0.5*vec(d[k])'*Quu*vec(d[k])]

        k = k - 1;
    end

    # Decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

"""
$(SIGNATURES)
Perform a backwards pass with Cholesky Factorizations of the Cost-to-Go to
avoid ill-conditioning.
"""
function _backwardpass_sqrt!(res::SolverVectorResults,solver::Solver)
    N = solver.N
    n = solver.model.n
    m = solver.model.m

    if solver.model.m != length(res.U[1])
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
        Su[N] = cholesky(Qf + Cx'*res.IμN*Cx).U
        s[N] = Qf*(X[N] - xf) + Cx'*res.IμN*res.CN + Cx'*res.λN
    else
        Su[N] = cholesky(Qf).U
        s[N] = Qf*(X[N] - xf)
    end

    # Initialization of expected change in cost-to-go
    Δv = [0. 0.]

    k = N-1

    # Backward pass
    while k >= 1
        lx = dt*Q*(X[k] - xf)
        lu = dt*R*(U[k])
        lxx = dt*Q
        luu = dt*R

        fx, fu = res.fdx[k], res.fdu[k]

        Qx = lx + fx'*s[k+1]
        Qu = lu + fu'*s[k+1]

        Wxx = chol_plus(Su[k+1]*fx, cholesky(lxx).U)
        Wuu = chol_plus(Su[k+1]*fu, cholesky(luu).U)

        Qxu = (fx'*Su[k+1]')*(Su[k+1]*fu)

        # Constraints
        if isa(solver.obj, ConstrainedObjective)
            Iμ = res.Iμ; C = res.C; λ = res.λ;
            Cx, Cu = res.Cx[k], res.Cu[k]
            Iμ2 = sqrt.(Iμ[k])
            Qx += (Cx'*Iμ[k]*C[k] + Cx'*λ[k])
            Qu += (Cu'*Iμ[k]*C[k] + Cu'*λ[k])
            Qxu += Cx'*Iμ[k]*Cu

            Wxx = chol_plus(Wxx.R, Iμ2*Cx)
            Wuu = chol_plus(Wuu.R, Iμ2*Cu)
        end

        K[k] = -Wuu.R\(Array(Wuu.R')\Array(Qxu'))
        d[k] = -Wuu.R\(Wuu.R'\Qu)

        s[k] = Qx - Qxu*(Wuu.R\(Wuu.R'\Qu))

        try  # Regularization
            Su[k] = chol_minus(Wxx.R,(Array(Wuu.R'))\Array(Qxu'))
        catch ex
            error("sqrt bp not implemented")
        end

        # Expected change in cost-to-go
        Δv += [vec(Qu)'*vec(d[k]) 0.5*vec(d[k])'*Wuu.R'*Wuu.R*vec(d[k])]

        k = k - 1;
    end

    return Δv
end

function _backwardpass_foh!(res::SolverVectorResults,solver::Solver)
    n, m, N = get_sizes(solver)

    # Check for infeasible start
    if solver.model.m != length(res.U[1])
        m += n
    end

    dt = solver.dt

    Q = solver.obj.Q
    R = getR(solver)
    Qf = solver.obj.Qf
    xf = solver.obj.xf

    K = res.K
    b = res.b
    d = res.d

    X = res.X
    U = res.U

    # Boundary conditions
    S = zeros(n+m,n+m)
    s = zeros(n+m)
    S[1:n,1:n] = Qf
    s[1:n] = Qf*(X[N]-xf)

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ
        CxN = res.Cx_N
        S[1:n,1:n] += CxN'*res.IμN*CxN
        s[1:n] += CxN'*res.IμN*res.CN + CxN'*res.λN

        k = N-1
        Cy, Cv = res.Cx[k+1], res.Cu[k+1]
        s[1:n] += Cy'*Iμ[k+1]*C[k+1] + Cy'*λ[k+1]
        s[n+1:n+m] += Cv'*Iμ[k+1]*C[k+1] + Cv'*λ[k+1]
        S[1:n,1:n] += Cy'*Iμ[k+1]*Cy
        S[n+1:n+m,n+1:n+m] += Cv'*Iμ[k+1]*Cv
        S[1:n,n+1:n+m] += Cy'*Iμ[k+1]*Cv
        S[n+1:n+m,1:n] += Cv'*Iμ[k+1]*Cy
    end

    # create a copy of BC in case of regularization
    SN = copy(S)
    sN = copy(s)

    # Backward pass
    k = N-1
    Δv = [0. 0.] # Initialization of expected change in cost-to-go
    while k >= 1
        ## Calculate the L(x,u,y,v) second order expansion
        # Unpack Jacobians, ̇x
        fcx, fcu = res.fcx[k], res.fcu[k]
        fcy, fcv = res.fcx[k+1], res.fcu[k+1]
        fdx, fdu, fdv = res.fdx[k], res.fdu[k], res.fdv[k]

        xm = res.xm[k]
        um = (U[k] + U[k+1])/2.

        # Expansion of stage cost L(x,u,y,v) -> dL(dx,du,dy,dv)
        Lx = dt/6*Q*(X[k] - xf) + 4*dt/6*(I/2 + dt/8*fcx)'*Q*(xm - xf)
        Lu = dt/6*R*U[k] + 4*dt/6*((dt/8*fcu)'*Q*(xm - xf) + 0.5*R*um)
        Ly = dt/6*Q*(X[k+1] - xf) + 4*dt/6*(I/2 - dt/8*fcy)'*Q*(xm - xf)
        Lv = dt/6*R*U[k+1] + 4*dt/6*((-dt/8*fcv)'*Q*(xm - xf) + 0.5*R*um)

        Lxx = dt/6*Q + 4*dt/6*(I/2 + dt/8*fcx)'*Q*(I/2 + dt/8*fcx)
        Luu = dt/6*R + 4*dt/6*((dt/8*fcu)'*Q*(dt/8*fcu) + 0.5*R*0.5)
        Lyy = dt/6*Q + 4*dt/6*(I/2 - dt/8*fcy)'*Q*(I/2 - dt/8*fcy)
        Lvv = dt/6*R + 4*dt/6*((-dt/8*fcv)'*Q*(-dt/8*fcv) + 0.5*R*0.5)

        Lxu = 4*dt/6*(I/2 + dt/8*fcx)'*Q*(dt/8*fcu)
        Lxy = 4*dt/6*(I/2 + dt/8*fcx)'*Q*(I/2 - dt/8*fcy)
        Lxv = 4*dt/6*(I/2 + dt/8*fcx)'*Q*(-dt/8*fcv)
        Luy = 4*dt/6*(dt/8*fcu)'*Q*(I/2 - dt/8*fcy)
        Luv = 4*dt/6*((dt/8*fcu)'*Q*(-dt/8*fcv) + 0.5*R*0.5)
        Lyv = 4*dt/6*(I/2 - dt/8*fcy)'*Q*(-dt/8*fcv)

        # Constraints
        if res isa ConstrainedIterResults
            # Cy, Cv = res.Cx[k+1], res.Cu[k+1]
            # Ly += Cy'*Iμ[k+1]*C[k+1] + Cy'*λ[k+1]
            # Lv += Cv'*Iμ[k+1]*C[k+1] + Cv'*λ[k+1]
            # Lyy += Cy'*Iμ[k+1]*Cy
            # Lvv += Cv'*Iμ[k+1]*Cv
            # Lyv += Cy'*Iμ[k+1]*Cv
            println("uh what?")
            Cx, Cu = res.Cx[k], res.Cu[k]
            Lx += Cx'*Iμ[k]*C[k] + Cx'*λ[k]
            Lu += Cu'*Iμ[k]*C[k] + Cu'*λ[k]
            Lxx += Cx'*Iμ[k]*Cx
            Luu += Cu'*Iμ[k]*Cu
            Lxu += Cx'*Iμ[k]*Cu
        end

        # Unpack cost-to-go P
        Sy = s[1:n]
        Sv = s[n+1:n+m]
        Syy = S[1:n,1:n]
        Svv = S[n+1:n+m,n+1:n+m]
        Syv = S[1:n,n+1:n+m]

        # Substitute in discrete dynamics (second order approximation)
        Qx = vec(Lx) + fdx'*vec(Ly) + fdx'*vec(Sy)
        Qu = vec(Lu) + fdu'*vec(Ly) + fdu'*vec(Sy)
        Qv = vec(Lv) + fdv'*vec(Ly) + fdv'*vec(Sy) + Sv

        Qxx = Lxx + Lxy*fdx + fdx'*Lxy' + fdx'*Lyy*fdx + fdx'*Syy*fdx
        Quu = Luu + Luy*fdu + fdu'*Luy' + fdu'*Lyy*fdu + fdu'*Syy*fdu
        Qvv = Lvv + Lyv'*fdv + fdv'*Lyv + fdv'*Lyy*fdv + fdv'*Syy*fdv + fdv'*Syv + Syv'*fdv + Svv
        Qxu = Lxu + Lxy*fdu + fdx'*Luy' + fdx'*Lyy*fdu + fdx'*Syy*fdu
        Qxv = Lxv + Lxy*fdv + fdx'*Lyv + fdx'*Lyy*fdv + fdx'*Syy*fdv + fdx'*Syv
        Quv = Luv + Luy*fdv + fdu'*Lyv + fdu'*Lyy*fdv + fdu'*Syy*fdv + fdu'*Syv

        # Regularization
        #TODO double check state regularization
        # if solver.opts.regularization_type == :state
        #     Qvv_reg = Qvv + res.ρ[1]*fdv'*fdv
        #     Qxv_reg = Qxv + res.ρ[1]*fdx'*fdv
        #     Quv_reg = Quv + res.ρ[1]*fdu'*fdv
        # elseif solver.opts.regularization_type == :control
            Qvv_reg = Qvv + res.ρ[1]*I
            Qxv_reg = Qxv
            Quv_reg = Quv
        # end

        if !isposdef(Hermitian(Array(Qvv_reg)))
            # @logmsg InnerLoop "Regularized"
            # if solver.opts.verbose  # TODO move to logger
            #     println("regularized (foh bp)\n not implemented properly")
            #     println("-condition number: $(cond(Array(Qvv_reg)))")
            #     println("Qvv_reg: $Qvv_reg")
            #     println("iteration: $k")
            # end

            regularization_update!(res,solver,:increase)

            # Reset BCs
            # S = zeros(n+m,n+m)
            # s = zeros(n+m)
            # S[1:n,1:n] = Qf
            # s[1:n] = Qf*(X[N]-xf)
            #
            # # Terminal constraints
            # if res isa ConstrainedIterResults
            #     C = res.C; Iμ = res.Iμ; λ = res.λ
            #     CxN = res.Cx_N
            #     S[1:n,1:n] += CxN'*res.IμN*CxN
            #     s[1:n] += CxN'*res.IμN*res.CN + CxN'*res.λN
            # end
            S = SN
            s = sN
            ############
            k = N-1
            Δv = [0. 0.]
            continue
        end

        # calculate gains
        K[k+1] = -Qvv_reg\Qxv_reg'
        b[k+1] = -Qvv_reg\Quv_reg'
        d[k+1] = -Qvv_reg\vec(Qv)

        # calculate optimized values
        Qx_ = vec(Qx) + K[k+1]'*vec(Qv) + Qxv*vec(d[k+1]) + K[k+1]'Qvv*d[k+1]
        Qu_ = vec(Qu) + b[k+1]'*vec(Qv) + Quv*vec(d[k+1]) + b[k+1]'*Qvv*d[k+1]
        Qxx_ = Qxx + Qxv*K[k+1] + K[k+1]'*Qxv' + K[k+1]'*Qvv*K[k+1]
        Quu_ = Quu + Quv*b[k+1] + b[k+1]'*Quv' + b[k+1]'*Qvv*b[k+1]
        Qxu_ = Qxu + K[k+1]'*Quv' + Qxv*b[k+1] + K[k+1]'*Qvv*b[k+1]

        # cache (approximate) cost-to-go at timestep k
        s[1:n] = Qx_
        s[n+1:n+m] = Qu_
        S[1:n,1:n] = Qxx_
        S[n+1:n+m,n+1:n+m] = Quu_
        S[1:n,n+1:n+m] = Qxu_
        S[n+1:n+m,1:n] = Qxu_'

        # line search terms
        Δv += [vec(Qv)'*vec(d[k+1]) 0.5*vec(d[k+1])'*Qvv*vec(d[k+1])]

        # at last time step, optimize over final control
        if k == 1
            # if res isa ConstrainedIterResults
            #     Cx, Cu = res.Cx[k], res.Cu[k]
            #     Qx_ += Cx'*Iμ[k]*C[k] + Cx'*λ[k]
            #     Qu_ += Cu'*Iμ[k]*C[k] + Cu'*λ[k]
            #     Qxx_ += Cx'*Iμ[k]*Cx
            #     Quu_ += Cu'*Iμ[k]*Cu
            #     Qxu_ += Cx'*Iμ[k]*Cu
            # end

            # regularize Quu_
            Quu__reg = Quu_ + res.ρ[1]*I

            if !isposdef(Array(Hermitian(Quu__reg)))
                # @logmsg InnerLoop "Regularized"
                # if solver.opts.verbose  # TODO: Move to logger
                #     println("regularized (foh bp)")
                #     println("part 2")
                #     println("-condition number: $(cond(Array(Quu__reg)))")
                #     println("Quu__reg: $Quu__reg")
                # end

                regularization_update!(res,solver,:increase)

                ## Reset BCs ##
                # S = zeros(n+m,n+m)
                # s = zeros(n+m)
                # S[1:n,1:n] = Qf
                # s[1:n] = Qf*(X[N]-xf)
                #
                # # Terminal constraints
                # if res isa ConstrainedIterResults
                #     C = res.C; Iμ = res.Iμ; λ = res.λ
                #     CxN = res.Cx_N
                #     S[1:n,1:n] += CxN'*res.IμN*CxN
                #     s[1:n] += CxN'*res.IμN*res.CN + CxN'*res.λN
                # end
                S = SN
                s = sN
                ################
                k = N-1
                Δv = [0. 0.]
                continue
            end

            K[1] = -Quu__reg\Qxu_'
            b[1] = zeros(m,m)
            d[1] = -Quu__reg\vec(Qu_)

            res.s[1] = vec(Qx_) + K[1]'*Quu_*vec(d[1]) + K[1]'*vec(Qu_) + Qxu_*vec(d[1]) # calculate for gradient check in solve

            Δv += [vec(Qu_)'*vec(d[1]) 0.5*vec(d[1])'*Quu_*vec(d[1])]

        end

        k = k - 1;
    end

    # if successful backward pass, reduce regularization
    regularization_update!(res,solver,:decrease)
    return Δv
end

function _backwardpass_foh_min_time!(res::SolverVectorResults,solver::Solver)
    # Problem dimensions
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    # Objective parameters
    Q = solver.obj.Q; Qf = solver.obj.Qf; xf = solver.obj.xf; c = solver.obj.c;
    R = getR(solver)
    dt = solver.dt

    # Check for minimum time solve
    min_time = is_min_time(solver)

    # Pull out results
    X = res.X; U = res.U; K = res.K; b = res.b; d = res.d; S = res.S; s = res.s

    # Boundary conditions
    S = zeros(n+mm,n+mm)
    s = zeros(n+mm)
    S[1:n,1:n] = Qf
    s[1:n] = Qf*(X[N]-xf)

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ
        CxN = res.Cx_N
        S[1:n,1:n] += CxN'*res.IμN*CxN
        s[1:n] += CxN'*res.IμN*res.CN + CxN'*res.λN

        # We include the the k = N expansions here for a cleaner backward pass
        Cy, Cv = res.Cx[N], res.Cu[N]
        s[1:n] += Cy'*Iμ[N]*C[N] + Cy'*λ[N]
        s[n+1:n+m] += Cv'*Iμ[N]*C[N] + Cv'*λ[N]
        S[1:n,1:n] += Cy'*Iμ[N]*Cy
        S[n+1:n+m,n+1:n+m] += Cv'*Iμ[N]*Cv
        S[1:n,n+1:n+m] += Cy'*Iμ[N]*Cv
        S[n+1:n+m,1:n] += Cv'*Iμ[N]*Cy
    end

    # create a copy of BC in case of regularization
    SN = copy(S)
    sN = copy(s)

    # Backward pass
    k = N-1
    Δv = [0. 0.] # Initialization of expected change in cost-to-go
    while k >= 1
        # Check for minimum time solve
        min_time ? dt = U[k][m̄]^2 : nothing

        # Unpack results
        fcx, fcu = res.fcx[k], res.fcu[k]
        fcy, fcv = res.fcx[k+1], res.fcu[k+1]
        fdx, fdu, fdv = res.fdx[k], res.fdu[k], res.fdv[k]

        x = res.X[k]
        y = res.X[k+1]
        u = res.U[k]
        v = res.U[k+1]
        xm = res.xm[k]
        um = (U[k] + U[k+1])/2.
        dx = res.dx[k]
        dy = res.dx[k+1]

        ## L(x,u,y,v) = L(x,u) + L(xm,um) + L(y,v) = L1 + L2 + L3
        # Assembling ℓ(x,u) expansion
        ℓ1 = stage_cost(x,u,Q,R,xf)
        ℓ1x = Q*(x - xf)
        ℓ1u = R*u

        ℓ1xx = Q
        ℓ1uu = R

        # Assembling ℓ(xm,um) expansion
        ℓ2 = stage_cost(xm,um,Q,R,xf)
        ℓ2x = (I/2 + dt/8*fcx)'*Q*(xm - xf)
        ℓ2u = ((dt/8*fcu)'*Q*(xm - xf) + 0.5*R*um)
        ℓ2y = (I/2 - dt/8*fcy)'*Q*(xm - xf)
        ℓ2v = ((-dt/8*fcv)'*Q*(xm - xf) + 0.5*R*um)

        ℓ2xx = (I/2.0 + dt/8.0*fcx)'*Q*(I/2.0 + dt/8.0*fcx)
        ℓ2uu = ((dt/8*fcu)'*Q*(dt/8*fcu) + 0.5*R*0.5)
        ℓ2yy = (I/2 - dt/8*fcy)'*Q*(I/2 - dt/8*fcy)
        ℓ2vv = ((-dt/8*fcv)'*Q*(-dt/8*fcv) + 0.5*R*0.5)

        ℓ2xu = (I/2 + dt/8*fcx)'*Q*(dt/8*fcu)
        ℓ2xy = (I/2 + dt/8*fcx)'*Q*(I/2 - dt/8*fcy)
        ℓ2xv = (I/2 + dt/8*fcx)'*Q*(-dt/8*fcv)
        ℓ2uy = (dt/8*fcu)'*Q*(I/2 - dt/8*fcy)
        ℓ2uv = ((dt/8*fcu)'*Q*(-dt/8*fcv) + 0.5*R*0.5)  # note the name change; workspace conflict
        ℓ2yv = (I/2 - dt/8*fcy)'*Q*(-dt/8*fcv)

        # Assembling ℓ(y,v) expansion
        ℓ3 = stage_cost(y,v,Q,R,xf)

        ℓ3y = Q*(y - xf)
        ℓ3v = R*v

        ℓ3yy = Q
        ℓ3vv = R
        #
        # # Assemble results to form δL
        # if min_time
        #     println("Minimum time foh")
        #     h = u[m̄]
        #
        #     # Additional expansion terms
        #     dxm = 2*h/8*dx - 2*h/8*dy
        #
        #     L2h = 4/6*((h^2)*dxm'*Q*(xm - xf) + 2*h*ℓ2)
        #     L2hh = 4/6*(2/8*((h^3)*dxm'*Q*dx + 3*(h^2)*dx'*Q*xm) - 2/8*((h^3)*dxm'*Q*dy + 3*(h^2)*dy'*Q*xm) - 6*(h^2)/8*dx'*Q*xf + 6*(h^2)/8*dy'*Q*xf + 2*(h*dxm'*Q*(xm - xf) + ℓ2))
        #     L2hu = 4*(h^2)/6*(2*(h^3)/8*(Bc1'*Q*xm + (h^2)/8*Bc1'*Q*dx) - 2*(h^3)/8*Bc1'*Q*xf - 2*(h^5)/64*Bc1'*Q*dy + 2*h*ℓ2u)
        #
        #     L2xh = 2/6*Q*(h*x + h*y + (h^3)/2*dx - (h^3)/2*dy) - 4/6*h*Q*xf + 1/12*Ac1'*Q*(2*(h^3)*x + 2*(h^3)*y + 6/8*(h^5)*dx -6/8*(h^5)*dy) - (h^3)/3*Ac1'*Q*xf
        #     L2uh = 1/12*Bc1'*Q*(2*(h^3)*x + 2*(h^3)*y + 6/8*(h^5)*dx - 6/8*(h^5)*dy) - 1/3*(h^3)*Bc1'*Q*xf + 4/6*h*R*um
        #     L2yh = 2/6*Q*(h*x + h*y + (h^3)/2*dx - (h^3)/2*dy) - 4/6*h*Q*xf - 1/12*Ac2'*Q*(2*(h^3)*x + 2*(h^3)*y + 6/8*(h^5)*dx -6/8*(h^5)*dy) + (h^3)/3*Ac2'*Q*xf
        #     L2vh = -1/12*Bc2'*Q*(2*(h^3)*x + 2*(h^3)*y + 6/8*(h^5)*dx - 6/8*(h^5)*dy) + 1/3*(h^3)*Bc2'*Q*xf + 4/6*h*R*um
        #
        #     # Assemble expansion
        #     Lx = (h^2)/6*ℓ1x + 4/6*(h^2)*ℓ2x
        #     Lu = [(h^2)/6*ℓ1u + 4/6*(h^2)*ℓ2u; (2/6*h*ℓ1 + L2h + 2/6*ℓ3 + 2*c*h)]
        #     Ly = 4/6*(h^2)*ℓ2y + (h^2)/6*ℓ3y
        #     Lv = [4/6*(h^2)*ℓ2v + (h^2)/6*ℓ3v; 0]
        #
        #     Lxx = (h^2)/6*ℓ1xx + 4/6*(h^2)*ℓ2xx
        #     Luu = [((h^2)/6*ℓ1uu + 4/6*(h^2)*ℓ2uu) (2/6*h*ℓ1u + L2uh); (2/6*h*ℓ1u + L2hu)' (2/6*ℓ1 + L2hh + 2/6*ℓ3 + 2*c)]
        #     Lyy =  4/6*(h^2)*ℓ2yy + (h^2)/6*ℓ3yy
        #     Lvv = [(4/6*(h^2)*ℓ2vv + (h^2)/6*ℓ3vv) zeros(m); zeros(m)' 0]
        #
        #     Lxu = [((h^2)/6*ℓ1xu + 4/6*(h^2)*ℓ2xu) (2/6*h*ℓ1x + L2xh)]
        #     Lxy = 4/6*(h^2)*ℓ2xy
        #     Lxv = [4/6*(h^2)*ℓ2xv zeros(n)]
        #     Luy = [4/6*(h^2)*ℓ2uy' (L2yh + 2/6*h*ℓ3y)]'
        #     Luv = [4/6*(h^2)*ℓ2uv' (L2vh + 2/6*h*ℓ3v); zeros(m)' 0]'
        #     Lyv = [(4/6*(h^2)*ℓ2yv + (h^2)/6*ℓ3yv) zeros(n)]
        # else
            # _Lx = dt/6*ℓ1x + 4*dt/6*ℓ2x
            # _Lu = dt/6*ℓ1u + 4*dt/6*ℓ2u
            # _Ly = 4*dt/6*ℓ2y + dt/6*ℓ3y
            # _Lv = 4*dt/6*ℓ2v + dt/6*ℓ3v
            #
            # _Lxx = dt/6*ℓ1xx + 4*dt/6*ℓ2xx
            # _Luu = dt/6*ℓ1uu + 4*dt/6*ℓ2uu
            # _Lyy = 4*dt/6*ℓ2yy + dt/6*ℓ3yy
            # _Lvv = 4*dt/6*ℓ2vv + dt/6*ℓ3vv
            #
            # _Lxu = 4*dt/6*ℓ2xu
            # _Lxy = 4*dt/6*ℓ2xy
            # _Lxv = 4*dt/6*ℓ2xv
            # _Luy = 4*dt/6*ℓ2uy
            # _Luv = 4*dt/6*ℓ2uv
            # _Lyv = 4*dt/6*ℓ2yv
        # end
        # Lx = dt/6*Q*(X[k] - xf) + 4*dt/6*(I/2 + dt/8*fcx)'*Q*(xm - xf)
        # Lu = dt/6*R*U[k] + 4*dt/6*((dt/8*fcu)'*Q*(xm - xf) + 0.5*R*um)
        # Ly = dt/6*Q*(X[k+1] - xf) + 4*dt/6*(I/2 - dt/8*fcy)'*Q*(xm - xf)
        # Lv = dt/6*R*U[k+1] + 4*dt/6*((-dt/8*fcv)'*Q*(xm - xf) + 0.5*R*um)
        #
        # Lxx = dt/6*Q + 4*dt/6*(I/2 + dt/8*fcx)'*Q*(I/2 + dt/8*fcx)
        # Luu = dt/6*R + 4*dt/6*((dt/8*fcu)'*Q*(dt/8*fcu) + 0.5*R*0.5)
        # Lyy = dt/6*Q + 4*dt/6*(I/2 - dt/8*fcy)'*Q*(I/2 - dt/8*fcy)
        # Lvv = dt/6*R + 4*dt/6*((-dt/8*fcv)'*Q*(-dt/8*fcv) + 0.5*R*0.5)
        #
        # Lxu = 4*dt/6*(I/2 + dt/8*fcx)'*Q*(dt/8*fcu)
        # Lxy = 4*dt/6*(I/2 + dt/8*fcx)'*Q*(I/2 - dt/8*fcy)
        # Lxv = 4*dt/6*(I/2 + dt/8*fcx)'*Q*(-dt/8*fcv)
        # Luy = 4*dt/6*(dt/8*fcu)'*Q*(I/2 - dt/8*fcy)
        # Luv = 4*dt/6*((dt/8*fcu)'*Q*(-dt/8*fcv) + 0.5*R*0.5)
        # Lyv = 4*dt/6*(I/2 - dt/8*fcy)'*Q*(-dt/8*fcv)
        if min_time
            nothing
        else
            Lx = dt/6*ℓ1x + 4*dt/6*ℓ2x
            Lu = dt/6*ℓ1u + 4*dt/6*ℓ2u
            Ly = 4*dt/6*ℓ2y + dt/6*ℓ3y
            Lv = 4*dt/6*ℓ2v + dt/6*ℓ3v
            Lxx = dt/6*ℓ1xx + 4*dt/6*ℓ2xx
            Luu = dt/6*ℓ1uu + 4*dt/6*ℓ2uu
            Lyy = 4*dt/6*ℓ2yy + dt/6*ℓ3yy
            Lvv = 4*dt/6*ℓ2vv + dt/6*ℓ3vv

            Lxu = 4*dt/6*ℓ2xu
            Lxy = 4*dt/6*ℓ2xy
            Lxv = 4*dt/6*ℓ2xv
            Luy = 4*dt/6*ℓ2uy
            Luv = 4*dt/6*ℓ2uv
            Lyv = 4*dt/6*ℓ2yv
        end

        # @test isapprox(_Lx,Lx)
        # @test isapprox(_Lu,Lu)
        # @test isapprox(_Ly,Ly)
        # @test isapprox(_Lv,Lv)
        # @test isapprox(_Lxx, Lxx)
        # @test isapprox(_Luu,Luu)
        # @test isapprox(_Lyy,Lyy)
        # @test isapprox(_Lvv,Lvv)
        # @test isapprox(_Lxu,Lxu)
        # @test isapprox(_Lxy,Lxy)
        # @test isapprox(_Lxv,Lxv)
        # @test isapprox(_Luy,Luy)
        # @test isapprox(_Luv,Luv)
        # @test isapprox(_Lyv,Lyv)

        # Constraints
        if res isa ConstrainedIterResults
            # if k == N-1
            #     Cy, Cv = res.Cx[k+1], res.Cu[k+1]
            #     Ly += Cy'*Iμ[k+1]*C[k+1] + Cy'*λ[k+1]
            #     Lv += Cv'*Iμ[k+1]*C[k+1] + Cv'*λ[k+1]
            #     Lyy += Cy'*Iμ[k+1]*Cy
            #     Lvv += Cv'*Iμ[k+1]*Cv
            #     Lyv += Cy'*Iμ[k+1]*Cv
            # end

            Cx, Cu = res.Cx[k], res.Cu[k]
            Lx += Cx'*Iμ[k]*C[k] + Cx'*λ[k]
            Lu += Cu'*Iμ[k]*C[k] + Cu'*λ[k]
            Lxx += Cx'*Iμ[k]*Cx
            Luu += Cu'*Iμ[k]*Cu
            Lxu += Cx'*Iμ[k]*Cu
        end

        # Unpack cost-to-go P
        Sy = s[1:n]
        Sv = s[n+1:n+mm]
        Syy = S[1:n,1:n]
        Svv = S[n+1:n+mm,n+1:n+mm]
        Syv = S[1:n,n+1:n+mm]

        # Substitute in discrete dynamics (second order approximation)
        Qx = vec(Lx) + fdx'*vec(Ly) + fdx'*vec(Sy)
        Qu = vec(Lu) + fdu'*vec(Ly) + fdu'*vec(Sy)
        Qv = vec(Lv) + fdv'*vec(Ly) + fdv'*vec(Sy) + Sv

        Qxx = Lxx + Lxy*fdx + fdx'*Lxy' + fdx'*Lyy*fdx + fdx'*Syy*fdx
        Quu = Luu + Luy*fdu + fdu'*Luy' + fdu'*Lyy*fdu + fdu'*Syy*fdu
        Qvv = Lvv + Lyv'*fdv + fdv'*Lyv + fdv'*Lyy*fdv + fdv'*Syy*fdv + fdv'*Syv + Syv'*fdv + Svv
        Qxu = Lxu + Lxy*fdu + fdx'*Luy' + fdx'*Lyy*fdu + fdx'*Syy*fdu
        Qxv = Lxv + Lxy*fdv + fdx'*Lyv + fdx'*Lyy*fdv + fdx'*Syy*fdv + fdx'*Syv
        Quv = Luv + Luy*fdv + fdu'*Lyv + fdu'*Lyy*fdv + fdu'*Syy*fdv + fdu'*Syv

        # Regularization
        #TODO double check state regularization
        # if solver.opts.regularization_type == :state
        #     Qvv_reg = Qvv + res.ρ[1]*fdv'*fdv
        #     Qxv_reg = Qxv + res.ρ[1]*fdx'*fdv
        #     Quv_reg = Quv + res.ρ[1]*fdu'*fdv
        # elseif solver.opts.regularization_type == :control
            Qvv_reg = Qvv + res.ρ[1]*I
            Qxv_reg = Qxv
            Quv_reg = Quv
        # end

        if !isposdef(Hermitian(Array(Qvv_reg)))
            # @logmsg InnerLoop "Regularized"
            # if solver.opts.verbose  # TODO move to logger
            #     println("regularized (foh bp)\n not implemented properly")
            #     println("-condition number: $(cond(Array(Qvv_reg)))")
            #     println("Qvv_reg: $Qvv_reg")
            #     println("iteration: $k")
            # end

            regularization_update!(res,solver,:increase)

            # Reset BCs
            S = copy(SN)
            s = copy(sN)
            ############
            k = N-1
            Δv = [0. 0.]
            continue
        end

        # calculate gains
        K[k+1] = -Qvv_reg\Qxv_reg'
        b[k+1] = -Qvv_reg\Quv_reg'
        d[k+1] = -Qvv_reg\vec(Qv)

        # calculate optimized values
        Qx_ = vec(Qx) + K[k+1]'*vec(Qv) + Qxv*vec(d[k+1]) + K[k+1]'Qvv*d[k+1]
        Qu_ = vec(Qu) + b[k+1]'*vec(Qv) + Quv*vec(d[k+1]) + b[k+1]'*Qvv*d[k+1]
        Qxx_ = Qxx + Qxv*K[k+1] + K[k+1]'*Qxv' + K[k+1]'*Qvv*K[k+1]
        Quu_ = Quu + Quv*b[k+1] + b[k+1]'*Quv' + b[k+1]'*Qvv*b[k+1]
        Qxu_ = Qxu + K[k+1]'*Quv' + Qxv*b[k+1] + K[k+1]'*Qvv*b[k+1]

        # cache (approximate) cost-to-go at timestep k
        s[1:n] = Qx_
        s[n+1:n+mm] = Qu_
        S[1:n,1:n] = Qxx_
        S[n+1:n+mm,n+1:n+mm] = Quu_
        S[1:n,n+1:n+mm] = Qxu_
        S[n+1:n+mm,1:n] = Qxu_'

        # line search terms
        Δv += [vec(Qv)'*vec(d[k+1]) 0.5*vec(d[k+1])'*Qvv*vec(d[k+1])]

        # at last time step, optimize over final control
        if k == 1
            # if res isa ConstrainedIterResults
            #     Cx, Cu = res.Cx[k], res.Cu[k]
            #     Qx_ += Cx'*Iμ[k]*C[k] + Cx'*λ[k]
            #     Qu_ += Cu'*Iμ[k]*C[k] + Cu'*λ[k]
            #     Qxx_ += Cx'*Iμ[k]*Cx
            #     Quu_ += Cu'*Iμ[k]*Cu
            #     Qxu_ += Cx'*Iμ[k]*Cu
            # end

            # regularize Quu_
            Quu__reg = Quu_ + res.ρ[1]*I

            if !isposdef(Array(Hermitian(Quu__reg)))
                # @logmsg InnerLoop "Regularized"
                # if solver.opts.verbose  # TODO: Move to logger
                #     println("regularized (foh bp)")
                #     println("part 2")
                #     println("-condition number: $(cond(Array(Quu__reg)))")
                #     println("Quu__reg: $Quu__reg")
                # end

                regularization_update!(res,solver,:increase)

                ## Reset BCs ##
                S = copy(SN)
                s = copy(sN)
                ################
                k = N-1
                Δv = [0. 0.]
                continue
            end

            K[1] = -Quu__reg\Qxu_'
            b[1] = zeros(mm,mm)
            d[1] = -Quu__reg\vec(Qu_)

            res.s[1] = vec(Qx_) + K[1]'*Quu_*vec(d[1]) + K[1]'*vec(Qu_) + Qxu_*vec(d[1]) # calculate for gradient check in solve

            Δv += [vec(Qu_)'*vec(d[1]) 0.5*vec(d[1])'*Quu_*vec(d[1])]
        end

        k = k - 1;
    end

    # if successful backward pass, reduce regularization
    regularization_update!(res,solver,:decrease)
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
    U = AmB.U
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
Propagate dynamics with a line search (in-place)
"""
function forwardpass!(res::SolverIterResults, solver::Solver, Δv::Array{Float64,2})
    status = :progress_made

    # Pull out values from results
    X = res.X
    U = res.U
    K = res.K
    d = res.d
    X_ = res.X_
    U_ = res.U_

    # Compute original cost
    update_constraints!(res,solver,X,U)

    Ju_prev = _cost(solver, res, X, U)   # Unconstrained cost
    Jc_prev = cost_constraints(solver, res)  # constraint cost
    J_prev = Ju_prev + Jc_prev


    J = Inf
    alpha = 1.0
    iter = 0
    z = 0.
    expected = 0.

    logger = current_logger()
    # print_header(logger,InnerIters) #TODO: fix, this errored out
    @logmsg InnerIters :iter value=0
    @logmsg InnerIters :cost value=J_prev
    # print_row(logger,InnerIters) #TODO: fix, same issue
    while z ≤ solver.opts.c1 || z > solver.opts.c2

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            Ju = _cost(solver, res, X_, U_)   # Unconstrained cost
            Jc = cost_constraints(solver, res)  # constraint cost

            # set trajectories to original trajectory
            X_ .= X
            U_ .= U

            # Examine components
            zu = (Ju_prev - Ju)
            zc = (Jc_prev - Jc)

            status = :no_progress_made
            if zu > 0
                status = :uncon_progress_made
            end
            if zc > 0
                status = :constraint_progress_made
            end

            if solver.control_integration == :foh
                calculate_derivatives!(res, solver, X_, U_)
                calculate_midpoints!(res, solver, X_, U_)
            end

            update_constraints!(res,solver,X_,U_)
            J = copy(J_prev)
            z = 0.
            alpha = 0.0
            expected = 0.

            @logmsg InnerLoop "Max iterations (forward pass) -No improvement made"
            regularization_update!(res,solver,:increase) # increase regularization
            res.ρ[1] *= solver.opts.ρ_forwardpass
            break
        end

        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout!(res,solver,alpha)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            @logmsg InnerIters "Non-finite values in rollout"
            iter += 1
            alpha /= 2.0
            continue
        end

        # Calcuate cost
        update_constraints!(res,solver,X_,U_)

        Ju = _cost(solver, res, X_, U_)   # Unconstrained cost
        Jc = cost_constraints(solver, res)  # constraint cost
        J = Ju + Jc
        expected = -alpha*(Δv[1] + alpha*Δv[2])
        if expected > 0
            z  = (J_prev - J)/expected
        else
            @logmsg InnerIters "Non-positive expected decrease"
            z = -1
        end

        iter += 1
        alpha /= 2.0

        # Log messages
        @logmsg InnerIters :iter value=iter
        @logmsg InnerIters :α value=2*alpha
        @logmsg InnerIters :cost value=J
        @logmsg InnerIters :z value=z
        # print_row(logger,InnerIters)

    end  # forward pass loop

    if res isa ConstrainedIterResults
        # @logmsg :scost value=cost(solver,res,res.X,res.U,true)
        @logmsg InnerLoop :c_max value=max_violation(res)
    end
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ value=J_prev-J
    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*alpha
    @logmsg InnerLoop :ρ value=res.ρ[1]

    return J
end
