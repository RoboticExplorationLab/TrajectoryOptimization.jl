abstract type BackwardPass end

struct BackwardPassZOH <: BackwardPass
    Qx::Vector{Vector{Float64}}
    Qu::Vector{Vector{Float64}}
    Qxx::Vector{Matrix{Float64}}
    Qux::Vector{Matrix{Float64}}
    Quu::Vector{Matrix{Float64}}

    Qux_reg::Vector{Matrix{Float64}}
    Quu_reg::Vector{Matrix{Float64}}

    function BackwardPassZOH(n::Int,m::Int,N::Int)
        Qx = [zeros(n) for i = 1:N-1]
        Qu = [zeros(m) for i = 1:N-1]
        Qxx = [zeros(n,n) for i = 1:N-1]
        Qux = [zeros(m,n) for i = 1:N-1]
        Quu = [zeros(m,m) for i = 1:N-1]

        Qux_reg = [zeros(m,n) for i = 1:N-1]
        Quu_reg = [zeros(m,m) for i = 1:N-1]

        new(Qx,Qu,Qxx,Qux,Quu,Qux_reg,Quu_reg)
    end
end

"""
$(SIGNATURES)
Solve the dynamic programming problem, starting from the terminal time step
Computes the gain matrices K and d by applying the principle of optimality at
each time step, solving for the gradient (s) and Hessian (S) of the cost-to-go
function. Also returns parameters Δv for line search (see Synthesis and Stabilization of Complex Behaviors through
Online Trajectory Optimization)
"""
function backwardpass!(results::SolverVectorResults,solver::Solver,bp::BackwardPass)
    # if solver.control_integration == :foh
    #     Δv = _backwardpass_foh_speedup!(results,solver, bp)
    # elseif solver.opts.square_root
    #     Δv = _backwardpass_sqrt!(results, solver, bp) #TODO option to help avoid ill-conditioning [see algorithm xx]
    # else
        Δv = _backwardpass_speedup!(results, solver, bp)
    # end

    return Δv
end

function _backwardpass!(res::SolverVectorResults,solver::Solver{Obj}) where Obj <: Union{UnconstrainedObjective,ConstrainedObjective}
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    costfun = solver.obj.cost
    # Q = solver.obj.Q; R = getR(solver); Qf = solver.obj.Qf; xf = solver.obj.xf; c = solver.obj.c;

    dt = solver.dt
    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s

    # Boundary Conditions
    S[N], s[N] = taylor_expansion(costfun,X[N])
    # S[N] = Qf
    # s[N] = Qf*(X[N] - xf)

    # Initialize expected change in cost-to-go
    Δv = [0.0 0.0]

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ
        CxN = res.Cx_N
        S[N] += CxN'*res.IμN*CxN
        s[N] += CxN'*res.IμN*res.CN + CxN'*res.λN
    end


    # Backward pass
    k = N-1
    while k >= 1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        expansion = taylor_expansion(costfun,X[k],U[k])
        lxx,luu,lxu,lx,lu = expansion .* dt
        # lx = dt*Q*vec(X[k] - xf)
        # lu = dt*R*vec(U[k])
        # lxx = dt*Q
        # luu = dt*R

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = lx + fdx'*vec(s[k+1])
        Qu = lu + fdu'*vec(s[k+1])
        Qxx = lxx + fdx'*S[k+1]*fdx

        Quu = luu + fdu'*S[k+1]*fdu
        Qux = fdu'*S[k+1]*fdx

        # Constraints
        if res isa ConstrainedIterResults
            Cx, Cu = res.Cx[k], res.Cu[k]
            Qx += Cx'*Iμ[k]*C[k] + Cx'*λ[k]
            Qu += Cu'*Iμ[k]*C[k] + Cu'*λ[k]
            Qxx += Cx'*Iμ[k]*Cx
            Quu += Cu'*Iμ[k]*Cu
            Qux += Cu'*Iμ[k]*Cx

            if solver.opts.minimum_time
                h = U[k][m̄]
                Qu[m̄] += 2*h*stage_cost(X[k],U[k],Q,R,xf,c)
                Qux[m̄,1:n] += vec(2h*(X[k]-xf)'Q)
                tmp = zero(Quu)
                tmp[:,m̄] = R*U[k]
                Quu += 2h*(tmp+tmp')
                Quu[m̄,m̄] += 2*stage_cost(X[k],U[k],Q,R,xf,c)

                if k > 1
                    Qu[m̄] += - C[k-1][end]*Iμ[k-1][end,end] - λ[k-1][end]
                    Quu[m̄,m̄] += Iμ[k-1][end,end]
                end

            end

        end

        if solver.opts.regularization_type == :state
            Quu_reg = Quu + res.ρ[1]*fdu'*fdu
            Qux_reg = Qux + res.ρ[1]*fdu'*fdx
        elseif solver.opts.regularization_type == :control
            Quu_reg = Quu + res.ρ[1]*I
            Qux_reg = Qux
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays

            # increase regularization
            regularization_update!(res,solver,:increase)

            # reset backward pass
            k = N-1
            Δv = [0.0 0.0]
            continue
        end

        # Compute gains
        K[k] = -Quu_reg::Matrix{Float64}\Qux_reg::Matrix{Float64}
        d[k] = -Quu_reg\Qu::Vector{Float64}

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = vec(Qx) + K[k]'*Quu*vec(d[k]) + K[k]'*vec(Qu) + Qux'*vec(d[k])
        S[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # calculated change is cost-to-go over entire trajectory
        Δv += [vec(d[k])'*vec(Qu) 0.5*vec(d[k])'*Quu*vec(d[k])]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

function _backwardpass_speedup!(res::SolverVectorResults,solver::Solver,bp)
    # Get problem sizes
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    # Objective
    costfun = solver.obj.cost

    # Minimum time and infeasible options
    solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s
    # L = res.L; l = res.l
    L = zeros(n+mm,n+mm); l = zeros(n+mm)

    Qx = bp.Qx; Qu = bp.Qu; Qxx = bp.Qxx; Quu = bp.Quu; Qux = bp.Qux
    Quu_reg = bp.Quu_reg; Qux_reg = bp.Qux_reg

    # TEMP resets values for now - this will get fixed
    for k = 1:N-1
        Qx[k] = zeros(n); Qu[k] = zeros(mm); Qxx[k] = zeros(n,n); Quu[k] = zeros(mm,mm); Qux[k] = zeros(mm,n)
        Quu_reg[k] = zeros(mm,mm); Qux_reg[k] = zeros(mm,n)
    end

    # Useful linear indices
    idx = Array(1:2*(n+mm))
    Idx = reshape(Array(1:(n+mm)^2),n+mm,n+mm)

    x_idx = idx[1:n]
    u_idx = idx[n+1:n+m]
    ū_idx = idx[n+1:n+mm]

    # Boundary Conditions
    S[N], s[N] = taylor_expansion(costfun, X[N])

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ
        Cx = res.Cx; Cu = res.Cu
        CN = res.CN; CxN = res.Cx_N; IμN = res.IμN; λN = res.λN

        S[N] += CxN'*IμN*CxN
        s[N] += CxN'*(IμN*CN + λN)
    end

    # Backward pass
    k = N-1
    while k >= 1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        x = X[k]
        u = U[k][1:m]
        #
        # ℓ1 = ℓ(x,u,W,R,xf)
        # ℓx = W*(x - xf)
        # ℓu = R*u
        # ℓxx = W
        # ℓuu = R
        # ℓux = zeros(m,n)
        #
        # # Assemble expansion
        # Lx = dt*ℓx
        # Lu = dt*ℓu
        # Lxx = dt*ℓxx
        # Luu = dt*ℓuu
        # Lux = dt*ℓux

        expansion = taylor_expansion(costfun,x,u)
        Qxx[k],Quu[k][1:m,1:m],Qux[k][1:m,1:n],Qx[k],Qu[k][1:m] = expansion .* dt

        # Minimum time expansion components
        if solver.opts.minimum_time
            ℓ1 = stage_cost(costfun,x,u)
            h = U[k][m̄]

            Qu[k][m̄] = 2*h*(ℓ1 + R_minimum_time)

            tmp = 2*h*ℓu
            Quu[k][1:m,m̄] = tmp
            Quu[k][m̄,1:m] = tmp'
            Quu[k][m̄,m̄] = 2*(ℓ1 + R_minimum_time)

            Qux[k][m̄,1:n] = 2*h*expansion[4]'
        end

        # Infeasible expansion components
        if solver.opts.infeasible
            Qu[k][m̄+1:mm] = R_infeasible*U[k][m̄+1:m̄+n]
            Quu[k][m̄+1:mm,m̄+1:mm] = R_infeasible
        end

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx[k] += fdx'*s[k+1]
        Qu[k] += fdu'*s[k+1]
        Qxx[k] += fdx'*S[k+1]*fdx
        Quu[k] += fdu'*S[k+1]*fdu
        Qux[k] += fdu'*S[k+1]*fdx

        # Constraints
        if res isa ConstrainedIterResults

            Qx[k] += Cx[k]'*(Iμ[k]*C[k] + λ[k])
            Qu[k] += Cu[k]'*(Iμ[k]*C[k] + λ[k])
            Qxx[k] += Cx[k]'*Iμ[k]*Cx[k]
            Quu[k] += Cu[k]'*Iμ[k]*Cu[k]
            Qux[k] += Cu[k]'*Iμ[k]*Cx[k]
        end

        if solver.opts.regularization_type == :state
            Quu_reg[k] = Quu[k] + res.ρ[1]*fdu'*fdu
            Qux_reg[k] = Qux[k] + res.ρ[1]*fdu'*fdx
        elseif solver.opts.regularization_type == :control
            Quu_reg[k] = Quu[k] + res.ρ[1]*I
            Qux_reg[k] = Qux[k]
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg[k])))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            regularization_update!(res,solver,:increase)

            # reset backward pass
            k = N-1
            Δv[1] = 0.
            Δv[2] = 0.
            continue
        end

        # Compute gains
        K[k] = -Quu_reg[k]\Qux_reg[k]
        d[k] = -Quu_reg[k]\Qu[k]

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = Qx[k] + K[k]'*Quu[k]*d[k] + K[k]'*Qu[k] + Qux[k]'*d[k]
        S[k] = Qxx[k] + K[k]'*Quu[k]*K[k] + K[k]'*Qux[k] + Qux[k]'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # calculated change is cost-to-go over entire trajectory
        Δv[1] += d[k]'*Qu[k]
        Δv[2] += 0.5*d[k]'*Quu[k]*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

function _backwardpass_alt!(res::SolverVectorResults,solver::Solver,bp::BackwardPass)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf

    solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d;
    S = res.S; s = res.s

    Qx = bp.Qx; Qu = bp.Qu; Qxx = bp.Qxx; Quu = bp.Quu; Qux = bp.Qux
    Quu_reg = bp.Quu_reg; Qux_reg = bp.Qux_reg

    for k = 1:N-1
        Qx[k] = zeros(n); Qu[k] = zeros(mm); Qxx[k] = zeros(n,n); Quu[k] = zeros(mm,mm); Qux[k] = zeros(mm,n)
        Quu_reg[k] = zeros(mm,mm); Qux_reg[k] = zeros(mm,n)
    end

    # Boundary Conditions
    S[N] = Wf
    s[N] = Wf*(X[N] - xf)

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ; Cx = res.Cx; Cu = res.Cu
        CN = res.CN; CxN = res.Cx_N; IμN = res.IμN; λN = res.λN

        S[N] += CxN'*IμN*CxN
        s[N] += CxN'*(IμN*CN + λN)
    end

    # Backward pass
    k = N-1
    while k >= 1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        x = X[k]
        u = U[k][1:m]

        ℓ1 = ℓ(x,u,W,R,xf)
        ℓx = W*(x - xf)
        ℓu = R*u
        ℓxx = W
        ℓuu = R
        ℓux = zeros(m,n)

        # Assemble expansion

        Qx[k] = dt*ℓx
        Qu[k][1:m] = dt*ℓu
        Qxx[k] = dt*ℓxx
        Quu[k][1:m,1:m] = dt*ℓuu
        Qux[k][1:m,1:n] = dt*ℓux

        # Minimum time expansion components
        if solver.opts.minimum_time
            h = U[k][m̄]

            Qu[k][m̄] = 2*h*(ℓ1 + R_minimum_time)

            tmp = 2*h*ℓu
            Quu[k][1:m,m̄] = tmp
            Quu[k][m̄,1:m] = tmp'
            Quu[k][m̄,m̄] = 2*(ℓ1 + R_minimum_time)

            Qux[k][m̄,1:n] = 2*h*ℓx'
        end

        # Infeasible expansion components
        if solver.opts.infeasible
            Qu[k][m̄+1:mm] = R_infeasible*U[k][m̄+1:m̄+n]
            Quu[k][m̄+1:mm,m̄+1:mm] = R_infeasible
        end

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx[k] += fdx'*s[k+1]
        Qu[k] += fdu'*s[k+1]
        Qxx[k] += fdx'*S[k+1]*fdx
        Quu[k] += fdu'*S[k+1]*fdu
        Qux[k] += fdu'*S[k+1]*fdx

        # Constraints
        if res isa ConstrainedIterResults
            Qx[k] += Cx[k]'*(Iμ[k]*C[k] + λ[k])
            Qu[k] += Cu[k]'*(Iμ[k]*C[k] + λ[k])
            Qxx[k] += Cx[k]'*Iμ[k]*Cx[k]
            Quu[k] += Cu[k]'*Iμ[k]*Cu[k]
            Qux[k] += Cu[k]'*Iμ[k]*Cx[k]
        end

        if solver.opts.regularization_type == :state
            Quu_reg[k] = Quu[k] + res.ρ[1]*fdu'*fdu
            Qux_reg[k] = Qux[k] + res.ρ[1]*fdu'*fdx
        elseif solver.opts.regularization_type == :control
            Quu_reg[k] = Quu[k] + res.ρ[1]*I
            Qux_reg[k] = Qux[k]
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg[k])))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            regularization_update!(res,solver,:increase)

            # reset backward pass
            k = N-1
            Δv[1] = 0.
            Δv[2] = 0.
            continue
        end

        # Compute gains
        K[k] = -Quu_reg[k]\Qux_reg[k]
        d[k] = -Quu_reg[k]\Qu[k]

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = Qx[k] + K[k]'*Quu[k]*d[k] + K[k]'*Qu[k] + Qux[k]'*d[k]
        S[k] = Qxx[k] + K[k]'*Quu[k]*K[k] + K[k]'*Qux[k] + Qux[k]'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # calculated change is cost-to-go over entire trajectory
        Δv[1] += d[k]'*Qu[k]
        Δv[2] += 0.5*d[k]'*Quu[k]*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

"""
$(SIGNATURES)
Perform a backwards pass with Cholesky Factorizations of the Cost-to-Go to
avoid ill-conditioning.
"""
function _backwardpass_sqrt!(res::SolverVectorResults,solver::Solver,bp)
    N = solver.N
    n = solver.model.n
    m = solver.model.m

    if solver.model.m != length(res.U[1])
        m += n
    end

    # Q = solver.obj.Q
    # R = solver.obj.R
    # xf = solver.obj.xf
    # Qf = solver.obj.Qf
    costfun = solver.obj.cost
    dt = solver.dt

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; Su = res.S; s = res.s

    # Terminal Cost-to-go
    lxx,lx = taylor_expansion(costfun, X[N])
    if isa(solver.obj, ConstrainedObjective)
        Cx = res.Cx_N
        Su[N] = cholesky(lxx + Cx'*res.IμN*Cx).U
        s[N] = lxx + Cx'*res.IμN*res.CN + Cx'*res.λN
    else
        Su[N] = cholesky(lxx).U
        s[N] = lx
    end

    # Initialization of expected change in cost-to-go
    Δv = [0. 0.]

    k = N-1

    # Backward pass
    while k >= 1
        expansion = taylor_expansion(costfun,X[k],U[k])
        lxx,luu,lxu,lx,lu = expansion .* dt
        # lx = dt*Q*(X[k] - xf)
        # lu = dt*R*(U[k])
        # lxx = dt*Q
        # luu = dt*R

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
function forwardpass!(res::SolverIterResults, solver::Solver, Δv::Array)#, J_prev::Float64)
    # Pull out values from results
    X = res.X; U = res.U; X_ = res.X_; U_ = res.U_

    update_constraints!(res,solver,X,U)
    J_prev = cost(solver, res, X, U)

    J = Inf
    alpha = 1.0
    iter = 0
    z = -1.
    expected = 0.

    logger = current_logger()
    # print_header(logger,InnerIters) #TODO: fix, this errored out
    @logmsg InnerIters :iter value=0
    @logmsg InnerIters :cost value=J_prev
    # print_row(logger,InnerIters) #TODO: fix, same issue
    while (z ≤ solver.opts.z_min || z > solver.opts.z_max) && J >= J_prev

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            # set trajectories to original trajectory
            X_ .= deepcopy(X)
            U_ .= deepcopy(U)

            update_constraints!(res,solver,X_,U_)
            J = cost(solver, res, X_, U_)

            z = 0.
            alpha = 0.0
            expected = 0.

            @logmsg InnerLoop "Max iterations (forward pass)"
            regularization_update!(res,solver,:increase) # increase regularization
            res.ρ[1] += solver.opts.ρ_forwardpass
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
        # update_constraints!(res,solver,X_,U_)
        J = cost(solver, res, X_, U_)   # Unconstrained cost

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
        @logmsg InnerLoop :c_max value=max_violation(res)
    end
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ value=J_prev-J
    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*alpha
    @logmsg InnerLoop :ρ value=res.ρ[1]

    if J > J_prev
        error("Error: Cost increased during Forward Pass")
    end
    return J
end
