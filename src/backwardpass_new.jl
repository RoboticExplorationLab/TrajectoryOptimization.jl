function backwardpass!(prob::Problem,solver::iLQRSolver)
    if solver.opts.square_root
        error("Square root bp not implemented yet!")
        ΔV = _backwardpass_sqrt!(prob,solver)
    else
        ΔV = _backwardpass!(prob,solver)
    end
    return ΔV
end

function _backwardpass!(prob::Problem,solver::iLQRSolver)
    N = prob.N
    dt = prob.dt

    # Objective
    cost = prob.cost

    X = prob.X; U = prob.U; K = solver.K; d = solver.d; S = solver.S; s = solver.s

    Q = solver.Q # cost-to-go expansion
    reset!(Q)

    # Boundary Conditions
    cost_expansion!(solver,cost,X[N])

    # Initialize expected change in cost-to-go
    ΔV = zeros(2)

    # Backward pass
    k = N-1
    while k >= 1
        cost_expansion!(solver, cost, X[k], U[k], dt, k)

        fdx, fdu = dynamics_jacobians(prob,solver,k)

        Q[k].x .+= fdx'*s[k+1]
        Q[k].u .+= fdu'*s[k+1]
        Q[k].xx .+= fdx'*S[k+1]*fdx
        Q[k].uu .+= fdu'*S[k+1]*fdu
        Q[k].ux .+= fdu'*S[k+1]*fdx


        if solver.opts.bp_reg_type == :state
            Quu_reg = Q[k].uu + solver.ρ[1]*fdu'*fdu
            Qux_reg = Q[k].ux + solver.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            Quu_reg = Q[k].uu + solver.ρ[1]*I
            Qux_reg = Q[k].ux
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            @logmsg InnerIters "Regularizing Quu "
            regularization_update!(solver,:increase)

            # reset backward pass
            k = N-1
            ΔV[1] = 0.
            ΔV[2] = 0.
            continue
        end

        # Compute gains
        K[k] = -Quu_reg\Qux_reg
        d[k] = -Quu_reg\Q[k].u

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = Q[k].x + K[k]'*Q[k].uu*d[k] + K[k]'*Q[k].u + Q[k].ux'*d[k]
        S[k] = Q[k].xx + K[k]'*Q[k].uu*K[k] + K[k]'*Q[k].ux + Q[k].ux'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # calculated change is cost-to-go over entire trajectory
        ΔV[1] += d[k]'*Q[k].u
        ΔV[2] += 0.5*d[k]'*Q[k].uu*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(solver,:decrease)

    return ΔV
end

function dynamics_jacobians(prob::Problem{T},solver::AbstractSolver,k::Int) where T
    #TODO quaternion
    return solver.∇F[k].xx, solver.∇F[k].xu
end

function _backwardpass_sqrt!(prob::Problem,solver::iLQRSolver)
    N = prob.N
    dt = prob.dt

    # Objective
    cost = prob.cost

    X = prob.X; U = prob.U; K = solver.K; d = solver.d; S = solver.S; s = solver.s

    Q = solver.Q # cost-to-go expansion
    reset!(Q)

    # Boundary Conditions
    cost_expansion!(solver,cost,X[N])
    try
        S[N] = cholesky(S[N]).U
    catch PosDefException
        error("Terminal cost Hessian must be PD for sqrt backward pass")
    end

    # Initialize expected change in cost-to-go
    ΔV = zeros(2)

    tmp1 = []; tmp2 = []

    # Backward pass
    k = N-1
    while k >= 1
        cost_expansion!(solver, cost, X[k], U[k], dt, k)

        fdx, fdu = dynamics_jacobians(prob,solver,k)

        Q[k].x .+= fdx'*s[k+1]
        Q[k].u .+= fdu'*s[k+1]
        tmp_x = S[k+1]*fdx
        tmp_u = S[k+1]*fdu
        Q[k].xx .= cholesky(Q[k].xx).U
        chol_plus!(Q[k].xx,tmp_x)
        Q[k].uu .= cholesky(Q[k].uu).U
        chol_plus!(Q[k].uu,tmp_u)
        Q[k].ux .+= tmp_u'*tmp_x

        if solver.opts.bp_reg_type == :state
            Quu_reg = chol_plus(Q[k].uu, sqrt(solver.ρ[1])*fdu)
            Qux_reg = Q[k].ux + solver.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            Quu_reg = chol_plus(Q[k].uu, sqrt(solver.ρ[1])*Diagonal(ones(prob.model.m)))
            Qux_reg = Q[k].ux
        end

        #TODO regularization scheme

        # Compute gains
        K[k] = -Quu_reg\(Quu_reg'\Qux_reg)
        d[k] = -Quu_reg\(Quu_reg'\Q[k].u)

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = Q[k].x + (K[k]'*Q[k].uu')*(Q[k].uu*d[k]) + K[k]'*Q[k].u + Q[k].ux'*d[k]

        try
            tmp1 = (Q[k].xx')\Q[k].ux'
        catch SingularException
            tmp1 = pinv(Array(Q[k].xx'))*Q[k].ux'
        end

        Quu = Q[k].uu'*Q[k].uu
        _tmp = tmp1'*tmp1
        try
            tmp2 = cholesky(Quu - _tmp).U
        catch
            tmp = eigen(Quu - _tmp)
            tmp2 = Diagonal(sqrt.(tmp.values))*tmp.vectors'
        end

        S[k] = chol_plus(Q[k].xx + tmp1*K[k],tmp2*K[k])

        # calculated change is cost-to-go over entire trajectory
        ΔV[1] += d[k]'*Q[k].u

        tmp = (Q[k].uu*d[k])
        ΔV[2] += 0.5*tmp'*tmp
        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(solver,:decrease)

    return ΔV
end

#TODO neither of these are particularly fast, preallocating for the inplace version wasn't much faster either
function chol_plus(A::AbstractMatrix{T},B::AbstractMatrix{T}) where T
    n1,m = size(A)
    n2 = size(B,1)
    P = zeros(n1+n2,m)
    P[1:n1,:] = A
    P[n1+1:end,:] = B
    return qr(P).R
end

function chol_plus!(X::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T}) where T
    X .= qr([A;B]).R
    return nothing
end
