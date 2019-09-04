function backwardpass!(prob::Problem,solver::iLQRSolver)
    if solver.opts.square_root
        return _backwardpass_sqrt!(prob,solver)
    else
        return _backwardpass!(prob,solver)
    end
end

function _backwardpass!(prob::Problem,solver::iLQRSolver)
    N = prob.N

    # Objective
    obj = prob.obj
    X = prob.X; U = prob.U; K = solver.K; d = solver.d

    if prob.model isa Vector{Model}
        m = prob.model[1].m
        G = state_diff_jacobian(prob.model[N-1], X[N])
    else
        m = prob.model.m
        G = state_diff_jacobian(prob.model, X[N])
    end


    S = solver.S
    Q = solver.Q

    # Terminal cost-to-go
    S[N].xx .= G*Q[N].xx*G'
    S[N].x .= G*Q[N].x

    # Initialize expected change in cost-to-go
    ΔV = zeros(2)

    # Backward pass
    k = N-1
    while k >= 1
        if prob.model isa Vector{Model}
            fdx, fdu = dynamics_expansion(solver.∇F[k], prob.model[k], X[k], U[k])
            Qxx, Quu, Qux, Qx, Qu = cost_expansion(solver.Q[k], prob.model[k], X[k], U[k])
        else
            fdx, fdu = dynamics_expansion(solver.∇F[k], prob.model, X[k], U[k])
            Qxx, Quu, Qux, Qx, Qu = cost_expansion(solver.Q[k], prob.model, X[k], U[k])
        end
        Qx .+= fdx'*S[k+1].x
        Qu .+= fdu'*S[k+1].x
        Qxx .+= fdx'*S[k+1].xx*fdx
        Quu .+= fdu'*S[k+1].xx*fdu
        Qux .+= fdu'*S[k+1].xx*fdx

        # fdx, fdu = solver.∇F[k].xx, solver.∇F[k].xu
        # Qxx,Quu,Qux,Qx,Qu = Q[k].xx, Q[k].uu, Q[k].ux, Q[k].x, Q[k].u
        # Q[k].x .+= fdx'*S[k+1].x
        # Q[k].u .+= fdu'*S[k+1].x
        # Q[k].xx .+= fdx'*S[k+1].xx*fdx
        # Q[k].uu .+= fdu'*S[k+1].xx*fdu
        # Q[k].ux .+= fdu'*S[k+1].xx*fdx

        if solver.opts.bp_reg_type == :state
            # Quu_reg = cholesky(Q[k].uu + solver.ρ[1]*fdu'*fdu,check=false)
            Quu_reg = Quu + solver.ρ[1]*fdu'*fdu
            Qux_reg = Qux + solver.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            # Quu_reg = cholesky(Q[k].uu + solver.ρ[1]*I,check=false)
            Quu_reg = Quu + solver.ρ[1]*Diagonal(ones(prob.model.m))
            Qux_reg = Qux
        end



        # Regularization
        # if Quu_reg.info == -1
        if !isposdef(Hermitian(Quu_reg))
            # increase regularization
            println("Regularizing Quu")

            @logmsg InnerIters "Regularizing Quu "
            regularization_update!(solver,:increase)

            # reset backward pass
            k = N-1
            ΔV[1] = 0.
            ΔV[2] = 0.
            continue
        end

        # Compute gains
        K[k] = -1.0*(Quu_reg\Qux_reg)
        d[k] = -1.0*(Quu_reg\Q[k].u)

        # Calculate cost-to-go (using unregularized Quu and Qux)
        S[k].x .=   Qx + K[k]'*Quu*d[k] + K[k]'*Qu + Qux'*d[k]
        S[k].xx .= Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'*K[k]
        S[k].xx .= 0.5*(S[k].xx + S[k].xx')

        # calculated change is cost-to-go over entire trajectory
        ΔV[1] += d[k]'*Qu
        ΔV[2] += 0.5*d[k]'*Quu*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(solver,:decrease)

    return ΔV
end

function _backwardpass_sqrt!(prob::Problem,solver::iLQRSolver)
    N = prob.N

    # Objective
    obj = prob.obj

    X = prob.X; U = prob.U; K = solver.K; d = solver.d

    S = solver.S
    Q = solver.Q # cost-to-go expansion

    # Terminal cost-to-go
    try
        S[N].xx .= cholesky(Q[N].xx).U
    catch PosDefException
        error("Terminal cost Hessian must be PD for sqrt backward pass")
    end

    S[N].x .= Q[N].x

    # Initialize expected change in cost-to-go
    ΔV = zeros(2)

    tmp1 = []; tmp2 = []

    # Backward pass
    k = N-1
    while k >= 1
        fdx, fdu = solver.∇F[k].xx, solver.∇F[k].xu

        Q[k].x .+= fdx'*S[k+1].x
        Q[k].u .+= fdu'*S[k+1].x
        tmp_x = S[k+1].xx*fdx
        tmp_u = S[k+1].xx*fdu
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
        S[k].x .= Q[k].x + (K[k]'*Q[k].uu')*(Q[k].uu*d[k]) + K[k]'*Q[k].u + Q[k].ux'*d[k]

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

        S[k].xx .= chol_plus(Q[k].xx + tmp1*K[k],tmp2*K[k])

        # calculated change is cost-to-go over entire trajectory
        ΔV[1] += d[k]'*Q[k].u

        tmp = Q[k].uu*d[k]
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

function chol_plus!(A::AbstractMatrix{T},B::AbstractMatrix{T}) where T
    A .= qr([A;B]).R
end
