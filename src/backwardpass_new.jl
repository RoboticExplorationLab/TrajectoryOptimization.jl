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

    # Objective
    cost = prob.cost

    dt = prob.dt

    X = prob.X; U = prob.U; K = solver.K; d = solver.d; S = solver.S; s = solver.s

    reset!(solver.bp)
    Qx = solver.bp.Qx; Qu = solver.bp.Qu; Qxx = solver.bp.Qxx; Quu = solver.bp.Quu; Qux = solver.bp.Qux
    Quu_reg = solver.bp.Quu_reg; Qux_reg = solver.bp.Qux_reg

    # Boundary Conditions
    S[N], s[N] = cost_expansion(prob.cost, X[N])

    # Initialize expected change in cost-to-go
    ΔV = zeros(2)

    # Backward pass
    k = N-1
    while k >= 1
        Qxx[k],Quu[k],Qux[k],Qx[k],Qu[k] = cost_expansion(prob.cost,X[k],U[k],k)

        fdx, fdu = solver.∇F[k].xx, solver.∇F[k].xu

        Qx[k] += fdx'*s[k+1]
        Qu[k] += fdu'*s[k+1]
        Qxx[k] += fdx'*S[k+1]*fdx
        Quu[k] += fdu'*S[k+1]*fdu
        Qux[k] += fdu'*S[k+1]*fdx

        if solver.opts.bp_reg_type == :state
            Quu_reg[k] = Quu[k] + solver.ρ[1]*fdu'*fdu
            Qux_reg[k] = Qux[k] + solver.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            Quu_reg[k] = Quu[k] + solver.ρ[1]*I
            Qux_reg[k] = Qux[k]
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg[k])))  # need to wrap Array since isposdef doesn't work for static arrays
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
        K[k] = -Quu_reg[k]\Qux_reg[k]
        d[k] = -Quu_reg[k]\Qu[k]

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = Qx[k] + K[k]'*Quu[k]*d[k] + K[k]'*Qu[k] + Qux[k]'*d[k]
        S[k] = Qxx[k] + K[k]'*Quu[k]*K[k] + K[k]'*Qux[k] + Qux[k]'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # calculated change is cost-to-go over entire trajectory
        ΔV[1] += d[k]'*Qu[k]
        ΔV[2] += 0.5*d[k]'*Quu[k]*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(solver,:decrease)

    return ΔV
end
