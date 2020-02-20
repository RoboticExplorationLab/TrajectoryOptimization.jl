function initialize!(solver::iLQRSolver2)
    set_verbosity!(solver.opts)
    clear_cache!(solver.opts)

    solver.ρ[1] = solver.opts.bp_reg_initial
    solver.dρ[1] = 0.0

    # Initial rollout
    rollout!(solver)
    cost!(solver.obj, solver.Z)
end

"""
Calculates the optimal feedback gains K,d as well as the 2nd Order approximation of the
Cost-to-Go, using a backward Riccati-style recursion. (non-allocating)
"""
function backwardpass!(solver::iLQRSolver2{T,QUAD,L,O,n,n̄,m}) where {T,QUAD<:QuadratureRule,L,O,n,n̄,m}
	N = solver.N

    # Objective
    obj = solver.obj
    model = solver.model

    # Extract variables
    Z = solver.Z; K = solver.K; d = solver.d;
    G = solver.G
    S = solver.S
    Q = solver.E
	Quu_reg = solver.Quu_reg
	Qux_reg = solver.Qux_reg

    # Terminal cost-to-go
    S[N].xx .= solver.Q[N].xx
    S[N].x .= solver.Q[N].x

    # Initialize expecte change in cost-to-go
    ΔV = @SVector zeros(2)

    k = N-1
    while k > 0
        ix = Z[k]._x
        iu = Z[k]._u

		# Get error state expanions
		fdx,fdu = solver.D[k].A, solver.D[k].B
		error_expansion!(Q, solver.Q[k], model, Z[k], solver.G[k])

		# inputs: Q, S, fdx, fdu, Quu_reg, Qux_reg, K, d, ρ

		# Compute the cost-to-go, stashing temporary variables in S[k]
        # Qx =  Q.x[k] + fdx'S.x[k+1]
		mul!(Q.x, Transpose(fdx), S[k+1].x, 1.0, 1.0)

        # Qu =  Q.u[k] + fdu'S.x[k+1]
		mul!(Q.u, Transpose(fdu), S[k+1].x, 1.0, 1.0)

        # Qxx = Q.xx[k] + fdx'S.xx[k+1]*fdx
		mul!(S[k].xx, Transpose(fdx), S[k+1].xx)
		mul!(Q.xx, S[k].xx, fdx, 1.0, 1.0)

        # Quu = Q.uu[k] + fdu'S.xx[k+1]*fdu
		mul!(S[k].ux, Transpose(fdu), S[k+1].xx)
		mul!(Q.uu, S[k].ux, fdu, 1.0, 1.0)

        # Qux = Q.ux[k] + fdu'S.xx[k+1]*fdx
		mul!(S[k].ux, Transpose(fdu), S[k+1].xx)
		mul!(Q.ux, S[k].ux, fdx, 1.0, 1.0)

		# Regularization
        if solver.opts.bp_reg_type == :state
            Quu_reg .= Q.uu #+ solver.ρ[1]*fdu'fdu
			mul!(Quu_reg, Transpose(fdu), fdu, solver.ρ[1], 1.0)
            Qux_reg .= Q.ux #+ solver.ρ[1]*fdu'fdx
			mul!(Qux_reg, fdu', fdx, solver.ρ[1], 1.0)
        elseif solver.opts.bp_reg_type == :control
            Quu_reg .= Q.uu #+ solver.ρ[1]*I
			Quu_reg .+= solver.ρ[1]*Diagonal(@SVector ones(m))
            Qux_reg .= Q.ux
        end

        if solver.opts.bp_reg
            vals = eigvals(Hermitian(Quu_reg))
            if minimum(vals) <= 0
                @warn "Backward pass regularized"
                regularization_update!(solver, :increase)
                k = N-1
                ΔV = @SVector zeros(2)
                continue
            end
        end

        # Compute gains
		LAPACK.potrf!('U',Quu_reg.data)
		K[k] .= Qux_reg
		d[k] .= Q.u
		LAPACK.potrs!('U', Quu_reg.data, K[k].data)
		LAPACK.potrs!('U', Quu_reg.data, d[k].data)
		K[k] .*= -1
		d[k] .*= -1
		# Quu = cholesky!(Quu_reg)
		# ldiv!(K[k], Quu, -Qux_reg)
		# ldiv!(d[k], Quu, -Q.u)
        # K[k] = -(Quu_reg\Qux_reg)
        # d[k] = -(Quu_reg\Qu)

        # Calculate cost-to-go (using unregularized Quu and Qux)
		# S.x[k]  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
		tmp1 = S[k].u
		S[k].x .= Q.x
		mul!(tmp1, Q.uu, d[k])
		mul!(S[k].x, Transpose(K[k]), tmp1, 1.0, 1.0)
		mul!(S[k].x, Transpose(K[k]), Q.u, 1.0, 1.0)
		mul!(S[k].x, Transpose(Q.ux), d[k], 1.0, 1.0)

		# S.xx[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
		tmp2 = S[k].ux
		S[k].xx .= Q.xx
		mul!(tmp2, Q.uu, K[k])
		mul!(S[k].xx, Transpose(K[k]), tmp2, 1.0, 1.0)
		mul!(S[k].xx, Transpose(K[k]), Q.ux, 1.0, 1.0)
		mul!(S[k].xx, Transpose(Q.ux), K[k], 1.0, 1.0)
		transpose!(Q.xx, S[k].xx)
		S[k].xx .+= Q.xx
		S[k].xx .*= 0.5

        # calculated change is cost-to-go over entire trajectory
		t1 = d[k]'Q.u
		mul!(Q.u, Q.uu, d[k])
		t2 = 0.5*d[k]'Q.u
        ΔV += @SVector [t1, t2]

        k -= 1
    end

    regularization_update!(solver, :decrease)

    return ΔV

end

function _calc_Q!(Q, S1, S, fdx, fdu)
	# Compute the cost-to-go, stashing temporary variables in S[k]
    # Qx =  Q.x[k] + fdx'S.x[k+1]
	mul!(Q.x, fdx', S1.x, 1.0, 1.0)

    # Qu =  Q.u[k] + fdu'S.x[k+1]
	mul!(Q.u, fdu', S1.x, 1.0, 1.0)

    # Qxx = Q.xx[k] + fdx'S.xx[k+1]*fdx
	mul!(S.xx, fdx', S1.xx)
	mul!(Q.xx, S.xx, fdx, 1.0, 1.0)

    # Quu = Q.uu[k] + fdu'S.xx[k+1]*fdu
	mul!(S.ux, fdu', S1.xx)
	mul!(Q.uu, S.ux, fdu, 1.0, 1.0)

    # Qux = Q.ux[k] + fdu'S.xx[k+1]*fdx
	mul!(S.ux, fdu', S1.xx)
	mul!(Q.ux, S.ux, fdx, 1.0, 1.0)
	return nothing
end

function _calc_gains!(K, d, Quu, Qux)
	LAPACK.potrf!('U',Quu)
	K[k] .= Qux
	d[k] .= Q.u
	LAPACK.potrs!('U', Quu, K[k])
	LAPACK.potrs!('U', Quu, d[k])
	K[k] .*= -1
	d[k] .*= -1
end

function _calc_ctg!(S, Q, K, d)
    # Calculate cost-to-go (using unregularized Quu and Qux)
	# S.x[k]  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
	tmp1 = S[k].u
	S[k].x .= Q.x
	mul!(tmp1, Q.uu, d[k])
	mul!(S[k].x, K[k]', tmp1, 1.0, 1.0)
	mul!(S[k].x, K[k]', Q.u, 1.0, 1.0)
	mul!(S[k].x, Q.ux', d[k], 1.0, 1.0)

	# S.xx[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
	tmp2 = S[k].ux
	S[k].xx .= Q.xx
	mul!(tmp2, Q.uu, K[k])
	mul!(S[k].xx, K[k]', tmp2, 1.0, 1.0)
	mul!(S[k].xx, K[k]', Q.ux, 1.0, 1.0)
	mul!(S[k].xx, Q.ux', K[k], 1.0, 1.0)
	Q.xx .= (S[k].xx .+ S[k].xx')
	S[k].xx .= Q.xx
	S[k].xx .*= 0.5

    # calculated change is cost-to-go over entire trajectory
	t1 = d[k]'Q.u
	mul!(Q.u, Q.uu, d[k])
	t2 = 0.5*d[k]'Q.u
    return @SVector [t1, t2]
end

function rollout!(solver::iLQRSolver2{T,Q,n}, α) where {T,Q,n}
    Z = solver.Z; Z̄ = solver.Z̄
    K = solver.K; d = solver.d;

    Z̄[1].z = [solver.x0; control(Z[1])]

    temp = 0.0
	δx = solver.E.x
	δu = solver.E.u

    for k = 1:solver.N-1
        δx .= state_diff(solver.model, state(Z̄[k]), state(Z[k]))
		δu .= d[k]
		mul!(δu, K[k], δx, 1.0, α)
        ū = control(Z[k]) + δu
        set_control!(Z̄[k], ū)

        # Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]
        Z̄[k+1].z = [discrete_dynamics(Q, solver.model, Z̄[k]);
            control(Z[k+1])]

        temp = norm(Z̄[k+1].z)
        if temp > solver.opts.max_state_value
            return false
        end
    end
    return true
end

function gradient_todorov!(solver::iLQRSolver2)
	tmp = solver.E.u
    for k in eachindex(solver.d)
		tmp .= abs.(solver.d[k])
		u = abs.(control(solver.Z[k])) .+ 1
		tmp ./= u
		solver.grad[k] = maximum(tmp)
    end
end

function step!(solver::iLQRSolver2, J)
    Z = solver.Z
    state_diff_jacobian!(solver.G, solver.model, Z)
    # discrete_jacobian!(solver.∇F, solver.model, Z)
	dynamics_expansion!(solver.D, solver.G, solver.model, solver.Z)
    cost_expansion!(solver.Q, solver.G, solver.obj, solver.model, solver.Z)
    ΔV = backwardpass!(solver)
    forwardpass!(solver, ΔV, J)
end
