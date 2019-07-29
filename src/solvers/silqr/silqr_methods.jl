
# Generic solve methods
"iLQR solve method"
function solve!(prob::Problem{T,Discrete}, solver::iLQRSolver{T}) where T<:AbstractFloat
    reset!(solver)
    to = solver.stats[:timer]

    n,m,N = size(prob)
    J = Inf

    logger = default_logger(solver)

    # Initial rollout
    rollout!(prob)
    live_plotting(prob,solver)

    J_prev = cost(prob)


    with_logger(logger) do
        record_iteration!(prob, solver, J_prev, Inf)
        for i = 1:solver.opts.iterations
            J = step!(prob, solver, J_prev)

            # check for cost blow up
            if J > solver.opts.max_cost_value
                @warn "Cost exceeded maximum cost"
                return solver
            end

            @timeit to "copy" begin
                copyto!(prob.X, solver.X̄)
                copyto!(prob.U, solver.Ū)
            end

            dJ = abs(J - J_prev)
            J_prev = copy(J)
            record_iteration!(prob, solver, J, dJ)
            live_plotting(prob,solver)

            println(logger, InnerLoop)
            @timeit to "convergence" evaluate_convergence(solver) ? break : nothing
        end
    end
    return solver
end

function step!(prob::Problem{T}, solver::iLQRSolver{T}, J::T) where T
    to = solver.stats[:timer]
    @timeit to "jacobians" jacobian!(prob,solver)   # TODO: rename this to dynamics jacobian
    @timeit to "cost expansion" cost_expansion!(prob,solver)
    @timeit to "backward pass" ΔV = backwardpass!(prob,solver)
    @timeit to "forward pass" forwardpass!(prob,solver,ΔV,J)
end

function cost_expansion!(prob::Problem{T}, solver::StaticiLQRSolver{T}) where T
    E = solver.Q
    N = prob.N
    X,U,Dt = prob.X, prob.U, get_dt_traj(prob)
    for k in eachindex(U)
        E.xx[k], E.uu[k], E.ux[k], E.x[k], E.u[k] =
            cost_expansion(prob.obj[k], X[k], U[k], Dt[k])
    end
    E.xx[N], E.x[N] =
        cost_expansion(prob.obj[N], X[N])
    nothing
end

function backwardpass!(prob::Problem, solver::StaticiLQRSolver)
    N = prob.N

    # Objective
    obj = prob.obj

    X = prob.X; U = prob.U; K = solver.K; d = solver.d

    S = solver.S
    Q = solver.Q

    # Terminal cost-to-go
    S.xx[N] = Q.xx[N]
    S.x[N] = Q.x[N]

    # Initialize expected change in cost-to-go
    ΔV = zeros(2)

    ix = @SVector [i for i = 1:n]
    iu = @SVector [i for i = 1:m]

    # Backward pass
    k = N-1
    while k >= 1
        fdx, fdu = solver.∇F[k][ix,ix], solver.∇F[k][ix,iu]

        Q.x[k] += fdx'*S.x[k+1]
        Q.u[k] += fdu'*S.x[k+1]
        Q.xx[k] += fdx'*S.xx[k+1]*fdx
        Q.uu[k] += fdu'*S.xx[k+1]*fdu
        Q.ux[k] += fdu'*S.xx[k+1]*fdx

        if solver.opts.bp_reg_type == :state
            # Quu_reg = cholesky(Q[k].uu + solver.ρ[1]*fdu'*fdu,check=false)
            Quu_reg = Q[k].uu + solver.ρ[1]*fdu'*fdu
            Qux_reg = Q[k].ux + solver.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            # Quu_reg = cholesky(Q[k].uu + solver.ρ[1]*I,check=false)
            Quu_reg = Q.uu[k] + solver.ρ[1]*Diagonal(ones(prob.model.m))
            Qux_reg = Q.ux[k]
        end



        # Regularization
        # if Quu_reg.info == -1
        if !isposdef(Hermitian(Array(Quu_reg)))
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
        K[k] = -1.0*(Quu_reg\Qux_reg)
        d[k] = -1.0*(Quu_reg\Q.u[k])

        # Calculate cost-to-go (using unregularized Quu and Qux)
        S.x[k]  = Q.x[k] + K[k]'*Q.uu[k]*d[k] + K[k]'*Q.u[k] + Q.ux[k]'*d[k]
        S.xx[k] = Q.xx[k] + K[k]'*Q.uu[k]*K[k] + K[k]'*Q.ux[k] + Q.ux[k]'*K[k]
        S.xx[k] = 0.5*(S.xx[k] + S.xx[k]')

        # calculated change is cost-to-go over entire trajectory
        ΔV[1] += d[k]'*Q.u[k]
        ΔV[2] += 0.5*d[k]'*Q.uu[k]*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    # regularization_update!(solver,:decrease)

    return ΔV
end
