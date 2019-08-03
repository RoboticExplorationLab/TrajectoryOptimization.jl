
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
        cost_expansion!(E[k], prob.obj[k], X[k], U[k], Dt[k])
    end
    cost_expansion!(E[N], prob.obj[N], X[N])
    nothing
end

function backwardpass!(prob::Problem, solver::StaticiLQRSolver)
    n,m,N = size(prob)

    # Objective
    obj = prob.obj

    X = prob.X; U = prob.U; K = solver.K; d = solver.d

    S = solver.S
    Q = solver.Q

    # Terminal cost-to-go
    S[N].xx = copy(Q[N].xx)
    S[N].x = copy(Q[N].x)

    # Initialize expected change in cost-to-go
    ΔV = zeros(2)

    # ix = @SVector [i for i = 1:n]
    # iu = @SVector [i for i = 1:m]
    ix, iu = 1:n, n .+ (1:m)

    # Backward pass
    k = N-1
    while k >= 1
        fdx, fdu = solver.∇F[k][ix,ix], solver.∇F[k][ix,iu]

        Q[k].x += fdx'*S[k+1].x
        Q[k].u += fdu'*S[k+1].x
        Q[k].xx += fdx'*S[k+1].xx*fdx
        Q[k].uu += fdu'*S[k+1].xx*fdu
        Q[k].ux += fdu'*S[k+1].xx*fdx

        if solver.opts.bp_reg_type == :state
            # Quu_reg = cholesky(Q[k].uu + solver.ρ[1]*fdu'*fdu,check=false)
            Quu_reg = Q[k].uu + solver.ρ[1]*fdu'*fdu
            Qux_reg = Q[k].ux + solver.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            # Quu_reg = cholesky(Q[k].uu + solver.ρ[1]*I,check=false)
            Quu_reg = Q[k].uu + solver.ρ[1]*Diagonal(ones(prob.model.m))
            Qux_reg = Q[k].ux
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
        d[k] = -1.0*(Quu_reg\Q[k].u)

        # Calculate cost-to-go (using unregularized Quu and Qux)
        S[k].x  =  Q[k].x + K[k]'*Q[k].uu*d[k] + K[k]'* Q[k].u + Q[k].ux'*d[k]
        S[k].xx = Q[k].xx + K[k]'*Q[k].uu*K[k] + K[k]'*Q[k].ux + Q[k].ux'*K[k]
        S[k].xx = 0.5*(S[k].xx + S[k].xx')

        # calculated change is cost-to-go over entire trajectory
        ΔV[1] += d[k]'*Q[k].u
        ΔV[2] += 0.5*d[k]'*Q[k].uu*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    # regularization_update!(solver,:decrease)

    return ΔV
end
