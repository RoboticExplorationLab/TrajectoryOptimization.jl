"""
$(SIGNATURES)
Solve the dynamic programming problem, starting from the terminal time step
Computes the gain matrices K and d by applying the principle of optimality at
each time step, solving for the gradient (s) and Hessian (S) of the cost-to-go
function. Also returns parameters Δv for line search (see Synthesis and Stabilization of Complex Behaviors through
Online Trajectory Optimization)
"""
function backwardpass!(results::SolverVectorResults,solver::Solver)
    if solver.opts.square_root
        Δv = _backwardpass_sqrt!(results, solver)
    else
        Δv = _backwardpass!(results, solver)
    end
    return Δv
end

function _backwardpass!(res::SolverVectorResults,solver::Solver)
    # Get problem sizes
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    # Objective
    costfun = solver.obj.cost

    # Minimum time and infeasible options
    solver.state.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.state.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s

    Qx = res.bp.Qx; Qu = res.bp.Qu; Qxx = res.bp.Qxx; Quu = res.bp.Quu; Qux = res.bp.Qux
    Quu_reg = res.bp.Quu_reg; Qux_reg = res.bp.Qux_reg

    # TEMP resets values for now - this will get fixed
    for k = 1:N-1
        Qx[k] = zeros(nn); Qu[k] = zeros(mm); Qxx[k] = zeros(nn,nn); Quu[k] = zeros(mm,mm); Qux[k] = zeros(mm,nn)
        Quu_reg[k] = zeros(mm,mm); Qux_reg[k] = zeros(mm,nn)
    end

    # Boundary Conditions
    S[N][1:n,1:n], s[N][1:n] = taylor_expansion(costfun, X[N][1:n])

    if solver.state.minimum_time
        s[N][n̄] = R_minimum_time*X[N][n̄]
        S[N][n̄,n̄] = R_minimum_time
    end

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ
        Cx = res.Cx; Cu = res.Cu

        S[N] += Cx[N]'*Iμ[N]*Cx[N]
        s[N] += Cx[N]'*(Iμ[N]*C[N] + λ[N])
    end

    # Backward pass
    k = N-1
    while k >= 1
        solver.state.minimum_time ? dt = U[k][m̄]^2 : dt = solver.dt

        x = X[k][1:n]
        u = U[k][1:m]

        expansion = taylor_expansion(costfun,x,u)
        Qxx[k][1:n,1:n],Quu[k][1:m,1:m],Qux[k][1:m,1:n],Qx[k][1:n],Qu[k][1:m] = expansion .* dt

        # Minimum time expansion components
        if solver.state.minimum_time
            ℓ1 = stage_cost(costfun,x,u)
            h = U[k][m̄]
            tmp = 2*h*expansion[5]

            Qu[k][m̄] = h*(2*ℓ1 + R_minimum_time)
            Quu[k][1:m,m̄] = tmp
            Quu[k][m̄,1:m] = tmp'
            Quu[k][m̄,m̄] = (2*ℓ1 + R_minimum_time)
            Qux[k][m̄,1:n] = 2*h*expansion[4]'

            Qx[k][n̄] = R_minimum_time*X[k][n̄]
            Qxx[k][n̄,n̄] = R_minimum_time
        end

        # Infeasible expansion components
        if solver.state.infeasible
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

        if solver.opts.bp_reg_type == :state
            Quu_reg[k] = Quu[k] + res.ρ[1]*fdu'*fdu
            Qux_reg[k] = Qux[k] + res.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            Quu_reg[k] = Quu[k] + res.ρ[1]*I
            Qux_reg[k] = Qux[k]
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg[k])))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            @logmsg InnerIters "Regularizing Quu "
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

function _backwardpass_new!(res::SolverVectorResults,solver::Solver)
    # Get problem sizes
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    # Objective
    costfun = solver.obj.cost

    # Minimum time and infeasible options
    solver.state.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.state.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s

    Qx = res.bp.Qx; Qu = res.bp.Qu; Qxx = res.bp.Qxx; Quu = res.bp.Quu; Qux = res.bp.Qux
    Quu_reg = res.bp.Quu_reg; Qux_reg = res.bp.Qux_reg

    # TEMP resets values for now - this will get fixed
    for k = 1:N-1
        Qx[k] = zeros(nn); Qu[k] = zeros(mm); Qxx[k] = zeros(nn,nn); Quu[k] = zeros(mm,mm); Qux[k] = zeros(mm,nn)
        Quu_reg[k] = zeros(mm,mm); Qux_reg[k] = zeros(mm,nn)
    end

    # Boundary Conditions
    al_expansion!(res,solver,N)
    S[N][1:n,1:n], s[N][1:n] = Qxx[N][1:n,1:n], Qx[N][1:n]

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Backward pass
    k = N-1
    while k >= 1
        solver.state.minimum_time ? dt = U[k][m̄]^2 : nothing

        x = X[k][1:n]
        u = U[k][1:m]

        al_expansion!(res,solver,k)

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx[k] += fdx'*s[k+1]
        Qu[k] += fdu'*s[k+1]
        Qxx[k] += fdx'*S[k+1]*fdx
        Quu[k] += fdu'*S[k+1]*fdu
        Qux[k] += fdu'*S[k+1]*fdx

        if solver.opts.bp_reg_type == :state
            Quu_reg[k] = Quu[k] + res.ρ[1]*fdu'*fdu
            Qux_reg[k] = Qux[k] + res.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            Quu_reg[k] = Quu[k] + res.ρ[1]*I
            Qux_reg[k] = Qux[k]
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg[k])))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            @logmsg InnerIters "Regularizing Quu "
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

function al_expansion!(res::SolverIterResults, solver::Solver, k::Int)
    X = res.X; U = res.U
    costfun = solver.obj.cost
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)
    R_infeasible = solver.opts.R_infeasible

    Qx = res.bp.Qx; Qu = res.bp.Qu; Qxx = res.bp.Qxx; Quu = res.bp.Quu; Qux = res.bp.Qux

    x = X[k][1:n]

    if k == N
        Qxx[k][1:n,1:n],Qx[k][1:n] = taylor_expansion(costfun,x)
    else
        u = U[k][1:m]
        expansion = taylor_expansion(costfun,x,u)
        Qxx[k][1:n,1:n],Quu[k][1:m,1:m],Qux[k][1:m,1:n],Qx[k][1:n],Qu[k][1:m] = expansion .* solver.dt
    end


    if solver.state.constrained
        C = res.C; Iμ = res.Iμ; λ = res.λ; μ = res.μ
        Cx = res.Cx; Cu = res.Cu

        if solver.opts.al_type == :default
            Qx[k] += Cx[k]'*(Iμ[k]*C[k] + λ[k])
            Qxx[k] += Cx[k]'*Iμ[k]*Cx[k]
            if k < solver.N
                Qu[k] += Cu[k]'*(Iμ[k]*C[k] + λ[k])
                Quu[k] += Cu[k]'*Iμ[k]*Cu[k]
                Qux[k] += Cu[k]'*Iμ[k]*Cx[k]
            end
        elseif solver.opts.al_type == :algencan
            a = res.active_set[k]
            Ia = Diagonal(a)
            Iμk = Diagonal(μ[k])

            # Declare some temp variables that are re-used
            gg = Ia*Iμk
            g = Ia*(λ[k] + Iμk*C[k])

            # Expansion
            Qx[k] += Cx[k]'g
            Qxx[k] += Cx[k]'gg*Cx[k]
            if k < solver.N
                Qu[k] += Cu[k]'g
                Quu[k] += Cu[k]'gg*Cu[k]
                Qux[k] += Cu[k]'gg*Cx[k]
            end

            # Qx[k] += Cx[k]'*(Iμ[k]*C[k] + λ[k])
            # Qxx[k] += Cx[k]'*Iμ[k]*Cx[k]
            # if k < solver.N
            #     Qu[k] += Cu[k]'*(Iμ[k]*C[k] + λ[k])
            #     Quu[k] += Cu[k]'*Iμ[k]*Cu[k]
            #     Qux[k] += Cu[k]'*Iμ[k]*Cx[k]
            # end
        end

        # Minimum time expansion components
        if solver.state.minimum_time
            R_minimum_time = solver.opts.R_minimum_time
            Qx[k][n̄] = R_minimum_time*X[k][n̄]
            Qxx[k][n̄,n̄] = R_minimum_time
            if k < N
                ℓ1 = stage_cost(costfun,x,u)
                h = U[k][m̄]
                tmp = 2*h*expansion[5]

                Qu[k][m̄] = h*(2*ℓ1 + R_minimum_time)
                Quu[k][1:m,m̄] = tmp
                Quu[k][m̄,1:m] = tmp'
                Quu[k][m̄,m̄] = (2*ℓ1 + R_minimum_time)
                Qux[k][m̄,1:n] = 2*h*expansion[4]'
            end
        end

        # Infeasible expansion components
        if solver.state.infeasible && k < N
            Qu[k][m̄+1:mm] = R_infeasible*U[k][m̄+1:m̄+n]
            Quu[k][m̄+1:mm,m̄+1:mm] = R_infeasible*Diagonal(I,n)
        end
    end
end

"""
$(SIGNATURES)
Perform a backwards pass with Cholesky Factorizations of the Cost-to-Go to
avoid ill-conditioning.
"""
function _backwardpass_sqrt!(res::SolverVectorResults,solver::Solver)
    # Get problem sizes
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    # Objective
    costfun = solver.obj.cost

    # Minimum time and infeasible options
    solver.state.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.state.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; Su = res.S; s = res.s

    Qx = res.bp.Qx; Qu = res.bp.Qu; Qxx = res.bp.Qxx; Quu = res.bp.Quu; Qux = res.bp.Qux
    Quu_reg = res.bp.Quu_reg; Qux_reg = res.bp.Qux_reg

    # TEMP resets values for now - this will get fixed
    for k = 1:N-1
        Qx[k] = zeros(nn); Qu[k] = zeros(mm); Qxx[k] = zeros(nn,nn); Quu[k] = zeros(mm,mm); Qux[k] = zeros(mm,nn)
        Quu_reg[k] = zeros(mm,mm); Qux_reg[k] = zeros(mm,nn)
    end

    # Boundary Conditions
    Su[N][1:n,1:n], s[N][1:n] = taylor_expansion(costfun, X[N][1:n])

    if solver.state.minimum_time
        s[N][n̄] = R_minimum_time*X[N][n̄]
        Su[N][n̄,n̄] = R_minimum_time
    end

    # Take square root (via cholesky)
    try
        Su[N][1:nn,1:nn] = cholesky(Su[N][1:nn,1:nn]).U # if no terminal cost is provided cholesky will fail gracefully
    catch PosDefException
        if sum([Su[N][i,i] for i = 1:n]) != 0. # TODO there may be something faster here, but lu fails for this case
            # tmp = svd(Su[N])
            # Su[N] = Diagonal(sqrt.(tmp.S))*tmp.V'
            tmp = eigen(Su[N])
            Su[N] = Diagonal(sqrt.(tmp.values))*tmp.vectors'
        elseif tr(Su[N][1:n,1:n]) == 0. && n̄ > n
            Su[N][n̄,n̄] = sqrt(Su[N][n̄,n̄])
        end
    end

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ
        Cx = res.Cx; Cu = res.Cu
        Iμ_sqrt = sqrt.(Iμ[N])

        Su[N][1:nn,1:nn] = chol_plus(Su[N],Iμ_sqrt*Cx[N])
        s[N] += Cx[N]'*(Iμ[N]*C[N] + λ[N])
    end

    # Backward pass
    Δv = zeros(2)
    tmp1 = []
    tmp2 = []

    k = N-1
    while k >= 1
        solver.state.minimum_time ? dt = U[k][m̄]^2 : dt = solver.dt

        x = X[k][1:n]
        u = U[k][1:m]

        expansion = taylor_expansion(costfun,x,u)
        Qxx[k][1:n,1:n],Quu[k][1:m,1:m],Qux[k][1:m,1:n],Qx[k][1:n],Qu[k][1:m] = expansion .* dt

        # Minimum time expansion components
        if solver.state.minimum_time
            ℓ1 = stage_cost(costfun,x,u)
            h = U[k][m̄]
            tmp = 2*h*expansion[5]

            Qu[k][m̄] = h*(2*ℓ1 + R_minimum_time)
            Quu[k][1:m,m̄] = tmp
            Quu[k][m̄,1:m] = tmp'
            Quu[k][m̄,m̄] = (2*ℓ1 + R_minimum_time)
            Qux[k][m̄,1:n] = 2*h*expansion[4]'

            Qx[k][n̄] = R_minimum_time*X[k][n̄]
            Qxx[k][n̄,n̄] = R_minimum_time
        end

        # Infeasible expansion components
        if solver.state.infeasible
            Qu[k][m̄+1:mm] = R_infeasible*U[k][m̄+1:m̄+n]
            Quu[k][m̄+1:mm,m̄+1:mm] = R_infeasible
        end

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx[k] += fdx'*s[k+1]
        Qu[k] += fdu'*s[k+1]
        Qux[k] += (fdu'*Su[k+1]')*(Su[k+1]*fdx)
        try
            Qxx[k][1:nn,1:nn] = cholesky(Qxx[k][1:nn,1:nn]).U
        catch
            if sum([Qxx[k][i,i] for i = 1:n]) != 0. #TODO faster
                tmp = eigen(Qxx[k])
                Qxx[k] = Diagonal(sqrt.(tmp.values))*tmp.vectors'
            elseif tr(Qxx[k][1:n,1:n]) == 0. && n̄ > n
                Qxx[k][n̄,n̄] = sqrt(Qxx[k][n̄,n̄])
            end
        end
        try
            Quu[k] = cholesky(Quu[k]).U
        catch #TODO fix this...
            error("problem with sqrt bp Quu...")
            # tmp = svd(Quu[k])
            # Quu[k] = Diagonal(sqrt.(tmp.S))*tmp.V'

            # tmp = eigen(Quu[k] + fdu'*Su[k+1]'*Su[k+1]*fdu)
            # Wuu = Diagonal(sqrt.(tmp.values))*tmp.vectors'
        end
        Wxx = chol_plus(Qxx[k], Su[k+1]*fdx)
        Wuu = chol_plus(Quu[k], Su[k+1]*fdu)



        # Constraints
        if res isa ConstrainedIterResults
            Iμ_sqrt = sqrt.(Iμ[k])

            Qx[k] += Cx[k]'*(Iμ[k]*C[k] + λ[k])
            Qu[k] += Cu[k]'*(Iμ[k]*C[k] + λ[k])
            Wxx = chol_plus(Wxx,Iμ_sqrt*Cx[k])
            Wuu = chol_plus(Wuu,Iμ_sqrt*Cu[k])
            Qux[k] += Cu[k]'*Iμ[k]*Cx[k]
        end

        if solver.opts.bp_reg_type == :state
            Wuu_reg = chol_plus(Wuu,sqrt(res.ρ[1])*fdu)
            Qux_reg[k] = Qux[k] + res.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            Wuu_reg = chol_plus(Wuu,sqrt(res.ρ[1])*Matrix(I,mm,mm))
            Qux_reg[k] = Qux[k]
        end

        #TODO find better PD check for Wuu_reg
        # # Regularization
        # if !isposdef(Hermitian(Array(Wuu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
        #     # increase regularization
        #     regularization_update!(res,solver,:increase)
        #
        #     # reset backward pass
        #     k = N-1
        #     Δv[1] = 0.
        #     Δv[2] = 0.
        #     continue
        # end

        # Compute gains
        K[k] = -Wuu_reg\(Wuu_reg'\Qux_reg[k])
        d[k] = -Wuu_reg\(Wuu_reg'\Qu[k])

        # Calculate cost-to-go
        s[k] = Qx[k] + (K[k]'*Wuu')*(Wuu*d[k]) + K[k]'*Qu[k] + Qux[k]'*d[k]

        try
            tmp1 = (Wxx')\Qux[k]'
        catch SingularException
            # if solver.opts.bp_sqrt_inv_type == :reg
            #     reg = solver.opts.bp_reg_sqrt_initial
            #     while minimum(eigvals(Wxx)) < 1e-6
            #         Wxx += reg*Matrix(I,nn,nn)
            #         try
            #             tmp1 = (Wxx')\Qux[k]'
            #             break
            #         catch
            #             reg *= solver.opts.bp_reg_sqrt_increase_factor
            #             if reg >= solver.opts.bp_reg_max
            #                 error("Square root regularization exceded")
            #             end
            #         end
            #     end
            # elseif solver.opts.bp_sqrt_inv_type == :pseudo
            tmp1 = pinv(Array(Wxx'))*Qux[k]'
            # end
        end

        try
            tmp2 = cholesky(Wuu'*Wuu - tmp1'*tmp1).U
        catch
            tmp = eigen(Wuu'*Wuu - tmp1'*tmp1)
            tmp2 = Diagonal(sqrt.(tmp.values))*tmp.vectors'
            # tmp = svd(Wuu'*Wuu - tmp1'*tmp1)
            # tmp2 = Diagonal(sqrt.(tmp.S))*tmp.V'
        end

        Su[k][1:nn,1:nn] = Wxx + tmp1*K[k]
        Su[k][nn+1:nn+mm,1:nn] = tmp2*K[k]

        # calculated change is cost-to-go over entire trajectory
        Δv[1] += d[k]'*Qu[k]
        Δv[2] += 0.5*d[k]'*Wuu'*Wuu*d[k]

        Quu_reg[k] = Array(Wuu_reg)

        Quu[k] = Array(Wuu)
        Qxx[k] = Array(Wxx)

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

function chol_plus(A,B)
    n1,m = size(A)
    n2 = size(B,1)
    P = zeros(n1+n2,m)
    P[1:n1,:] = A
    P[n1+1:end,:] = B
    return qr(P).R
end

function backwardpass_max_condition_number(bp::TrajectoryOptimization.BackwardPass)
    N = length(bp.Quu)
    max_cn = 0.
    for k = 1:N-1
        cn = cond(bp.Quu_reg[k])
        if cn > max_cn
            max_cn = cn
        end
    end
    return max_cn
end

function backwardpass_max_condition_number(results::TrajectoryOptimization.SolverVectorResults)
    N = length(results.S)
    max_cn = 0.
    for k = 1:N
        cn = cond(results.S[k])
        if cn > max_cn && cn < Inf
            max_cn = cn
        end
    end
    return max_cn
end
