using Test

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
    if solver.opts.square_root
        Δv = _backwardpass_sqrt!(results, solver, bp)
    else
        Δv = _backwardpass!(results, solver, bp)
    end
    return Δv
end

function _backwardpass!(res::SolverVectorResults,solver::Solver,bp::BackwardPass)
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

    Qx = bp.Qx; Qu = bp.Qu; Qxx = bp.Qxx; Quu = bp.Quu; Qux = bp.Qux
    Quu_reg = bp.Quu_reg; Qux_reg = bp.Qux_reg

    # TEMP resets values for now - this will get fixed
    for k = 1:N-1
        Qx[k] = zeros(nn); Qu[k] = zeros(mm); Qxx[k] = zeros(nn,nn); Quu[k] = zeros(mm,mm); Qux[k] = zeros(mm,nn)
        Quu_reg[k] = zeros(mm,mm); Qux_reg[k] = zeros(mm,nn)
    end

    # Boundary Conditions
    S[N][1:n,1:n], s[N][1:n] = taylor_expansion(costfun, X[N][1:n])

    if solver.state.minimum_time
        s[N][n̄] = 0.5*R_minimum_time*X[N][n̄]
        S[N][n̄,n̄] = 0.5*R_minimum_time
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
        solver.state.minimum_time ? dt = U[k][m̄]^2 : nothing

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
            @logmsg InnerIters "Fixing Quu with regularization"
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
function _backwardpass_sqrt!(res::SolverVectorResults,solver::Solver,bp::BackwardPass)
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

    for k = 1:N
        res.S[k] = zeros(nn+mm,nn)
    end
    Su = res.S

    Qx = bp.Qx; Qu = bp.Qu; Qxx = bp.Qxx; Quu = bp.Quu; Qux = bp.Qux
    Quu_reg = bp.Quu_reg; Qux_reg = bp.Qux_reg

    # TEMP resets values for now - this will get fixed
    for k = 1:N-1
        Qx[k] = zeros(nn); Qu[k] = zeros(mm); Qxx[k] = zeros(nn,nn); Quu[k] = zeros(mm,mm); Qux[k] = zeros(mm,nn)
        Quu_reg[k] = zeros(mm,mm); Qux_reg[k] = zeros(mm,nn)
    end

    # Boundary Conditions
    Su[N][1:n,1:n], s[N][1:n] = taylor_expansion(costfun, X[N][1:n])

    if solver.state.minimum_time
        s[N][n̄] = 0.5*R_minimum_time*X[N][n̄]
        Su[N][n̄,n̄] = 0.5*R_minimum_time
    end

    # Take square root (via cholesky)
    try
        Su[N][1:nn,1:nn] = cholesky(Su[N][1:nn,1:nn]).U # if no terminal cost is provided cholesky will fail gracefully
    catch PosDefException
        if tr(Su[N][1:nn,1:nn]) != 0.
            error("Square root bp not currently implemented with positive semi-definite terminal cost Hessian")
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
            if tr(Qxx[k][1:n,1:n]) != 0.
                error("Square root bp not currently implemented with positive semi-definite stage cost Hessian for states")
            else
                Qxx[k][n̄,n̄] = sqrt(Qxx[k][n̄,n̄])
            end
        end
        try
            Quu[k] = cholesky(Quu[k]).U
        catch
            error("Control Cost Hessian is not Positive Definite")
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

        tmp1 = []
        try
            tmp1 = (Wxx')\Qux[k]'
        catch SingularException
            reg = solver.opts.bp_reg_min
            iter = 0
            while minimum(eigvals(Wxx)) < 1e-6 && iter < 15
                Wxx += reg*Matrix(I,nn,nn)
                try
                    tmp1 = (Wxx')\Qux[k]'
                    break
                catch
                    reg *= 10
                    iter += 1
                    if iter > 15
                        error("broken :<")
                    end
                end
            end
        end

        tmp2 = []
        try
            tmp2 = cholesky(Wuu'*Wuu - tmp1'*tmp1).U
        catch
            tmp2 = chol_minus(Wuu,tmp1)
        end

        Su[k][1:nn,1:nn] = Wxx + tmp1*K[k]
        Su[k][nn+1:nn+mm,1:nn] = tmp2*K[k]

        # calculated change is cost-to-go over entire trajectory
        Δv[1] += d[k]'*Qu[k]
        Δv[2] += 0.5*d[k]'*Wuu'*Wuu*d[k]

        bp.Quu_reg[k] = Array(Wuu_reg)
        bp.Quu[k] = Array(Wuu)

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

function chol_minus(A,B::Matrix)
    AmB = Cholesky(A,:U,0)
    for i = 1:size(B,1)
        lowrankdowndate!(AmB,B[i,:])
    end
    U = AmB.U
end
