"""
$(SIGNATURES)
    Lagrange multiplier updates
        -see Bertsekas 'Constrained Optimization' chapter 2 (p.135)
        -see Toussaint 'A Novel Augmented Lagrangian Approach for Inequalities and Convergent Any-Time Non-Central Updates'
"""
function λ_update!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    for k = 1:N-1
        results.λ[k] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, results.λ[k] + results.Iμ[k]*results.C[k]))
        results.λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])
    end

    results.λ[N] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, results.λ[N] + results.Iμ[N]*results.C[N]))
    results.λ[N][1:pI_N] = max.(0.0,results.λ[N][1:pI_N])
end

"""
$(SIGNATURES)
    Second order dual update - Batch Dual QP solve
    -UNDER DEVELOPMENT
"""
function λ_second_order_update!(results::SolverIterResults,solver::Solver,verbose::Bool=false)
    n = solver.model.n
    m = solver.model.m
    nm = n+m
    N = solver.N
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_constraints(solver)

    Nz = n*N + m*(N-1) # number of decision variables x, u
    Nu = m*(N-1) # number of control decision variables u

    Np = p*(N-1) + p_N # number of constraints

    Mz = N + N-1 # number of decision vectors
    Mu = N-1 # number of control decision vectors

    B̄ = zeros(Nz,Nu)
    Ā = zeros(Nz,n)

    Q̄ = zeros(Nz,Nz)
    q̄ = zeros(Nz)

    C̄ = zeros(Np,Nz)
    c̄ = zeros(Np)

    λ_tmp = zeros(Np)
    idx_inequality = zeros(Bool,Np)
    active_set = zeros(Bool,Np)

    x0 = solver.obj.x0
    costfun = solver.obj.cost

    for k = 1:N
        x = results.X[k]

        if k != N
            u = results.U[k]
            Q,R,H,q,r = taylor_expansion(costfun,x,u)
        else
            Qf,qf = taylor_expansion(costfun,x)
        end

        # Indices
        idx = ((k-1)*nm + 1):k*nm
        idx2 = ((k-1)*p + 1):k*p
        idx3 = ((k-1)*nm + 1):((k-1)*nm + n)
        idx4 = ((k-1)*nm + n + 1):k*nm
        idx5 = ((k-1)*m + 1):k*m
        idx6 = ((k-1)*p + 1):((k-1)*p + pI)

        # Calculate Ā
        k == 1 ? Ā[idx3,1:n] = 1.0*Matrix(I,n,n) : Ā[idx3,1:n] = prod(results.fdx[1:k-1])

        # Calculate B̄
        if k > 1
            for j = 1:k-1
                idx7 = ((j-1)*m + 1):j*m
                j == k-1 ? B̄[idx3,idx7] = results.fdu[j][1:n,1:m] : B̄[idx3,idx7] = prod(results.fdx[j+1:(k-1)])*results.fdu[j]
            end
        end

        # Calculate Q̄, q̄, C̄, c̄
        if k != N
            Q̄[idx,idx] = [Q H';H R]
            q̄[idx] = [q;r]

            G = results.Cx[k]
            H = results.Cu[k]
            C̄[idx2,idx] = [G H]
            c̄[idx2] = results.C[k]

            λ_tmp[idx2] = results.λ[k]
            idx_inequality[idx6] .= true
            active_set[idx2] = results.active_set[k]

            B̄[idx4,idx5] = 1.0*Matrix(I,m,m)
        else
            idx = ((k-1)*nm + 1):Nz
            Q̄[idx,idx] = Qf
            q̄[idx] = qf

            idx2 = ((k-1)*p + 1):Np
            C̄[idx2,idx] = results.Cx[N]
            c̄[idx2] = results.C[N]

            idx6 = ((k-1)*p + 1):((k-1)*p + pI_N)

            λ_tmp[idx2] = results.λ[N]
            idx_inequality[idx6] .= true
            active_set[idx2] .= results.active_set[N]
        end

    end
    N_active_set = sum(active_set)

    # if verbose
    #     # Tests
    #     @test Ā[1:n,:] == Matrix(I,n,n)
    #     k = 3
    #     @test Ā[(k-1)*nm+1:(k-1)*nm+n,:] == prod(results.fdx[1:2])
    #     @test results.fdx[1]*results.fdx[2] == prod(results.fdx[1:2])
    #
    #     k = 7
    #     @test Ā[(k-1)*nm+1:(k-1)*nm+n,:] == prod(results.fdx[1:k-1])
    #     @test Ā[(N-1)*nm+1:Nz,:] == prod(results.fdx[1:N-1])
    #
    #     @test B̄[1:n,1:n] == zeros(n,n)
    #     @test B̄[n+1:nm,1:m] == 1.0*Matrix(I,m,m)
    #     @test B̄[nm+1:nm+n,1:m] == results.fdu[1][1:n,1:m]
    #     @test B̄[nm+n+1:2*nm,m+1:2*m] == 1.0*Matrix(I,m,m)
    #     @test B̄[(N-1)*nm+1:Nz,1:m] == prod(results.fdx[2:N-1])*results.fdu[1]
    #     @test B̄[(N-1)*nm+1:Nz,m+1:2*m] == prod(results.fdx[3:N-1])*results.fdu[2]
    #     @test B̄[(N-1)*nm+1:Nz,(N-2)*m+1:(N-1)*m] == results.fdu[N-1]
    #
    #     k = 1
    #     Q,R,H,q,r = taylor_expansion(costfun,results.X[k],results.U[k])
    #     @test Q̄[1:nm,1:nm] == [Q H'; H R]
    #     @test q̄[1:nm] == [q;r]
    #
    #     k = 13
    #     Q,R,H,q,r = taylor_expansion(costfun,results.X[k],results.U[k])
    #     @test Q̄[(k-1)*nm+1:k*nm,(k-1)*nm+1:k*nm] == [Q H'; H R]
    #     @test q̄[(k-1)*nm+1:k*nm] == [q;r]
    #
    #     k = N
    #     Qf,qf = taylor_expansion(costfun,results.X[k])
    #     @test Q̄[(k-1)*nm+1:Nz,(k-1)*nm+1:Nz] == Qf
    #     @test q̄[(k-1)*nm+1:Nz] == qf
    #
    #     @test C̄[1:p,1:nm] == [results.Cx[1] results.Cu[1]]
    #
    #     k = 9
    #     @test C̄[(k-1)*p+1:k*p,(k-1)*nm+1:k*nm] == [results.Cx[k] results.Cu[k]]
    #     @test C̄[(N-1)*p+1:Np,(N-1)*nm+1:Nz] == results.Cx[N]
    #
    #     @test c̄[1:p] == results.C[1]
    #
    #     k = 17
    #     @test c̄[(k-1)*p+1:k*p] == results.C[k]
    #     @test c̄[(N-1)*p+1:Np] == results.C[N]
    #
    #     @test all(idx_inequality[1:pI] .== true)
    #     @test all(idx_inequality[pI+1:p] .== false)
    #     k = 12
    #
    #     @test all(idx_inequality[(k-1)*p+1:(k-1)*p+pI] .== true)
    #     idx_inequality[(k-1)*p+1:(k-1)*p+pI]
    #     @test all(idx_inequality[(N-1)*p+1:Np] .== true)
    # end
    # ū = -(B̄'*Q̄*B̄)\(B̄'*(q̄ + Q̄*Ā*x0) + B̄'*C̄'*λ_tmp)

    P = B̄'*Q̄*B̄
    M = q̄ + Q̄*Ā*x0

    # L = 0.5*ū'*P*ū + M'*B̄*ū + λ_tmp'*(C̄*B̄*ū + C̄*Ā*x0 + c̄)

    # D = 0.5*M'*B̄*inv(P')*B̄'*M + 0.5*M'*B̄*inv(P')*B̄'*C̄'*λ_tmp + 0.5*λ_tmp'*C̄*B̄*inv(P')*B̄'*M + 0.5*λ_tmp'*C̄*B̄*inv(P')*B̄'*C̄'*λ_tmp - M'*B̄*inv(P)*B̄'*M - M'*B̄*inv(P)*B̄'*C̄'*λ_tmp - λ_tmp'*C̄*B̄*inv(P)*B̄'*M - λ_tmp'*C̄*B̄*inv(P)*B̄'*C̄'*λ_tmp + λ_tmp'*C̄*Ā*x0 + λ_tmp'*c̄
    Q_dual = C̄*B̄*inv(P')*B̄'*C̄' - 2*C̄*B̄*inv(P)*B̄'*C̄'
    q_dual = M'*B̄*inv(P')*B̄'*C̄' - 2*M'*B̄*inv(P)*B̄'*C̄' + x0'*Ā'*C̄' + c̄'
    qq_dual = 0.5*M'*B̄*inv(P')*B̄'*M - M'*B̄*inv(P)*B̄'*M

    # DD = 0.5*λ_tmp'*Q_dual*λ_tmp + q_dual*λ_tmp + qq_dual

    # @test isapprox(D,L)

    # solve QP
    m = JuMP.Model(solver=IpoptSolver(print_level=0))

    @variable(m, λ[1:Np])
    # @objective(m, Min, λ'*λ)
    # @constraint(m, con, Q_dual*λ .== -q_dual')

    @objective(m, Max, 0.5*λ'*Q_dual*λ + q_dual*λ + qq_dual)
    @constraint(m, con2, λ[idx_inequality] .>= 0)

    # print(m)

    status = JuMP.solve(m)

    # Solution
    # println("Objective value: ", JuMP.getobjectivevalue(m))
    # println("λ = ", getvalue(λ))

    for k = 1:N
        if k != N
            idx = ((k-1)*p + 1):k*p
            results.λ[k] = JuMP.getvalue(λ[idx])
            results.λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])
        else
            idx = ((N-1)*p + 1):Np
            results.λ[N] .= JuMP.getvalue(λ[idx])
            results.λ[N][1:pI_N] = max.(0.0,results.λ[N][1:pI_N])
        end
    end
end

""" @(SIGNATURES) Penalty update """
function μ_update!(results::ConstrainedIterResults,solver::Solver)
    if solver.opts.outer_loop_update_type == :default
        μ_update_default!(results,solver)
    elseif solver.opts.outer_loop_update_type == :individual
        μ_update_individual!(results,solver)
    end
    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('default') - all penalty terms are updated"""
function μ_update_default!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    for k = 1:N
        results.μ[k] = min.(solver.opts.penalty_max, solver.opts.penalty_scaling*results.μ[k])
    end
    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('individual')- all penalty terms are updated uniquely according to indiviual improvement compared to previous iteration"""
function μ_update_individual!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    constraint_decrease_ratio = solver.opts.constraint_decrease_ratio
    penalty_max = solver.opts.penalty_max
    penalty_scaling_no  = solver.opts.penalty_scaling_no
    penalty_scaling = solver.opts.penalty_scaling

    # Stage constraints
    for k = 1:N-1
        for i = 1:p
            if p <= pI
                if max(0.0,results.C[k][i]) <= constraint_decrease_ratio*max(0.0,results.C_prev[k][i])
                    results.μ[k][i] = min(penalty_max, penalty_scaling_no*results.μ[k][i])
                else
                    results.μ[k][i] = min(penalty_max, penalty_scaling*results.μ[k][i])
                end
            else
                if abs(results.C[k][i]) <= constraint_decrease_ratio*abs(results.C_prev[k][i])
                    results.μ[k][i] = min(penalty_max, penalty_scaling_no*results.μ[k][i])
                else
                    results.μ[k][i] = min(penalty_max, penalty_scaling*results.μ[k][i])
                end
            end
        end
    end

    k = N
    for i = 1:p_N
        if p_N <= pI_N
            if max(0.0,results.C[k][i]) <= constraint_decrease_ratio*max(0.0,results.C_prev[k][i])
                results.μ[k][i] = min(penalty_max, penalty_scaling_no*results.μ[k][i])
            else
                results.μ[k][i] = min(penalty_max, penalty_scaling*results.μ[k][i])
            end
        else
            if abs(results.C[k][i]) <= constraint_decrease_ratio*abs(results.C_prev[k][i])
                results.μ[k][i] = min(penalty_max, penalty_scaling_no*results.μ[k][i])
            else
                results.μ[k][i] = min(penalty_max, penalty_scaling*results.μ[k][i])
            end
        end
    end

    return nothing
end

"""
$(SIGNATURES)
    Updates penalty (μ) and Lagrange multiplier (λ) parameters for Augmented Lagrangian method
"""
function outer_loop_update(results::ConstrainedIterResults,solver::Solver)::Nothing

    ## Lagrange multiplier updates
    solver.state.second_order_dual_update ? λ_update_second_order!(results,solver) : λ_update!(results,solver)

    ## Penalty updates
    μ_update!(results,solver)

    ## Store current constraints evaluations for next outer loop update
    results.C_prev .= deepcopy(results.C)

    return nothing
end

function outer_loop_update(results::UnconstrainedIterResults,solver::Solver)::Nothing
    return nothing
end
