"""
$(SIGNATURES)
    Lagrange multiplier updates
        -see Bertsekas 'Constrained Optimization' chapter 2 (p.135)
        -see Toussaint 'A Novel Augmented Lagrangian Approach for Inequalities and Convergent Any-Time Non-Central Updates'
"""
function λ_update!(results::ConstrainedIterResults,solver::Solver)
    p,pI,pE = get_num_constraints(solver)
    N = solver.N

    for k = 1:N-1
        results.λ[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λ[k] + results.Iμ[k]*results.C[k]))
        results.λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])
    end

    results.λN .= max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λN + results.IμN*results.CN))
end

""" @(SIGNATURES) Penalty update """
function μ_update!(results::ConstrainedIterResults,solver::Solver)
    if solver.opts.outer_loop_update == :default
        μ_update_default!(results,solver)
    elseif solver.opts.outer_loop_update == :individual
        μ_update_individual!(results,solver)
    end
    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('default') - all penalty terms are updated"""
function μ_update_default!(results::ConstrainedIterResults,solver::Solver)
    N = solver.N

    for k = 1:N-1
        results.μ[k] = min.(solver.opts.μ_max, solver.opts.γ*results.μ[k])
    end

    results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)

    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('individual')- all penalty terms are updated uniquely according to indiviual improvement compared to previous iteration"""
function μ_update_individual!(results::ConstrainedIterResults,solver::Solver)
    N = solver.N
    p,pI,pE = get_num_constraints(solver)
    n = solver.model.n

    τ = solver.opts.τ
    μ_max = solver.opts.μ_max
    γ_no  = solver.opts.γ_no
    γ = solver.opts.γ



    # Stage constraints
    for k = 1:N-1
        for i = 1:p
            if p <= pI
                if max(0.0,results.C[k][i]) <= τ*max(0.0,results.C_prev[k][i])
                    results.μ[k][i] = min(μ_max, γ_no*results.μ[k][i])
                else
                    results.μ[k][i] = min(μ_max, γ*results.μ[k][i])
                end
            else
                if abs(results.C[k][i]) <= τ*abs(results.C_prev[k][i])
                    results.μ[k][i] = min(μ_max, γ_no*results.μ[k][i])
                else
                    results.μ[k][i] = min(μ_max, γ*results.μ[k][i])
                end
            end
        end
    end

    # Terminal constraints
    for i = 1:n
        if abs(results.CN[i]) <= τ*abs(results.CN_prev[i])
            results.μN[i] = min(μ_max, γ_no*results.μN[i])
        else
            results.μN[i] = min(μ_max, γ*results.μN[i])
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
    solver.state.second_order_dual_update ? solve_batch_qp_dual(results,solver) : λ_update!(results,solver)

    ## Penalty updates
    μ_update!(results,solver)

    ## Store current constraints evaluations for next outer loop update
    results.C_prev .= deepcopy(results.C)
    results.CN_prev .= deepcopy(results.CN)

    return nothing
end

function outer_loop_update(results::UnconstrainedIterResults,solver::Solver)::Nothing
    return nothing
end
