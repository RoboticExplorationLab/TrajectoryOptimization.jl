import Base.println

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Methods for settings and solving iLQR problems
#
#     METHODS
#         solve(solver, X0, U0): Call infeasible solver.
#         solve(solver, X0, []): Call infeasible solver and set controls to zeros
#         solve(solver, U0): Solve iLQR problem with initial guess for controls
#         solve(solver): Solve iLQR problem with random initial guess for controls
#         _solve: lower-level method for setting and solving iLQR problem
#
#         outer_loop_update: Update parameters on major iterations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
$(SIGNATURES)
Solve the trajectory optimization problem defined by `solver`, with `U0` as the
initial guess for the controls
"""
function solve(solver::Solver, X0::VecOrMat, U0::VecOrMat)::Tuple{SolverResults,Dict}
    # If infeasible without control initialization, initialize controls to zero
    isempty(U0) ? U0 = zeros(solver.m,solver.N) : nothing

    # Unconstrained original problem with infeasible start: convert to a constrained problem for solver
    if isa(solver.obj, UnconstrainedObjective)
        solver.opts.unconstrained_original_problem = true
        solver.opts.infeasible = true
        obj_c = ConstrainedObjective(solver.obj)
        solver = Solver(solver.model, obj_c, integration=solver.integration, dt=solver.dt, opts=solver.opts)
    end

    results, stats = _solve(solver,U0,X0)
    return results, stats
end

function solve(solver::Solver,U0::VecOrMat)::Tuple{SolverResults,Dict}
    _solve(solver,U0)
end

function solve(solver::Solver)::Tuple{SolverResults,Dict}
    # Generate random control sequence
    U0 = rand(solver.model.m, solver.N)
    solve(solver,U0)
end

"""
$(SIGNATURES)
Warm start solver with results from a previous solve.
# Arguments
* infeasible (bool): solve problem using infeasible controls. False by default
* warm_start (bool): warm start solver by passing lagrange multipliers from the previous problem. True by default.
"""
function solve(solver::Solver, results::SolverVectorResults; infeasible=false, warm_start=:true)
    U0 = to_array(results.U)
    if infeasible
        X0 = to_array(results.X)
    else
        X0 = Array{Float64,2}(undef,0,0)
    end

    _solve(solver, U0, X0, prevResults=results )
end

"""
$(SIGNATURES)
    Solve constrained optimization problem specified by `solver`
# Arguments
* solver::Solver
* U0::Matrix{Float64} - initial control trajectory
* X0::Matrix{Float64} (optional) - initial state trajectory. If specified, it will solve use infeasible controls
* λ::Vector{Vector} (optional) - initial Lagrange multipliers for warm starts. Must be passed in as a N+1 Vector of Vector{Float64}, with the N+1th entry the Lagrange multipliers for the terminal constraint.
"""
function _solve(solver::Solver{Obj}, U0::Array{Float64,2}, X0::Array{Float64,2}=Array{Float64}(undef,0,0); prevResults=ConstrainedVectorResults())::Tuple{SolverResults,Dict} where {Obj<:Objective}
    t_start = time_ns()

    ## Unpack model, objective, and solver parameters
    n,m,N = get_sizes(solver)

    # Check for minimum time solve
    # is_min_time(solver) ? solver.opts.minimum_time = true : solver.opts.minimum_time = false

    # Check for infeasible start
    isempty(X0) ? solver.opts.infeasible = false : solver.opts.infeasible = true

    # Check for constrained solve
    if solver.opts.infeasible || solver.opts.minimum_time || Obj <: ConstrainedObjective
        solver.opts.constrained = true
    else
        solver.opts.constrained = false
        iterations_outerloop_original = solver.opts.iterations_outerloop
        solver.opts.iterations_outerloop = 1
    end

    #****************************#
    #       INITIALIZATION       #
    #****************************#
    if isempty(prevResults)
        results = init_results(solver, X0, U0)
    else
        results = prevResults
    end

    # Unpack results for convenience
    X = results.X # state trajectory
    U = results.U # control trajectory
    X_ = results.X_ # updated state trajectory
    U_ = results.U_ # updated control trajectory

    # Set up logger
    solver.opts.verbose == false ? min_level = Logging.Warn : min_level = InnerLoop

    logger = SolverLogger(min_level)
    inner_cols = [:iter, :cost, :expected, :actual, :z, :α, :c_max, :info]
    inner_widths = [5,     14,      12,        12,  10, 10,   10,      50]
    outer_cols = [:outeriter, :iter, :iterations, :info]
    outer_widths = [10,          5,        12,        40]
    add_level!(logger, InnerLoop, inner_cols, inner_widths, print_color=:green,indent=4)
    add_level!(logger, OuterLoop, outer_cols, outer_widths, print_color=:yellow,indent=0)

    update_constraints!(results, solver)

    #****************************#
    #           SOLVER           #
    #****************************#
    ## Initial rollout
    if !solver.opts.infeasible #&& isempty(prevResults)
        X[1] = solver.obj.x0
        flag = rollout!(results,solver) # rollout new state trajectoy

        if !flag
            @info "Bad initial control sequence, setting initial control to zero"
            results.U .= zeros(mm,N)
            rollout!(results,solver)
        end
    end

    if solver.opts.infeasible
        if solver.control_integration == :foh
            calculate_derivatives!(results, solver, results.X, results.U)
            calculate_midpoints!(results, solver, results.X, results.U)
        end
        update_constraints!(results,solver,results.X,results.U)
    end
    ##

    # Solver Statistics
    iter = 0 # counter for total number of iLQR iterations
    iter_outer = 1
    iter_inner = 1
    time_setup = time_ns() - t_start
    J_hist = Vector{Float64}()
    c_max_hist = Vector{Float64}()
    t_solve_start = time_ns()

    #****************************#
    #         OUTER LOOP         #
    #****************************#

    dJ = Inf
    gradient = Inf
    Δv = [Inf, Inf]

    with_logger(logger) do
    for j = 1:solver.opts.iterations_outerloop
        iter_outer = j
        @info "Outer loop $j (begin)"

        if solver.opts.constrained && j == 1
            results.gs_prev .= deepcopy(results.gs)
            results.gc_prev .= deepcopy(results.gc)
            results.hs_prev .= deepcopy(results.hs)
            results.hc_prev .= deepcopy(results.hc)
        end
        c_max = 0.  # Init max constraint violation to increase scope
        dJ_zero_counter = 0  # Count how many time the forward pass is unsuccessful

        J_prev = cost(solver, results, X, U)
        j == 1 ? push!(J_hist, J_prev) : nothing  # store the first cost

        #****************************#
        #         INNER LOOP         #
        #****************************#

        for ii = 1:solver.opts.iterations_innerloop
            iter_inner = ii

            ### BACKWARD PASS ###
            calculate_jacobians!(results, solver)
            Δv = backwardpass!(results, solver)

            ### FORWARDS PASS ###
            J = forwardpass!(results, solver, Δv)#, J_prev)
            push!(J_hist,J)

            # increment iLQR inner loop counter
            iter += 1

            if solver.opts.live_plotting
                display(plot(to_array(results.U)'))
            end

            ### UPDATE RESULTS ###
            X .= deepcopy(X_)
            U .= deepcopy(U_)

            dJ = copy(abs(J-J_prev)) # change in cost
            J_prev = copy(J)
            dJ == 0 ? dJ_zero_counter += 1 : dJ_zero_counter = 0

            if solver.opts.constrained
                c_max = max_violation(results)
                push!(c_max_hist, c_max)
                @logmsg InnerLoop :c_max value=c_max
            end

            ## Check gradients for convergence ##
            todorov_grad = calculate_todorov_gradient(results)

            @logmsg InnerLoop :grad value=todorov_grad
            gradient = todorov_grad

            # Print Log
            @logmsg InnerLoop :iter value=iter
            @logmsg InnerLoop :cost value=J
            @logmsg InnerLoop :dJ value=dJ loc=3
            @logmsg InnerLoop :j value=j
            @logmsg InnerLoop :zero_counter value=dJ_zero_counter

            ii % 10 == 1 ? print_header(logger,InnerLoop) : nothing
            print_row(logger,InnerLoop)

            evaluate_convergence(solver,:inner,dJ,c_max,gradient,iter,j,dJ_zero_counter) ? break : nothing
            if J > solver.opts.max_cost
                error("Cost exceded maximum allowable cost")
            end
        end
        ### END INNER LOOP ###

        #****************************#
        #      OUTER LOOP UPDATE     #
        #****************************#

        # update multiplier and penalty terms
        outer_loop_update(results,solver)
        update_constraints!(results, solver)
        J_prev = cost(solver, results, results.X, results.U)

        # Logger output
        @logmsg OuterLoop :outeriter value=j
        @logmsg OuterLoop :iter value=iter
        @logmsg OuterLoop :iterations value=iter_inner
        print_header(logger,OuterLoop)
        print_row(logger,OuterLoop)

        #****************************#
        #    TERMINATION CRITERIA    #
        #****************************#
        # Check if maximum constraint violation satisfies termination criteria AND cost or gradient tolerance convergence
        evaluate_convergence(solver,:outer,dJ,c_max,gradient,iter,0,dJ_zero_counter) ? break : nothing
    end
    end
    ### END OUTER LOOP ###

    solver.opts.constrained ? nothing : solver.opts.iterations_outerloop = iterations_outerloop_original

    # Run Stats
    stats = Dict("iterations"=>iter,
                 "major iterations"=>iter_outer,
                 "runtime"=>float(time_ns() - t_solve_start)/1e9,
                 "setup_time"=>float(time_setup)/1e9,
                 "cost"=>J_hist,
                 "c_max"=>c_max_hist)

    if ((iter_outer == solver.opts.iterations_outerloop) && (iter_inner == solver.opts.iterations)) && solver.opts.verbose
        @warn "*Solve reached max iterations*"
    end

    ### Infeasible -> feasible trajectory
    if solver.opts.infeasible
        @info "Infeasible solve complete"

        # run single backward pass/forward pass to get dynamically feasible solution (ie, remove infeasible controls)
        results_feasible = get_feasible_trajectory(results,solver)

        # resolve feasible solution if necessary (should be fast)
        if solver.opts.resolve_feasible
            @info "Resolving feasible"

            # create unconstrained solver from infeasible solver if problem is unconstrained
            if solver.opts.unconstrained_original_problem
                obj_uncon = UnconstrainedObjective(solver.obj.Q,solver.obj.R,solver.obj.Qf,solver.obj.tf,solver.obj.x0,solver.obj.xf)
                solver_feasible = Solver(solver.model,obj_uncon,integration=solver.integration,dt=solver.dt,opts=solver.opts)
            else
                solver_feasible = solver
            end

            # Resolve feasible problem with warm start
            results_feasible, stats_feasible = solve(solver_feasible,to_array(results_feasible.U),prevResults=results_feasible)

            # Merge stats
            for key in keys(stats_feasible)
                stats[key * " (infeasible)"] = stats[key]
            end
            stats["iterations"] += stats_feasible["iterations"]
            stats["major iterations"] += stats_feasible["major iterations"]
            stats["runtime"] += stats_feasible["runtime"]
            stats["setup_time"] += stats_feasible["setup_time"]
            append!(stats["cost"], stats_feasible["cost"])
            append!(stats["c_max"], stats_feasible["c_max"])
        end

        # return feasible results
        @info "***Solve Complete***"
        return results_feasible, stats

    # if feasible solve, return results
    else
        @info "***Solve Complete***"
        return results, stats
    end
end

"""
$(SIGNATURES)
    Check convergence
    -return true is convergence criteria is met, else return false
"""

function evaluate_convergence(solver::Solver, loop::Symbol, dJ::Float64, c_max::Float64, gradient::Float64, iter_total::Int64, iter_outerloop::Int64, dJ_zero_counter::Int)
    # Check total iterations
    if iter_total >= solver.opts.iterations
        return true
    end
    if loop == :inner
        # Check for gradient convergence
        if ((~solver.opts.constrained && gradient < solver.opts.gradient_tolerance) || (solver.opts.constrained && gradient < solver.opts.gradient_intermediate_tolerance && iter_outerloop != solver.opts.iterations_outerloop))
            # @logmsg OuterLoop "--iLQR (inner loop) gradient eps criteria met at iteration: $ii"
            return true
        elseif ((solver.opts.constrained && gradient < solver.opts.gradient_tolerance && c_max < solver.opts.constraint_tolerance))
            # @logmsg OuterLoop "--iLQR (inner loop) gradient and constraint eps criteria met at iteration: $ii"
            return true
        end

        # Outer loop update if forward pass is repeatedly unsuccessful
        if dJ_zero_counter >= 10
            return true
        end

        # Check for cost convergence
            # note the  dJ > 0 criteria exists to prevent loop exit when forward pass makes no improvement
        if ((~solver.opts.constrained && (0.0 < dJ < solver.opts.cost_tolerance)) || (solver.opts.constrained && (0.0 < dJ < solver.opts.cost_intermediate_tolerance) && iter_outerloop != solver.opts.iterations_outerloop))
            # @logmsg OuterLoop "--iLQR (inner loop) cost eps criteria met at iteration: $ii"
            # ~solver.opts.constrained ? @info "Unconstrained solve complete": nothing
            return true
        elseif ((solver.opts.constrained && (0.0 < dJ < solver.opts.cost_tolerance) && c_max < solver.opts.constraint_tolerance))
            # @logmsg OuterLoop "--iLQR (inner loop) cost and constraint eps criteria met at iteration: $ii"
            return true
        end
    end

    if loop == :outer
        if solver.opts.constrained
            if c_max < solver.opts.constraint_tolerance && ((0.0 < dJ < solver.opts.cost_tolerance) || gradient < solver.opts.gradient_tolerance)
                return true
            end
        end
    end
    return false
end

"""
$(SIGNATURES)
    Infeasible start solution is run through time varying LQR to track state and control trajectories
"""
function get_feasible_trajectory(results::SolverIterResults,solver::Solver)::SolverIterResults
    # turn off infeasible solve
    solver.opts.infeasible = false

    # remove infeasible components
    results_feasible = remove_infeasible_controls_to_unconstrained_results(results,solver)

    # backward pass - project infeasible trajectory into feasible space using time varying lqr
    Δv = backwardpass!(results_feasible, solver)

    # forward pass
    forwardpass!(results_feasible,solver,Δv)#,cost(solver, results_feasible, results_feasible.X, results_feasible.U))

    # update trajectories
    results_feasible.X .= deepcopy(results_feasible.X_)
    results_feasible.U .= deepcopy(results_feasible.U_)

    # return constrained results if input was constrained
    if !solver.opts.unconstrained_original_problem
        results_feasible = unconstrained_to_constrained_results(results_feasible,solver,results.λ,results.λN)
        update_constraints!(results_feasible,solver,results_feasible.X,results_feasible.U)
        calculate_jacobians!(results_feasible,solver)
    end
    if solver.control_integration == :foh
        calculate_derivatives!(results_feasible,solver,results_feasible.X,results_feasible.U)
        calculate_midpoints!(results_feasible,solver,results_feasible.X,results_feasible.U)
    end

    return results_feasible
end

"""
$(SIGNATURES)
    Lagrange multiplier updates
        -see Bertsekas 'Constrained Optimization' chapter 2 (p.135)
        -see Toussaint 'A Novel Augmented Lagrangian Approach for Inequalities and Convergent Any-Time Non-Central Updates'
"""
function λ_update!(results::ConstrainedIterResults,solver::Solver)
    N = solver.N
    for k = 1:N
        if k != 1
            results.λs[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λs[k] + results.Iμs[k]*results.gs[k]))
            results.λs[k] = max.(0.0,results.λs[k])

            results.κs[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.κs[k] + results.Iνs[k]*results.hs[k]))
        end
        if k != N || solver.control_integration == :foh
            results.λc[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λc[k] + results.Iμc[k]*results.gc[k]))
            results.λc[k] = max.(0.0,results.λc[k])

            results.κc[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.κc[k] + results.Iνc[k]*results.hc[k]))
        end
    end
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
    for k = 1:N
        if k != 1
            results.μs[k] = min.(solver.opts.μ_max, solver.opts.γ*results.μs[k])
            results.νs[k] = min.(solver.opts.μ_max, solver.opts.γ*results.νs[k])
        end
        if k != N || solver.control_integration == :foh
            results.μc[k] = min.(solver.opts.μ_max, solver.opts.γ*results.μc[k])
            results.νc[k] = min.(solver.opts.μ_max, solver.opts.γ*results.νc[k])
        end
    end
    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('individual')- all penalty terms are updated uniquely according to indiviual improvement compared to previous iteration"""
function μ_update_individual!(results::ConstrainedIterResults,solver::Solver)
    N = solver.N
    τ = solver.opts.τ
    μ_max = solver.opts.μ_max
    γ_no  = solver.opts.γ_no
    γ = solver.opts.γ

    pIs, pIc, pEs, pEsN, pEc = get_num_constraints(solver)

    # Stage constraints
    for k = 1:N
        if k != 1
            for i = 1:pIs
                if max(0.0,results.gs[k][i]) <= τ*max(0.0,results.gs_prev[k][i])
                    results.μs[k][i] = min(μ_max, γ_no*results.μs[k][i])
                else
                    results.μs[k][i] = min(μ_max, γ*results.μs[k][i])
                end
            end

            for i = 1:pEs
                if abs(results.hs[k][i]) <= τ*abs(results.hs_prev[k][i])
                    results.νs[k][i] = min(μ_max, γ_no*results.νs[k][i])
                else
                    results.νs[k][i] = min(μ_max, γ*results.νs[k][i])
                end
            end
        end

        if k != N || solver.control_integration == :foh
            for i = 1:pIc
                if max(0.0,results.gc[k][i]) <= τ*max(0.0,results.gc_prev[k][i])
                    results.μc[k][i] = min(μ_max, γ_no*results.μc[k][i])
                else
                    results.μc[k][i] = min(μ_max, γ*results.μc[k][i])
                end
            end

            for i = 1:pEc
                if abs(results.hc[k][i]) <= τ*abs(results.hc_prev[k][i])
                    results.νc[k][i] = min(μ_max, γ_no*results.νc[k][i])
                else
                    results.νc[k][i] = min(μ_max, γ*results.νc[k][i])
                end
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
    λ_update!(results,solver)

    ## Penalty updates
    μ_update!(results,solver)

    ## Store current constraints evaluations for next outer loop update
    results.gs_prev .= deepcopy(results.gs)
    results.gc_prev .= deepcopy(results.gc)
    results.hs_prev .= deepcopy(results.hs)
    results.hc_prev .= deepcopy(results.hc)

    return nothing
end

function outer_loop_update(results::UnconstrainedIterResults,solver::Solver)::Nothing
    return nothing
end

"""
$(SIGNATURES)
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function calculate_todorov_gradient(res::SolverVectorResults)
    N = length(res.X)
    maxes = zeros(N)
    for k = 1:N
        maxes[k] = maximum(abs.(res.d[k])./(abs.(res.U[k]).+1))
    end
    mean(maxes)
end
