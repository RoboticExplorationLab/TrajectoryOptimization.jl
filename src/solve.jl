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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
$(SIGNATURES)
Solve the trajectory optimization problem defined by `solver`, with `U0` as the
initial guess for the controls
"""
function solve(solver::Solver, X0::VecOrMat, U0::VecOrMat)::Tuple{SolverResults,Dict}
    # If infeasible without control initialization, initialize controls to zero
    isempty(U0) ? U0 = zeros(solver.m,solver.N-1) : nothing

    # Unconstrained original problem with infeasible start: convert to a constrained problem for solver
    if isa(solver.obj, UnconstrainedObjective)
        solver.opts.unconstrained_original_problem = true
        solver.state.infeasible = true
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
    U0 = rand(solver.model.m, solver.N-1)
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
    if warm_start
        λ = deepcopy(results.λ)
        push!(λ, results.λN)
    else
        λ = []
    end
    _solve(solver, U0, X0, λ=λ)
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
function _solve(solver::Solver{M,Obj}, U0::Array{Float64,2}, X0::Array{Float64,2}=Array{Float64}(undef,0,0); λ::Vector=[], μ::Vector=[], prevResults=ConstrainedVectorResults(), bmark_stats::BenchmarkGroup=BenchmarkGroup())::Tuple{SolverResults,Dict} where {M<:Model,Obj<:Objective}
    # Start timer
    t_start = time_ns()

    # Reset solver state
    reset_SolverState(solver.state)

    # Check for penalty burn-in mode
    if !solver.opts.use_penalty_burnin
        solver.state.penalty_only = false
    end

    # Check for minimum time solve
    is_min_time(solver) ? solver.state.minimum_time = true : solver.state.minimum_time = false

    # Check for infeasible start
    isempty(X0) ? solver.state.infeasible = false : solver.state.infeasible = true

    # Check for constrained solve
    if solver.state.infeasible || solver.state.minimum_time || Obj <: ConstrainedObjective
        solver.state.constrained = true
    else
        solver.state.constrained = false
        iterations_outerloop_original = solver.opts.iterations_outerloop
        solver.opts.iterations_outerloop = 1
    end

    #****************************#
    #       INITIALIZATION       #
    #****************************#
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    if isempty(prevResults)
        results = init_results(solver, X0, U0, λ=λ, μ=μ)
    else
        results = prevResults
    end

    # Initialized backward pass expansion terms
    bp = BackwardPassZOH(nn,mm,N)

    # Unpack results for convenience
    X = results.X # state trajectory
    U = results.U # control trajectory
    X_ = results.X_ # updated state trajectory
    U_ = results.U_ # updated control trajectory

    # Set up logger
    logger = default_logger(solver)


    #****************************#
    #           SOLVER           #
    #****************************#
    ## Initial rollout
    if !solver.state.infeasible #&& isempty(prevResults)
        X[1][1:n] = solver.obj.x0
        flag = rollout!(results,solver) # rollout new state trajectoy
        !flag ? error("Bad initial control sequence") : nothing
    end

    if solver.state.constrained
        update_constraints!(results, solver)

        # Update constraints Jacobians; if fixed (ie, no custom constraints) set solver state to not update
        update_jacobians!(results,solver,:constraints)
        !check_custom_constraints(solver.obj) ? solver.state.fixed_constraint_jacobians = true : solver.state.fixed_constraint_jacobians = false
        !check_custom_terminal_constraints(solver.obj) ? solver.state.fixed_terminal_constraint_jacobian = true : solver.state.fixed_terminal_constraint_jacobian = false
    end

    # Solver Statistics
    iter = 0 # counter for total number of iLQR iterations
    iter_outer = 1
    iter_inner = 1
    time_setup = time_ns() - t_start
    J_hist = Vector{Float64}()
    grad_norm_hist = Vector{Float64}()
    c_max_hist = Vector{Float64}()
    max_cn_hist = Vector{Float64}()
    min_eig_hist = Vector{Float64}()
    outer_updates = Int[]
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

        if solver.state.constrained && j == 1
            copyto!(results.C_prev,results.C)
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
            update_jacobians!(results, solver)
            Δv = backwardpass!(results, solver, bp)

            # condition numbers and min eigen value
            max_cn = 0.
            min_eig = Inf
            for kkk = 1:N-1
                cn = cond(bp.Quu_reg[kkk])
                if cn > max_cn
                    max_cn = cn
                end
                me = minimum(real.(eigvals(bp.Quu_reg[kkk])))
                if me < min_eig
                    min_eig = me
                end
            end
            push!(max_cn_hist,max_cn)
            push!(min_eig_hist,min_eig)

            ### FORWARDS PASS ###
            J = forwardpass!(results, solver, Δv, J_prev)
            push!(J_hist,J)

            ## Check gradients for convergence ##
            solver.opts.use_gradient_aula ? gradient = gradient_AuLa(results,solver,bp) : gradient = gradient_todorov(results)
            push!(grad_norm_hist,gradient)

            # increment iLQR inner loop counter
            iter += 1

            solver.opts.live_plotting ? display(plot(to_array(results.U)')) : nothing

            ### UPDATE RESULTS ###
            copyto!(X,X_)
            copyto!(U,U_)

            dJ = copy(abs(J-J_prev)) # change in cost
            J_prev = copy(J)
            dJ == 0 ? dJ_zero_counter += 1 : dJ_zero_counter = 0

            if solver.state.constrained
                c_max = max_violation(results)
                push!(c_max_hist, c_max)
                @logmsg InnerLoop :c_max value=c_max

                if c_max <= solver.opts.constraint_tolerance_second_order_dual_update && solver.opts.use_second_order_dual_update
                    solver.state.second_order_dual_update = true
                end
                if (solver.state.penalty_only && c_max < solver.opts.constraint_tolerance_coarse) && solver.opts.use_penalty_burnin
                    solver.state.penalty_only = false
                    @logmsg InnerLoop "Switching to multipier updates"
                end

                if solver.state.second_order_dual_update
                    @logmsg InnerLoop "λ 2-update"
                end
            end

            @logmsg InnerLoop :iter value=iter
            @logmsg InnerLoop :cost value=J
            @logmsg InnerLoop :dJ value=dJ loc=3
            @logmsg InnerLoop :grad value=gradient
            @logmsg InnerLoop :j value=j
            @logmsg InnerLoop :max_cn value=max_cn
            @logmsg InnerLoop :min_eig value=min_eig
            @logmsg InnerLoop :zero_count value=dJ_zero_counter


            ii % 10 == 1 ? print_header(logger,InnerLoop) : nothing
            print_row(logger,InnerLoop)

            evaluate_convergence(solver,:inner,dJ,c_max,gradient,iter,j,dJ_zero_counter) ? break : nothing
            if J > solver.opts.max_cost_value
                error("Cost exceded maximum allowable cost")
            end
        end
        ### END INNER LOOP ###

        #****************************#
        #      OUTER LOOP UPDATE     #
        #****************************#

        # update multiplier and penalty terms
        outer_loop_update(results,solver,bp,j)
        update_constraints!(results, solver)
        J_prev = cost(solver, results, results.X, results.U)

        # Logger output
        @logmsg OuterLoop :outeriter value=j
        @logmsg OuterLoop :iter value=iter
        @logmsg OuterLoop :iterations value=iter_inner
        print_header(logger,OuterLoop)
        print_row(logger,OuterLoop)

        push!(outer_updates,iter)

        #****************************#
        #    TERMINATION CRITERIA    #
        #****************************#
        # Check if maximum constraint violation satisfies termination criteria AND cost or gradient tolerance convergence
        evaluate_convergence(solver,:outer,dJ,c_max,gradient,iter,0,dJ_zero_counter) ? break : nothing
    end
    end
    ### END OUTER LOOP ###

    solver.state.constrained ? nothing : solver.opts.iterations_outerloop = iterations_outerloop_original

    # Run Stats
    stats = Dict("iterations"=>iter,
        "major iterations"=>iter_outer,
        "runtime"=>float(time_ns() - t_solve_start)/1e9,
        "setup_time"=>float(time_setup)/1e9,
        "cost"=>J_hist,
        "c_max"=>c_max_hist,
        "gradient_norm"=>grad_norm_hist,
        "max_condition_number"=>max_cn_hist,
        "outer_updates"=>outer_updates)

    if !isempty(bmark_stats)
        for key in intersect(keys(bmark_stats), keys(stats))
            if stats[key] isa Vector
                if length(stats[key]) > 0
                    bmark_stats[key] = stats[key][end]
                else
                    bmark_stats[key] = 0
                end
            else
                bmark_stats[key] = stats[key]
            end
        end
    end

    if ((iter_outer == solver.opts.iterations_outerloop) && (iter_inner == solver.opts.iterations)) && solver.opts.verbose
        @warn "*Solve reached max iterations*"
    end

    ### Infeasible -> feasible trajectory
    if solver.state.infeasible
        @info "Infeasible solve complete"

        # run single backward pass/forward pass to get dynamically feasible solution (ie, remove infeasible controls)
        results_feasible = get_feasible_trajectory(results,solver)

        # resolve feasible solution if necessary (should be fast)
        if solver.opts.resolve_feasible
            @info "Resolving feasible"

            # create unconstrained solver from infeasible solver if problem is unconstrained
            if solver.opts.unconstrained_original_problem
                obj = solver.obj
                obj_uncon = UnconstrainedObjective(obj.cost, obj.tf, obj.x0, obj.xf)
                solver_feasible = Solver(solver.model,obj_uncon,integration=solver.integration,dt=solver.dt,opts=solver.opts)
            else
                solver_feasible = solver
            end

            # Reset second order dual update flag
            solver_feasible.state.second_order_dual_update = false

            # Resolve feasible problem with warm start
            results_feasible, stats_feasible = _solve(solver_feasible,to_array(results_feasible.U))

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
            append!(stats["gradient_norm"], stats_feasible["gradient_norm"])
            append!(stats["max_condition_number"], stats_feasible["max_condition_number"])
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
        if ((~solver.state.constrained && gradient < solver.opts.gradient_tolerance) || (solver.state.constrained && gradient < solver.opts.gradient_tolerance_intermediate && iter_outerloop != solver.opts.iterations_outerloop))
            return true
        elseif ((solver.state.constrained && gradient < solver.opts.gradient_tolerance && c_max < solver.opts.constraint_tolerance))
            return true
        end

        # Outer loop update if forward pass is repeatedly unsuccessful
        if dJ_zero_counter > solver.opts.dJ_counter_limit
            return true
        end

        # Check for cost convergence
            # note the  dJ > 0 criteria exists to prevent loop exit when forward pass makes no improvement
        if ((~solver.state.constrained && (0.0 < dJ < solver.opts.cost_tolerance)) || (solver.state.constrained && (0.0 < dJ < solver.opts.cost_tolerance_intermediate) && iter_outerloop != solver.opts.iterations_outerloop))
            return true
        elseif ((solver.state.constrained && (0.0 < dJ < solver.opts.cost_tolerance) && c_max < solver.opts.constraint_tolerance))
            return true
        end
    end

    if loop == :outer
        if solver.state.constrained
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
    remove_infeasible_controls!(results,solver)

    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_controls(solver)

    # Initialized backward pass expansion terms
    bp = BackwardPassZOH(nn,mm,N)

    # backward pass - project infeasible trajectory into feasible space using time varying lqr
    Δv = backwardpass!(results, solver, bp)

    # forward pass
    forwardpass!(results,solver,Δv,cost(solver, results, results.X, results.U))

    # update trajectories
    copyto!(results.X, results.X_)
    copyto!(results.U, results.U_)

    # return constrained results if input was constrained
    if !solver.opts.unconstrained_original_problem
        update_constraints!(results,solver,results.X,results.U)
        update_jacobians!(results,solver)
    else
        solver.state.constrained = false
    end

    return results
end

"""
$(SIGNATURES)
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function gradient_todorov(res::SolverVectorResults)
    N = length(res.X)
    maxes = zeros(N)
    for k = 1:N-1
        maxes[k] = maximum(abs.(res.d[k])./(abs.(res.U[k]).+1))
    end
    mean(maxes)
end
