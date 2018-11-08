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
function solve(solver::Solver, X0::VecOrMat, U0::VecOrMat; prevResults::SolverResults=ConstrainedVectorResults())::Tuple{SolverResults,Dict}
    # solver = Solver(solver)
    # Initialize zero controls if none are passed in
    if isempty(U0)
        U0 = zeros(solver.m,solver.N)
    end

    # Unconstrained original problem with infeasible start: convert to a constrained problem for solver
    if isa(solver.obj, UnconstrainedObjective)
        solver.opts.unconstrained_original_problem = true
        solver.opts.infeasible = true
        obj_c = ConstrainedObjective(solver.obj)
        solver = Solver(solver.model, obj_c, integration=solver.integration, dt=solver.dt, opts=solver.opts)
    end

    results, stats = _solve(solver,U0,X0,prevResults=prevResults)
    return results, stats
end

function solve(solver::Solver,U0::VecOrMat; prevResults::SolverResults=ConstrainedVectorResults())::Tuple{SolverResults,Dict}
    _solve(solver,U0,prevResults=prevResults)
end

function solve(solver::Solver)::Tuple{SolverResults,Dict}
    # Generate random control sequence
    U0 = rand(solver.model.m, solver.N)
    solve(solver,U0)
end


"""
$(SIGNATURES)
    Solve constrained optimization problem specified by `solver`
"""
function _solve(solver::Solver{Obj}, U0::Array{Float64,2}, X0::Array{Float64,2}=Array{Float64}(undef,0,0); prevResults::SolverResults=ConstrainedVectorResults())::Tuple{SolverResults,Dict} where {Obj<:Objective}
    t_start = time_ns()

    ## Unpack model, objective, and solver parameters
    N = solver.N # number of iterations for the solver (ie, knotpoints)
    n = solver.model.n # number of states
    m = solver.model.m # number of control inputs

    # Check for minimum time solve
    is_minimum_time(solver) ? solver.opts.minimum_time = true : solver.opts.minimum_time = false

    # Check for infeasible start
    isempty(X0) ? solver.opts.infeasible = false : solver.opts.infeasible = true

    # Check for constrained solve
    if solver.opts.infeasible || solver.opts.minimum_time || Obj <: ConstrainedObjective
        solver.opts.constrained = true
    else
        solver.opts.constrained = false
    end

    #****************************#
    #       INITIALIZATION       #
    #****************************#
    if !solver.opts.constrained
        @info "Solving Unconstrained Problem..."

        iterations_outerloop_original = solver.opts.iterations_outerloop
        solver.opts.iterations_outerloop = 1
        if solver.opts.use_static
            results = UnconstrainedStaticResults(n,m,N)
        else
            results = UnconstrainedVectorResults(n,m,N)
        end
        copyto!(results.U, U0)
    else
        if solver.opts.minimum_time
            solver.opts.infeasible ? sep = " and " : sep = " with "
            solve_string = sep * "minimum time..."

            # Initialize controls with sqrt(dt)
            U_init = [U0; ones(1,size(U0,2))*sqrt(get_initial_dt(solver))]
        else
            solve_string = "..."
            U_init = U0
        end

        if solver.opts.infeasible
            solve_string =  "Solving Constrained Problem with Infeasible Start" * solve_string

            # Generate infeasible controls
            ui = infeasible_controls(solver,X0,U_init)  # generates n additional control input sequences that produce the desired infeasible state trajectory
            U_init = [U_init; ui]  # augment control with additional control inputs that produce infeasible state trajectory

            # Assign state trajectory
            X_init = X0
        else
            solve_string = "Solving Constrained Problem" * solve_string
            X_init = zeros(n,N)
        end

        @info solve_string

        # Get system and constraint counts
        p,pI,pE = get_num_constraints(solver)
        m̄,mm = get_num_controls(solver)

        ## Initialize results
        solver.opts.use_static ? results = ConstrainedStaticResults(n,mm,p,N) : results = ConstrainedVectorResults(n,mm,p,N)

        # Set initial penalty term values
        results.μ .*= solver.opts.μ_initial

        # Special penalty initializations
        if solver.opts.minimum_time
            for k = 1:solver.N
                results.μ[k][p] *= solver.opts.μ_initial_minimum_time_equality
                results.μ[k][m̄] *= solver.opts.μ_initial_minimum_time_inequality
                results.μ[k][m̄+m̄] *= solver.opts.μ_initial_minimum_time_inequality
            end
        end
        if solver.opts.infeasible
            nothing #TODO
        end

        # Set initial regularization
        results.ρ[1] = solver.opts.ρ_initial

        # Assign initial trajectories to results
        copyto!(results.X, X_init)
        copyto!(results.U, U_init)

        # Initial evaluation of constraints
        update_constraints!(results,solver,results.X,results.U)
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

    #****************************#
    #           SOLVER           #
    #****************************#
    # Initial rollout
    if !solver.opts.infeasible
        X[1] = solver.obj.x0
        flag = rollout!(results,solver) # rollout new state trajectoy

        if !flag
            # error('Initial rollout failed')
            if solver.opts.verbose
                println("Bad initial control sequence, setting initial control to zero")
            end
            results.U .= zeros(mm,N)
            rollout!(results,solver)
        end
    end

    # Solver Statistics
    iter = 0 # counter for total number of iLQR iterations
    iter_outer = 1
    iter_inner = 1
    time_setup = time_ns() - t_start
    J_hist = Vector{Float64}()
    c_max_hist = Vector{Float64}()
    t_solve_start = time_ns()

    ## Relative cost analysis
    cost_minimum_time = 0.0
    cost_infeasible = 0.0
    cost_con = cost_constraints(solver,results)
    for k = 1:solver.N-1
        solver.opts.minimum_time ? cost_minimum_time += solver.opts.R_minimum_time*U[k][m̄]^2 : nothing
        solver.opts.infeasible ? cost_infeasible += 0.5*solver.opts.R_infeasible*U[k][m̄.+(1:n)]'*U[k][m̄.+(1:n)] : nothing
    end
    @info "Cost (minimum time): $cost_minimum_time | Cost (constraints): $cost_con \n"

    # error("stop here for now")
    #****************************#
    #         OUTER LOOP         #
    #****************************#

    dJ = Inf
    gradient = Inf
    Δv = [Inf, Inf]
    sqrt_tolerance = false

    with_logger(logger) do
    for j = 1:solver.opts.iterations_outerloop
        iter_outer = j
        @info "Outer loop $j (begin)"

        if solver.opts.constrained && j == 1
            results.C_prev .= deepcopy(results.C)
            results.CN_prev .= deepcopy(results.CN)
        end

        c_max = 0.  # Initialize max constraint violation to increase scope
        J_prev = cost(solver, results, X, U)
        j == 1 ? push!(J_hist, J_prev) : nothing  # store the first cost

        #****************************#
        #         INNER LOOP         #
        #****************************#

        for ii = 1:solver.opts.iterations
            iter_inner = ii

            # BACKWARD PASS
            calculate_jacobians!(results, solver)
            Δv = backwardpass!(results, solver)

            # FORWARDS PASS
            J = forwardpass!(results, solver, Δv)

            # UPDATE RESULTS
            X .= deepcopy(X_)
            U .= deepcopy(U_)

            dJ = copy(abs(J_prev-J)) # change in cost
            J_prev = copy(J)

            # increment iLQR inner loop counter
            iter += 1

            # live plotting for debugging
            if solver.opts.live_plotting
                p1 = plot(to_array(results.X)',label="",ylabel="state")
                p2 = plot(to_array(results.U)[:,1:solver.N-1]',label="",ylabel="control")
                plt = plot(p1,p2,layout=(2,1))
                # plt = plot(to_array(U)[:,1:solver.N-1]',label="",xlabel="time step (k)",ylabel="control u",title="Control")
                display(plt)
            end

            push!(J_hist,J)

            if solver.opts.constrained
                c_max = max_violation(results)
                push!(c_max_hist, c_max)
                @logmsg InnerLoop :c_max value=c_max
            end

            # Check gradients for convergence
            d_grad = maximum(map((x)->maximum(abs.(x)),results.d))
            s_grad = maximum(abs.(results.s[1]))
            todorov_grad = calculate_todorov_gradient(results)
            gradient = todorov_grad

            # Print Log
            @logmsg InnerLoop :dgrad value=d_grad
            @logmsg InnerLoop :sgrad value=s_grad
            @logmsg InnerLoop :grad value=todorov_grad
            @logmsg InnerLoop :iter value=iter
            @logmsg InnerLoop :cost value=J
            @logmsg InnerLoop :dJ value=dJ loc=3
            @logmsg InnerLoop :j value=j

            # if iter > 1
            #     c_diff = abs(c_max-c_max_hist[end-1])
            # else
            #     c_diff = 0.
            # end
            # @logmsg InnerLoop :c_diff value=c_diff

            if ii % 10 == 1
                print_header(logger,InnerLoop)
            end
            print_row(logger,InnerLoop)

            cost_minimum_time = 0.0
            cost_infeasible = 0.0
            cost_con = cost_constraints(solver,results)
            cost_controls = 0.0
            for k = 1:solver.N-1
                solver.opts.minimum_time ? cost_minimum_time += solver.opts.R_minimum_time*U[k][m̄]^2 : nothing
                solver.opts.infeasible ? cost_infeasible += 0.5*solver.opts.R_infeasible*U[k][m̄.+(1:n)]'*U[k][m̄.+(1:n)] : nothing
            end
            @info "Cost (minimum time): $cost_minimum_time| Cost (constraints): $cost_con"
            @info ""
            @info ""

            evaluate_convergence(solver,:inner,dJ,c_max,gradient,j) ? break : nothing
        end
        ### END INNER LOOP ###

        #****************************#
        #      OUTER LOOP UPDATE     #
        #****************************#
        outer_loop_update(results,solver,false)

        # Logger output
        @logmsg OuterLoop :outeriter value=j
        @logmsg OuterLoop :iter value=iter
        @logmsg OuterLoop :iterations value=iter_inner
        # print_header(logger,OuterLoop)
        # print_row(logger,OuterLoop)

        #****************************#
        #    TERMINATION CRITERIA    #
        #****************************#
        evaluate_convergence(solver,:outer,dJ,c_max,gradient,0) ? break : nothing
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

function evaluate_convergence(solver::Solver,loop::Symbol,dJ::Float64,c_max::Float64,gradient::Float64,iteration::Int64)
    if loop == :inner
        # Check for gradient convergence
        if ((~solver.opts.constrained && gradient < solver.opts.gradient_tolerance) || (solver.opts.constrained && gradient < solver.opts.gradient_intermediate_tolerance && iteration != solver.opts.iterations_outerloop))
            # @logmsg OuterLoop "--iLQR (inner loop) gradient eps criteria met at iteration: $ii"
            return true
        elseif ((solver.opts.constrained && gradient < solver.opts.gradient_tolerance && c_max < solver.opts.constraint_tolerance))
            # @logmsg OuterLoop "--iLQR (inner loop) gradient and constraint eps criteria met at iteration: $ii"
            return true
        end

        # Check for cost convergence
        if ((~solver.opts.constrained && dJ < solver.opts.cost_tolerance) || (solver.opts.constrained && dJ < solver.opts.cost_intermediate_tolerance && iteration != solver.opts.iterations_outerloop))
            # @logmsg OuterLoop "--iLQR (inner loop) cost eps criteria met at iteration: $ii"
            # ~solver.opts.constrained ? @info "Unconstrained solve complete": nothing
            return true
        elseif ((solver.opts.constrained && dJ < solver.opts.cost_tolerance && c_max < solver.opts.constraint_tolerance))
            # @logmsg OuterLoop "--iLQR (inner loop) cost and constraint eps criteria met at iteration: $ii"
            return true
        elseif (solver.opts.constrained && dJ < solver.opts.cost_tolerance && iteration == solver.opts.iterations_outerloop)
            # @logmsg OuterLoop "Terminated on last outerloop. No progress being made"
            return true
        end
    end

    if loop == :outer
        if solver.opts.constrained
            if c_max < solver.opts.constraint_tolerance && (dJ < solver.opts.cost_tolerance || gradient < solver.opts.gradient_tolerance)
                # if solver.opts.verbose
                #     # println("-Outer loop cost and constraint eps criteria met at outer iteration: $j\n")
                #     # println("Constrained solve complete")
                #     if dJ < solver.opts.cost_tolerance
                #         println("--Cost tolerance met")
                #     else
                #         println("--Gradient tolerance met")
                #     end
                # end
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
    println("pre removal: Xf - $(results.X[end])")

    # remove infeasible components
    results_feasible = remove_infeasible_controls_to_unconstrained_results(results,solver)
    println("post removal: Xf - $(results_feasible.X[end])")

    # backward pass - project infeasible trajectory into feasible space using time varying lqr
    Δv = backwardpass!(results_feasible, solver)

    # forward pass
    forwardpass!(results_feasible,solver,Δv)
    println("post fp: Xf - $(results_feasible.X[end])")
    println("post fp: Xf_ - $(results_feasible.X_[end])")

    # update trajectories
    results_feasible.X .= deepcopy(results_feasible.X_)
    results_feasible.U .= deepcopy(results_feasible.U_)

    # return constrained results if input was constrained
    if !solver.opts.unconstrained_original_problem
        # println("pre new constrained results")
        # println("size U: $(size(to_array(results_feasible.U)))")
        # println("size fcu: $(size(to_array(results_feasible.fcu)))")
        # println("size C: $(size(to_array(results.C)))")
        results_feasible = unconstrained_to_constrained_results(results_feasible,solver,results.λ,results.λN)
        # println("post new constrained results")
        # println("size U: $(size(to_array(results_feasible.U)))")
        # println("size fcu: $(size(to_array(results_feasible.fcu)))")
        # println("size C: $(size(to_array(results_feasible.C)))")
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
        -zoh sequential (1st and 2nd order)
        -zoh nonsequential (1st and 2nd order)
        -foh sequential (1st order)
        -foh nonsequential (1st and 2nd order)

        -see Bertsekas 'Constrained Optimization' chapter 2 (p.135)
        -see Toussaint 'A Novel Augmented Lagrangian Approach for Inequalities and Convergent Any-Time Non-Central Updates'
"""
function λ_update!(results::ConstrainedIterResults,solver::Solver,second_order::Bool=false)
    # ZOH
    if solver.control_integration == :zoh
        for k = 1:solver.N
            λ_update_1_zoh!(results,solver,k)
        end
    end

    # FOH
    if solver.control_integration == :foh
        for k = 1:solver.N
            λ_update_1_foh!(results,solver,k)
        end
    end
end

"""@(SIGNATURES) 1st order multiplier update for zoh (sequential)"""
function λ_update_1_zoh!(results::ConstrainedIterResults,solver::Solver,k::Int64)
    p,pI,pE = get_num_constraints(solver)
    if k != solver.N
        results.λ[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λ[k] + results.μ[k].*results.C[k]))
        results.λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])
    else
        results.λN .= max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λN + results.μN.*results.CN))
    end
    return nothing
end


"""@(SIGNATURES) 1st order multiplier update for foh (sequential)"""
function λ_update_1_foh!(results::ConstrainedIterResults,solver::Solver,k::Int64)
    p,pI,pE = get_num_constraints(solver)
    results.λ[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λ[k] + results.μ[k].*results.C[k]))
    results.λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])

    if k == solver.N
        results.λN .= max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λN + results.μN.*results.CN))
    end
    return nothing
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
    if solver.control_integration == :foh
        final_index = solver.N
    else
        final_index = solver.N-1
    end

    for k = 1:final_index
        results.μ[k] = min.(solver.opts.μ_max, solver.opts.γ*results.μ[k])
    end

    results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)

    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('individual')- all penalty terms are updated uniquely according to indiviual improvement compared to previous iteration"""
function μ_update_individual!(results::ConstrainedIterResults,solver::Solver)
    p,pI,pE = get_num_constraints(solver)
    n = solver.model.n

    τ = solver.opts.τ
    μ_max = solver.opts.μ_max
    γ_no  = solver.opts.γ_no
    γ = solver.opts.γ

    if solver.control_integration == :foh
        final_index = solver.N
    else
        final_index = solver.N-1
    end

    # Stage constraints
    for k = 1:final_index
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
function outer_loop_update(results::ConstrainedIterResults,solver::Solver,sqrt_tolerance::Bool=false)::Nothing

    if solver.opts.minimum_time
        n,m,N = get_sizes(solver)
        m̄,mm = get_num_controls(solver)
        p,pI,pE = get_num_constraints(solver)

        val,idx = findmax(to_array(results.C))
        if idx[1] == m̄
            println("max_dt violated!: $val at k=$(idx[2])")
        elseif idx[1] == m̄ + m̄
            println("min_dt violated: $val at k=$(idx[2])")
        elseif idx[1] == p
            println("h_k - h_{k+1} violated: $val at k=$(idx[2])")
        else
            println("other: $val")
        end
    end
    p,pI,pE = get_num_constraints(solver)
    n = solver.model.n

    τ = solver.opts.τ
    μ_max = solver.opts.μ_max
    γ_no  = solver.opts.γ_no
    γ = solver.opts.γ

    solver.control_integration == :foh ? final_index = solver.N : final_index = solver.N-1

    # Stage constraints
    for k = 1:final_index
        for i = 1:p
            if p <= pI
                if max(0.0,results.C[k][i]) <= τ*max(0.0,results.C_prev[k][i])
                    # multiplier update
                    results.λ[k][i] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λ[k][i] + results.μ[k][i]*results.C[k][i]))
                    results.λ[k][i] = max.(0.0,results.λ[k][i])
                    # no penalty update
                    results.μ[k][i] = min(μ_max, γ_no*results.μ[k][i])
                else
                    #penalty update
                    results.μ[k][i] = min(μ_max, γ*results.μ[k][i])
                end
            else
                if abs(results.C[k][i]) <= τ*abs(results.C_prev[k][i])
                    # multiplier update
                    results.λ[k][i] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λ[k][i] + results.μ[k][i]*results.C[k][i]))
                    # no penalty update
                    results.μ[k][i] = min(μ_max, γ_no*results.μ[k][i])
                else
                    # penalty update
                    results.μ[k][i] = min(μ_max, γ*results.μ[k][i])
                end
            end
        end
    end

    # Terminal constraints
    for i = 1:n
        if abs(results.CN[i]) <= τ*abs(results.CN_prev[i])
            results.λN[i] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λN[i] + results.μN[i]*results.CN[i]))
            results.μN[i] = min(μ_max, γ_no*results.μN[i])
        else
            results.μN[i] = min(μ_max, γ*results.μN[i])
        end
    end

    # ## Lagrange multiplier updates
    # λ_update!(results,solver,false)
    #
    # ## Penalty updates
    # μ_update!(results,solver)

    ## Store current constraints evaluations for next outer loop update
    results.C_prev .= deepcopy(results.C)
    results.CN_prev .= deepcopy(results.CN)

    return nothing
end

function outer_loop_update(results::UnconstrainedIterResults,solver::Solver,sqrt_tolerance::Bool)::Nothing
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
