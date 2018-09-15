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
function solve(solver::Solver, X0::Array{Float64,2}, U0::Array{Float64,2}; prevResults::SolverResults=ConstrainedResults())::Tuple{SolverResults,Dict}

    # If initialize zero controls if none are passed in
    if isempty(U0)
        U0 = zeros(solver.m,solver.N-1)
    end

    # Convert to a constrained problem
    if isa(solver.obj, UnconstrainedObjective)
        if solver.opts.solve_feasible == false
            solver.opts.infeasible = true
        end

        obj_c = ConstrainedObjective(solver.obj)
        solver.opts.unconstrained = true
        solver = Solver(solver.model, obj_c, integration=solver.integration, dt=solver.dt, opts=solver.opts)
    end



    results, stats = _solve(solver,U0,X0,prevResults=prevResults)
    return results, stats
end

function solve(solver::Solver,U0::Array{Float64,2}; prevResults::SolverResults=ConstrainedResults())::Tuple{SolverResults,Dict}
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
function _solve(solver::Solver, U0::Array{Float64,2}, X0::Array{Float64,2}=Array{Float64}(undef,0,0); prevResults::SolverResults=ConstrainedResults())::Tuple{SolverResults,Dict}
    t_start = time_ns()

    ## Unpack model, objective, and solver parameters
    N = solver.N # number of iterations for the solver (ie, knotpoints)
    n = solver.model.n # number of states
    m = solver.model.m # number of control inputs

    # Use infeasible start if an initial trajectory was passed in
    if isempty(X0)
        infeasible = false
    else
        infeasible = true
    end

    use_static = false#solver.opts.use_static

    #****************************#
    #       INITIALIZATION       #
    #****************************#
    if solver.obj isa UnconstrainedObjective
        if solver.opts.verbose
            println("Solving Unconstrained Problem...")
        end
        iterations_outerloop_original = solver.opts.iterations_outerloop
        solver.opts.iterations_outerloop = 1
        if use_static
            results = UnconstrainedResultsStatic(n,m,N)
        else
            results = UnconstrainedResults(n,m,N)
        end
        copyto!(results.U, U0)
        is_constrained = false

    elseif solver.obj isa ConstrainedObjective
        p = solver.obj.p # number of inequality and equality constraints
        pI = solver.obj.pI # number of inequality constraints
        is_constrained = true

        if infeasible
            if solver.opts.verbose
                println("Solving Constrained Problem with Infeasible Start...")
            end
            ui = infeasible_controls(solver,X0,U0) # generates n additional control input sequences that produce the desired infeasible state trajectory
            m += n # augment the number of control input sequences by the number of states
            p += n # increase the number of constraints by the number of additional control input sequences
            solver.opts.infeasible = true
        else
            if solver.opts.verbose
                println("Solving Constrained Problem...")
            end
            solver.opts.infeasible = false
        end

        ## Initialize results
        if use_static
            results = ConstrainedResultsStatic(n,m,p,N)
        else
            results = ConstrainedResults(n,m,p,N) # preallocate memory for results
        end
        results.MU .*= solver.opts.μ1 # set initial penalty term values

        if infeasible
            copyto!(results.X, X0)  # initialize state trajectory with infeasible trajectory input
            copyto!(results.U, [U0; ui])  # augment control with additional control inputs that produce infeasible state trajectory
        else
            copyto!(results.U, U0) # initialize control to control input sequence
        end

        # Diagonal indicies for the Iμ matrix (fast)
        diag_inds = CartesianIndex.(axes(results.Iμ,1),axes(results.Iμ,2))

        # Generate constraint function and jacobian functions from the objective
        update_constraints!(results,solver,results.X,results.U)
    end

    # Unpack results for convenience
    X = results.X # state trajectory
    U = results.U # control trajectory
    X_ = results.X_ # updated state trajectory
    U_ = results.U_ # updated control trajectory

    # Set initial regularization
    results.ρ[1] *= solver.opts.ρ_initial

    #****************************#
    #           SOLVER           #
    #****************************#
    # Initial rollout
    if !infeasible
        if use_static
            X[1] = solver.obj.x0
        else
            X[:,1] = solver.obj.x0 # set state trajector initial conditions
        end
        flag = rollout!(results,solver) # rollout new state trajectoy

        if !flag
            if solver.opts.verbose
                println("Bad initial control sequence, setting initial control to random")
            end
            results.U .= rand(solver.model.m,solver.N)
            rollout!(results,solver)
        end
    end

    # Solver Statistics
    iter = 1 # counter for total number of iLQR iterations
    iter_outer = 1
    iter_inner = 1
    time_setup = time_ns() - t_start
    J_hist = Vector{Float64}()
    c_max_hist = Vector{Float64}()
    t_solve_start = time_ns()

    if solver.opts.cache
        # Initialize cache and store initial trajectories and cost
        results_cache = ResultsCache(solver,solver.opts.iterations*solver.opts.iterations_outerloop+1) #TODO preallocate smaller arrays
        add_iter!(results_cache, results, cost(solver, results, X, U))
    end
    iter += 1


    #****************************#
    #         OUTER LOOP         #
    #****************************#

    dJ = Inf
    gradient = Inf
    Δv = Inf

    for k = 1:solver.opts.iterations_outerloop
        iter_outer = k
        if solver.opts.verbose
            println("Outer loop $k (begin)")
        end

        if results isa ConstrainedResults || results isa ConstrainedResultsStatic
            update_constraints!(results,solver,results.X,results.U)
            if k == 1
                results.C_prev .= deepcopy(results.C)
                results.CN_prev .= deepcopy(results.CN)
                # results.C_prev .= results.C # store initial constraints results for AL method outer loop update, after this first update C_prev gets updated in the outer loop update
                # results.CN_prev .= results.CN
            end
        end
        J_prev = cost(solver, results, X, U)
        k == 1 ? push!(J_hist, J_prev) : nothing  # store the first cost

        if solver.opts.verbose
            println("Cost ($k): $J_prev\n")
        end

        #****************************#
        #         INNER LOOP         #
        #****************************#

        for i = 1:solver.opts.iterations
            iter_inner = i
            if solver.opts.verbose
                println("--Iteration: $k-($i)--")
            end

            if solver.opts.cache
                t1 = time_ns() # time flag for iLQR inner loop start
            end

            ### BACKWARD PASS ###
            calc_jacobians(results, solver)
            if solver.control_integration == :foh
                Δv = backwardpass_foh!(results,solver) #TODO combine with square root
            elseif solver.opts.square_root
                Δv = backwardpass_sqrt!(results, solver) #TODO option to help avoid ill-conditioning [see algorithm xx]
            else
                Δv = backwardpass!(results, solver)
            end

            ### FORWARDS PASS ###
            J = forwardpass!(results, solver, Δv)
            push!(J_hist,J)

            if solver.opts.cache
                t2 = time_ns() # time flag of iLQR inner loop end
            end

            ### UPDATE RESULTS ###
            if use_static
                X .= deepcopy(X_)
                U .= deepcopy(U_)
            else
                X .= X_
                U .= U_
            end
            dJ = copy(abs(J-J_prev)) # change in cost
            J_prev = copy(J)

            if solver.opts.cache
                # Store current results and performance parameters
                time = (t2-t1)/(1.0e9)
                add_iter!(results_cache, results, J, time, iter)
            end
            iter += 1

            if is_constrained
                c_max = max_violation(results,diag_inds)
                push!(c_max_hist, c_max)
            end

            ## Check gradients for convergence ##
            if use_static
                d_grad = maximum(map((x)->maximum(abs.(x)),results.d))
                s_grad = maximum(abs.(results.s[1]))
                todorov_grad = mean(map((x)->maximum(x), map((x,y)-> x./y, map((x)->abs.(x),results.d),map((x)->abs.(x),results.u .+ 1.0))))
            else
                d_grad = maximum(abs.(results.d[:]))
                s_grad = maximum(abs.(results.s[:,1]))
                todorov_grad = mean(maximum(abs.(results.d)./(abs.(results.U) .+ 1),dims=1))
            end
            if solver.opts.verbose
                println("d gradient: $d_grad")
                println("s gradient: $s_grad")
                println("todorov gradient $(todorov_grad)")
            end
            gradient = todorov_grad

            if (~is_constrained && gradient < solver.opts.gradient_tolerance) || (results isa ConstrainedResults && gradient < solver.opts.gradient_intermediate_tolerance && k != solver.opts.iterations_outerloop)
                if solver.opts.verbose
                    println("--iLQR (inner loop) cost eps criteria met at iteration: $i\n")
                    if results isa UnconstrainedResults
                        println("Unconstrained solve complete")
                    end
                    println("---Gradient tolerance met")
                end
                break
            # Check for gradient and constraint tolerance convergence
        elseif (is_constrained && gradient < solver.opts.gradient_tolerance  && c_max < solver.opts.constraint_tolerance)
                if solver.opts.verbose
                    println("--iLQR (inner loop) cost and constraint eps criteria met at iteration: $i")
                    println("---Gradient tolerance met\n")
                end
                break
            end
            #####################

            ## Check for cost convergence ##
            if (~is_constrained && dJ < solver.opts.cost_tolerance) || (results isa ConstrainedResults && dJ < solver.opts.cost_intermediate_tolerance && k != solver.opts.iterations_outerloop)
                if solver.opts.verbose
                    println("--iLQR (inner loop) cost eps criteria met at iteration: $i\n")
                    if ~is_constrained
                        println("Unconstrained solve complete")
                    end
                    println("---Cost met tolerance")
                end
                break
            # Check for cost and constraint tolerance convergence
        elseif (is_constrained && dJ < solver.opts.cost_tolerance  && c_max < solver.opts.constraint_tolerance)
                if solver.opts.verbose
                    println("--iLQR (inner loop) cost and constraint eps criteria met at iteration: $i")
                    println("---Cost met tolerance\n")
                end
                break
            # Check for maxed regularization
            elseif results.ρ[1] > solver.opts.ρ_max
                if solver.opts.verbose
                    println("*Regularization maxed out*\n - terminating solve - ")
                end
                break
            end
            ################################

        end
        ### END INNER LOOP ###

        if solver.opts.cache
            results_cache.iter_type[iter-1] = 1 # flag outerloop update
        end

        #****************************#
        #      OUTER LOOP UPDATE     #
        #****************************#
        outer_loop_update(results,solver)

        if solver.opts.cache
            # Store current results and performance parameters
            add_iter_outerloop!(results_cache, results, iter-1) # we already iterated counter but this needs to update those results
        end

        #****************************#
        #    TERMINATION CRITERIA    #
        #****************************#
        # Check if maximum constraint violation satisfies termination criteria AND cost or gradient tolerance convergence
        if solver.obj isa ConstrainedObjective
            max_c = max_violation(results, diag_inds)
            if max_c < solver.opts.constraint_tolerance && (dJ < solver.opts.cost_tolerance || gradient < solver.opts.gradient_tolerance)
                if solver.opts.verbose
                    println("-Outer loop cost and constraint eps criteria met at outer iteration: $k\n")
                    println("Constrained solve complete")
                    if dJ < solver.opts.cost_tolerance
                        println("--Cost tolerance met")
                    else
                        println("--Gradient tolerance met")
                    end
                end
                break
            end
        end
        if solver.opts.verbose
            println("Outer loop $k (end)\n -----")
        end

    end
    ### END OUTER LOOP ###

    if solver.opts.cache
        # Store final results
        results_cache.termination_index = iter-1
        results_cache.X = results.X
        results_cache.U = results.U
    end


    if is_constrained
        # use_static ? results = ConstrainedResults(results) : nothing
    else
        # use_static ? results = UnconstrainedResults(results) : nothing
        solver.opts.iterations_outerloop = iterations_outerloop_original
    end

    # Run Stats
    stats = Dict("iterations"=>iter-1,
                 "major iterations"=>iter_outer,
                 "runtime"=>float(time_ns() - t_solve_start)/1e9,
                 "setup_time"=>float(time_setup)/1e9,
                 "cost"=>J_hist,
                 "c_max"=>c_max_hist)

    if ((iter_outer == solver.opts.iterations_outerloop) && (iter_inner == solver.opts.iterations)) && solver.opts.verbose
        println("*Solve reached max iterations*")
    end
    ## Return dynamically feasible trajectory
    if infeasible && solver.opts.solve_feasible
        if solver.opts.verbose
            # println("Infeasible -> Feasible ")
            println("Infeasible solve complete")
        end

        # run single backward pass/forward pass to get dynamically feasible solution
        results_feasible = get_feasible_trajectory(results,solver)

        # resolve feasible solution if necessary (should be fast)
        if solver.opts.resolve_feasible
            if solver.opts.verbose
                println("Resolving feasible")
            end
            # create unconstrained solver from infeasible solver if problem is unconstrained
            if solver.opts.unconstrained
                obj_uncon = UnconstrainedObjective(solver.obj.Q,solver.obj.R,solver.obj.Qf,solver.obj.tf,solver.obj.x0,solver.obj.xf)
                solver = Solver(solver.model,obj_uncon,integration=solver.integration,dt=solver.dt,opts=solver.opts)
            end

            # resolve feasible problem with warm start
            results_feasible, stats_feasible = solve(solver,results_feasible.U)

            # merge stats
            for key in keys(stats_feasible)
                stats[key * " (infeasible)"] = stats[key]
            end
            stats["iterations"] += stats_feasible["iterations"]-1
            stats["major iterations"] += stats_feasible["iterations"]
            stats["runtime"] += stats_feasible["runtime"]
            stats["setup_time"] += stats_feasible["setup_time"]
            append!(stats["cost"], stats_feasible["cost"])
            append!(stats["c_max"], stats_feasible["c_max"])
        end

        # return (now) feasible results
        if solver.opts.cache
            if solver.opts.resolve_feasible
                results_cache = merge_results_cache(results_cache,results_feasible)
            else
                add_iter!(results_cache, results_feasible, cost(solver, results_feasible, results_feasible.X, results_feasible.U))
            end
            return results_cache, stats
        else
            return results_feasible, stats
        end

    # if feasible solve, return results
    else
        if solver.opts.verbose
            println("***Solve Complete***")
        end
        if solver.opts.cache
            return results_cache, stats
        else
            return results, stats
        end
    end
end

"""
$(SIGNATURES)
Infeasible start solution is run through time varying LQR to track state and control trajectories
"""
function get_feasible_trajectory(results::SolverIterResults,solver::Solver)::SolverIterResults
    # turn off infeasible solve
    solver.opts.infeasible = false

    # remove infeasible components
    results_feasible = new_unconstrained_results(results,solver)

    # before backward pass (ie, time varying lqr)
    if solver.control_integration == :foh
        Δv = backwardpass_foh!(results_feasible,solver)
    elseif solver.opts.square_root
        Δv = backwardpass_sqrt!(results_feasible, solver)
    else
        Δv = backwardpass!(results_feasible, solver)
    end
    # rollout solution
    forwardpass!(results_feasible,solver,Δv)
    results_feasible.X .= results_feasible.X_
    results_feasible.U .= results_feasible.U_

    # return constrained results if input was constrained
    if !solver.opts.unconstrained
        results_feasible = new_constrained_results(results_feasible,solver)
        update_constraints!(results_feasible,solver,results_feasible.X,results_feasible.U)
    end

    return results_feasible
end



"""
$(SIGNATURES)
    Updates penalty (μ) and Lagrange multiplier (λ) parameters for Augmented Lagrange Method. λ is updated for equality and inequality constraints according to [insert equation ref] and μ is incremented by a constant term for all constraint types.
    -ALGENCAN 'uniform' update: see 'Practical Augmented Lagrangian Methods for Constrained Optimization' (Algorithm 4.1, p.33)
    -'individual' update: see Bertsekas Constrained Optimization (eq. 47, p.123)
"""
function outer_loop_update(results::ConstrainedResults,solver::Solver)::Nothing
    n = solver.model.n # number of terminal constraints (equal to number of states)
    p, N = size(results.C) # number of constraints and problem horizon (ie knot points)
    pI = solver.obj.pI # number of inequality constraints

    if solver.control_integration == :foh
        final_index = N
    else
        final_index = N-1
    end

    # store previous term for penalty update
    if solver.opts.outer_loop_update == :uniform
        results.V_al_prev .= results.V_al_current
    end

    ### Lagrange multiplier updates ###
    for jj = 1:final_index
        for ii = 1:p
            # inequality constraints
            if ii <= pI
                # calculate term for penalty update (see ALGENCAN ref.)
                if solver.opts.outer_loop_update == :uniform
                    results.V_al_current[ii,jj] = min(-1.0*results.C[ii,jj], results.LAMBDA[ii,jj]/results.MU[ii,jj])
                end

                # Lagrange multiplier update (1st order)
                results.LAMBDA[ii,jj] = max(solver.opts.λ_min, min(solver.opts.λ_max, max(0.0, results.LAMBDA[ii,jj] + results.MU[ii,jj]*results.C[ii,jj]))) # λ_min < λ < λ_max

                # penalty update for 'individual' scheme
                if  solver.opts.outer_loop_update == :individual
                    if max(0.0,results.C[ii,jj]) <= solver.opts.τ*max(0.0,results.C_prev[ii,jj])
                        results.MU[ii,jj] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[ii,jj])
                    else
                        results.MU[ii,jj] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[ii,jj])
                    end
                end

            # equality constraints
            else
                # Lagrange multiplier update (1st order)
                results.LAMBDA[ii,jj] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.LAMBDA[ii,jj] + results.MU[ii,jj]*results.C[ii,jj])) # λ_min < λ < λ_max

                # penalty update for 'individual' scheme
                if  solver.opts.outer_loop_update == :individual
                    if abs(results.C[ii,jj]) <= solver.opts.τ*abs(results.C_prev[ii,jj])
                        results.MU[ii,jj] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[ii,jj])
                    else
                        results.MU[ii,jj] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[ii,jj])
                    end
                end
            end
        end
    end

    # Lagrange multiplier update for terminal state equality constraints
    for ii = 1:solver.model.n
        results.λN[ii] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.λN[ii] + results.μN[ii].*results.CN[ii]))
    end
    ###################################

    ### Penalty updates ###
    # 'default' penaltiy update - all penalty terms are updated (no conditions)
    if solver.opts.outer_loop_update == :default
        results.MU .= min.(solver.opts.μ_max, solver.opts.γ*results.MU)
        results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)
    end

    # 'uniform' penalty update - see ALGENCAN reference
    if solver.opts.outer_loop_update == :uniform
        v1 = max(norm([results.C[pI+1:p,:][:]; results.CN]),norm(results.V_al_current))
        v2 = max(norm([results.C_prev[pI+1:p,:][:];results.CN_prev]),norm(results.V_al_prev))

        if v1 <= solver.opts.τ*v2
            results.MU .= min.(solver.opts.μ_max, solver.opts.γ_no*results.MU)
            results.μN .= min.(solver.opts.μ_max, solver.opts.γ_no*results.μN)
            if solver.opts.verbose
                println("no μ update\n")
            end
        else
            results.MU .= min.(solver.opts.μ_max, solver.opts.γ*results.MU)
            results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)
            if solver.opts.verbose
                println("$(solver.opts.γ)x μ update\n")
            end
        end
    end

    # 'individual' penalty update (only terminal constraints left to update)
    if solver.opts.outer_loop_update == :individual
        for ii = 1:n
            if abs(results.CN[ii]) <= solver.opts.τ*abs(results.CN_prev[ii])
                results.μN[ii] = min.(solver.opts.μ_max, solver.opts.γ_no*results.μN[ii])
            else
                results.μN[ii] = min.(solver.opts.μ_max, solver.opts.γ*results.μN[ii])
            end
        end
    end
    #######################

    ## Store current constraints evaluations for next outer loop update
    results.C_prev .= results.C
    results.CN_prev .= results.CN

    return nothing
end

function outer_loop_update(results::ConstrainedResultsStatic,solver::Solver)::Nothing
    N = solver.N
    p = length(results.C[1])
    pI = solver.obj.pI

    if solver.control_integration == :foh
        final_index = N
    else
        final_index = N-1
    end

    # store previous term for penalty update
    if solver.opts.outer_loop_update == :uniform
        results.V_al_prev .= deepcopy(results.V_al_current)
    end

    ### Lagrange multiplier updates ###
    for jj = 1:final_index
        for ii = 1:p
            ## inequality constraints
            if ii <= pI
                # calculate term for penalty update (see ALGENCAN ref.)
                if solver.opts.outer_loop_update == :uniform
                    results.V_al_current[ii,jj] = min(-1.0*results.C[jj][ii], results.LAMBDA[jj][ii]/results.MU[jj][ii])
                end

                # Lagrange multiplier update (1st order)
                results.LAMBDA[jj][ii] = max(solver.opts.λ_min, min(solver.opts.λ_max, max(0.0, results.LAMBDA[jj][ii] + results.MU[jj][ii]*results.C[jj][ii]))) # λ_min < λ < λ_max

                # penalty update for 'individual' scheme
                if  solver.opts.outer_loop_update == :individual
                    if max(0.0,results.C[ii][jj]) <= solver.opts.τ*max(0.0,results.C_prev[ii][jj])
                        results.MU[ii][jj] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[ii][jj])
                    else
                        results.MU[ii][jj] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[ii][jj])
                    end
                end

            # equality constraints
            else
                # Lagrange multiplier update (1st order)
                results.LAMBDA[jj][ii] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.LAMBDA[jj][ii] + results.MU[jj][ii]*results.C[jj][ii])) # λ_min < λ < λ_max

                # penalty update for 'individual' scheme
                if  solver.opts.outer_loop_update == :individual
                    if abs(results.C[ii][jj]) <= solver.opts.τ*abs(results.C_prev[ii][jj])
                        results.MU[ii][jj] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[ii][jj])
                    else
                        results.MU[ii][jj] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[ii][jj])
                    end
                end
            end
        end
    end

    # Lagrange multiplier update for terminal state equality constraints
    for ii = 1:solver.model.n
        results.λN[ii] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.λN[ii] + results.μN[ii].*results.CN[ii]))
    end
    ##############################

    ### Penalty updates ###
    # 'default' penalty update - update all penalty terms
    if solver.opts.outer_loop_update == :default
        results.MU .= min.(solver.opts.μ_max, solver.opts.γ.*results.MU)
        results.μN .= min.(solver.opts.μ_max, solver.opts.γ.*results.μN)
    end

    # 'uniform' penalty update (see ALGENCAN reference)
    if solver.opts.outer_loop_update == :uniform
        v1 = max(sqrt(norm2(results.C,pI+1:p) + norm2(results.CN)), norm(results.V_al_current))
        v2 = max(sqrt(norm2(results.C_prev,pI+1:p) + norm2(results.CN_prev)), norm(results.V_al_prev))

        if v1 <= solver.opts.τ*v2
            for jj = 1:N
                results.MU[jj] = min.(solver.opts.μ_max, solver.opts.γ_no.*results.MU[jj])
            end
            results.μN = min.(solver.opts.μ_max, solver.opts.γ_no.*results.μN)
            if solver.opts.verbose
                println("no μ update\n")
            end
        else
            for jj = 1:N
                results.MU[jj] = min.(solver.opts.μ_max, solver.opts.γ.*results.MU[jj])
            end
            results.μN .= min.(solver.opts.μ_max, solver.opts.γ.*results.μN)
            if solver.opts.verbose
                println("$(solver.opts.γ)x μ update\n")
            end
        end
    end

    # 'individual' penalty update (only terminal constraints left to update)
    if solver.opts.outer_loop_update == :individual
        for ii = 1:n
            if abs(results.CN[ii]) <= solver.opts.τ*abs(results.CN_prev[ii])
                results.μN[ii] = min.(solver.opts.μ_max, solver.opts.γ_no*results.μN[ii])
            else
                results.μN[ii] = min.(solver.opts.μ_max, solver.opts.γ*results.μN[ii])
            end
        end
    end
    #######################

    ## Store current constraints evaluations for next outer loop update
    results.C_prev .= deepcopy(results.C)
    results.CN_prev .= deepcopy(results.CN)

    return nothing
end

function update_criteria(results::ConstrainedResultsStatic)
    max(sqrt(norm2(results.C,pI+1:p) + norm2(results.CN)), norm(results.V_al_current)), max(sqrt(norm2(results.C_prev,pI+1:p) + norm2(results.CN_prev)), norm(results.V_al_prev))
end

function update_criteria(results::ConstrainedResults)
    max(norm([results.C[pI+1:p,:][:]; results.CN]),norm(results.V_al_current[:])), max(norm([results.C_prev[pI+1:p,:][:];results.CN_prev]),norm(results.V_al_prev[:]))
end

function outer_loop_update(results::UnconstrainedResults,solver::Solver)::Nothing
    return nothing
end

function outer_loop_update(results::UnconstrainedResultsStatic,solver::Solver)::Nothing
    return nothing
end
