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
function _solve(solver::Solver, U0::Array{Float64,2}, X0::Array{Float64,2}=Array{Float64}(0,0); prevResults::SolverResults=ConstrainedResults())::Tuple{SolverResults,Dict}
    tic()

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

    #****************************#
    #       INITIALIZATION       #
    #****************************#
    if solver.obj isa UnconstrainedObjective
        if solver.opts.verbose
            println("Solving Unconstrained Problem...")
        end
        solver.opts.iterations_outerloop = 1
        results = UnconstrainedResults(n,m,N)
        results.U .= U0
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
        results = ConstrainedResults(n,m,p,N) # preallocate memory for results
        results.MU .*= solver.opts.μ1 # set initial penalty term values

        if infeasible
            results.X .= X0 # initialize state trajectory with infeasible trajectory input
            results.U .= [U0; ui] # augment control with additional control inputs that produce infeasible state trajectory
        else
            results.U .= U0 # initialize control to control input sequence
            # if !isempty(prevResults) # bootstrap previous constraint solution
            #     results.LAMBDA .= prevResults.LAMBDA[1:p,:]
            #     results.MU .= prevResults.MU[1:p,:]
            #     results.λN .= prevResults.λN
            #     results.μN .= prevResults.μN
            # end
        end

        # Diagonal indicies for the Iμ matrix (fast)
        diag_inds = CartesianIndex.(indices(results.Iμ,1),indices(results.Iμ,2))

        # Generate constraint function and jacobian functions from the objective
        update_constraints!(results,solver,results.X,results.U)
    end

    # Unpack results for convenience
    X = results.X # state trajectory
    U = results.U # control trajectory
    X_ = results.X_ # updated state trajectory
    U_ = results.U_ # updated control trajectory


    #****************************#
    #           SOLVER           #
    #****************************#
    # Initial rollout
    if !infeasible
        X[:,1] = solver.obj.x0 # set state trajector initial conditions
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
    k = 1 # Init outer iter counter to increase its scope
    time_setup = toq()
    J_hist = Vector{Float64}()
    c_max_hist = Vector{Float64}()
    tic()

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
    i = 0 # declare outsize for scope
    k = 0
    for k = 1:solver.opts.iterations_outerloop
        if solver.opts.verbose
            println("Outer loop $k (begin)")
        end

        if results isa ConstrainedResults
            update_constraints!(results,solver,results.X,results.U)
            if k == 1
                results.C_prev .= results.C # store initial constraints results for AL method outer loop update, after this first update C_prev gets updated in the outer loop update
                results.CN_prev .= results.CN
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
            if solver.opts.verbose
                println("--Iteration: $k-($i)--")
            end

            if solver.opts.cache
                t1 = time_ns() # time flag for iLQR inner loop start
            end

            ### BACKWARD PASS ###
            calc_jacobians(results, solver)
            if solver.control_integration == :foh
                v1, v2, grad = backwardpass_foh!(results,solver) #TODO combine with square root
            elseif solver.opts.square_root
                v1, v2, grad = backwardpass_sqrt!(results, solver) #TODO option to help avoid ill-conditioning [see algorithm xx]
            else
                v1, v2, grad = backwardpass!(results, solver)
            end

            ### FORWARDS PASS ###
            J = forwardpass!(results, solver, v1, v2)
            push!(J_hist,J)

            if solver.opts.cache
                t2 = time_ns() # time flag of iLQR inner loop end
            end

            ### UPDATE RESULTS ###
            X .= X_
            U .= U_
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

            # Check for cost convergence
            if (~is_constrained && (dJ < solver.opts.eps || grad < solver.opts.gradient_tolerance)) || (results isa ConstrainedResults && (dJ < solver.opts.eps_intermediate || grad < solver.opts.gradient_tolerance) && k != solver.opts.iterations_outerloop)
                if solver.opts.verbose
                    println("--iLQR (inner loop) cost eps criteria met at iteration: $i\n")
                    if results isa UnconstrainedResults
                        println("Unconstrained solve complete")
                    end
                end
                break
            elseif (is_constrained && (dJ < solver.opts.eps || grad < solver.opts.gradient_tolerance) && c_max < solver.opts.eps_constraint)
                if solver.opts.verbose
                    println("--iLQR (inner loop) cost and constraint eps criteria met at iteration: $i\n")
                end
                break
            end
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
        # Check if maximum constraint violation satisfies termination criteria
        if solver.obj isa ConstrainedObjective
            max_c = max_violation(results, diag_inds)
            if max_c < solver.opts.eps_constraint && dJ < solver.opts.eps
                if solver.opts.verbose
                    println("-Outer loop cost and constraint eps criteria met at outer iteration: $k\n")
                    println("Constrained solve complete")
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

    # Run Stats
    stats = Dict("iterations"=>iter-1,
                 "major iterations"=>k,
                 "runtime"=>toq(),
                 "setup_time"=>time_setup,
                 "cost"=>J_hist,
                 "c_max"=>c_max_hist)

    if ((k == solver.opts.iterations_outerloop) && (i == solver.opts.iterations)) && solver.opts.verbose
        println("*Solve reached max iterations*")
    end
    ## Return dynamically feasible trajectory
    if infeasible && solver.opts.solve_feasible
        if solver.opts.verbose
            println("Infeasible -> Feasible ")
        end
        results_feasible, stats_feasible = solve_feasible_traj(results,solver) # using current control solution, warm-start another solve with dynamics strictly enforced
        if solver.opts.cache
            results_feasible = merge_results_cache(results_cache,results_feasible) # return infeasible results and final enforce dynamics results
        end
        for key in keys(stats_feasible)
            stats[key * " (infeasible)"] = stats[key]
        end
        stats["iterations"] += stats_feasible["iterations"]-1
        stats["major iterations"] += stats_feasible["iterations"]
        stats["runtime"] += stats_feasible["runtime"]
        stats["setup_time"] += stats_feasible["setup_time"]
        append!(stats["cost"], stats_feasible["cost"])
        append!(stats["c_max"], stats_feasible["c_max"])
        return results_feasible, stats # TODO: add stats together
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
Infeasible start solution is run through standard constrained solve to enforce dynamic feasibility. All infeasible augmented controls are removed.
"""
function solve_feasible_traj(results::ConstrainedResults,solver::Solver)
    solver.opts.infeasible = false
    if solver.opts.unconstrained
        # TODO method for generating a new solver with unconstrained objective
        obj_uncon = UnconstrainedObjective(solver.obj.Q,solver.obj.R,solver.obj.Qf,solver.obj.tf,solver.obj.x0,solver.obj.xf)
        solver_uncon = Solver(solver.model,obj_uncon,integration=solver.integration,dt=solver.dt,opts=solver.opts)
        return solve(solver_uncon,results.U[1:solver.model.m,:])
    else
        return solve(solver,results.U[1:solver.model.m,:],prevResults=results)
    end
end



"""
$(SIGNATURES)

Updates penalty (μ) and Lagrange multiplier (λ) parameters for Augmented Lagrange Method. λ is updated for equality and inequality constraints according to [insert equation ref] and μ is incremented by a constant term for all constraint types.
"""
function outer_loop_update(results::ConstrainedResults,solver::Solver)::Void
    p,N = size(results.C)
    pI = solver.obj.pI

    if solver.control_integration == :foh
        final_index = N
    else
        final_index = N-1
    end

    results.V_al_prev .= results.V_al_current

    ## Lagrange multiplier update for time step
    for jj = 1:final_index
        for ii = 1:p
            if ii <= pI
                results.V_al_current[ii,jj] .= min(-1.0*results.C[ii,jj], results.LAMBDA[ii,jj]/results.MU[ii,jj])

                results.LAMBDA[ii,jj] .= max(0.0, results.LAMBDA[ii,jj] + results.MU[ii,jj]*results.C[ii,jj]) # λ_min < λ < λ_max
                # results.LAMBDA[ii,jj] .= max(solver.opts.λ_min, min(solver.opts.λ_max, max(0.0, results.LAMBDA[ii,jj] + results.MU[ii,jj]*results.C[ii,jj]))) # λ_min < λ < λ_max
            else
                results.LAMBDA[ii,jj] .= results.LAMBDA[ii,jj] + results.MU[ii,jj]*results.C[ii,jj] # λ_min < λ < λ_max

                # results.LAMBDA[ii,jj] .= max(solver.opts.λ_min, min(solver.opts.λ_max, results.LAMBDA[ii,jj] + results.MU[ii,jj]*results.C[ii,jj])) # λ_min < λ < λ_max
            end

            # # Penalty update (individual)
            # if solver.opts.outer_loop_update == :individual
            #     if max(norm(results.C[ii,jj]),norm(results.V_al_current[ii,jj])) <= solver.opts.τ*max(norm(results.C_prev[ii,jj]),norm(results.V_al_prev[ii,jj]))
            #         results.MU[ii,jj] .= min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[ii,jj])
            #     else
            #         results.MU[ii,jj] .= min.(solver.opts.μ_max, solver.opts.γ*results.MU[ii,jj])
            #     end
            # end

        end

        # # Penalty update (time step)
        # if solver.opts.outer_loop_update == :uniform_time_step
        #     if max(norm(results.C[pI+1:p,jj]),norm(results.V_al_current[pI+1:p,jj])) <= solver.opts.τ*max(norm(results.C_prev[pI+1:p,jj]),norm(results.V_al_prev[pI+1:p,jj]))
        #         results.MU[:,jj] .= min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[:,jj])
        #     else
        #         results.MU[:,jj] .= min.(solver.opts.μ_max, solver.opts.γ*results.MU[:,jj])
        #     end
        # end

    end
    # Lagrange multiplier terminal state equality constraints update
    for ii = 1:solver.model.n
        results.λN[ii] .= results.λN[ii] + results.μN[ii].*results.CN[ii]
        # results.λN[ii] .= max(solver.opts.λ_min, min(solver.opts.λ_max, results.λN[ii] + results.μN[ii].*results.CN[ii]))
    end

    ## Penalty update
    #---Default---
    if solver.opts.outer_loop_update == :default # update all μ default
        results.MU .= solver.opts.γ*results.MU
        results.μN .= solver.opts.γ*results.μN
        # results.MU .= min.(solver.opts.μ_max, solver.opts.γ*results.MU)
        # results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)
    end
    #-------------

    # if solver.opts.outer_loop_update == :uniform_time_step
    #     if norm(results.CN) <= solver.opts.τ*norm(results.CN_prev)
    #         results.μN .= min.(solver.opts.μ_max, solver.opts.γ_no*results.μN)
    #     else
    #         results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)
    #     end
    # end

    if solver.opts.outer_loop_update == :uniform
        if max(norm([results.C[pI+1:p,:][:]; results.CN]),norm(results.V_al_current[pI+1:p,:][:])) <= solver.opts.τ*max(norm([results.C_prev[pI+1:p,:][:];results.CN_prev]),norm(results.V_al_prev[pI+1:p,:][:]))
            results.MU .= solver.opts.γ_no*results.MU
            results.μN .= solver.opts.γ_no*results.μN
            # results.MU .= min.(solver.opts.μ_max, solver.opts.γ_no*results.MU)
            # results.μN .= min.(solver.opts.μ_max, solver.opts.γ_no*results.μN)
            if solver.opts.verbose
                println("no μ update\n")
            end
        else
            results.MU .= solver.opts.γ*results.MU
            results.μN .= solver.opts.γ*results.μN
            # results.MU .= min.(solver.opts.μ_max, solver.opts.γ*results.MU)
            # results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)
            if solver.opts.verbose
                println("$(solver.opts.γ)x μ update\n")
            end
        end
    end

    ## Store current constraints evaluations for next outer loop update
    results.C_prev .= results.C
    results.CN_prev .= results.CN

    #TODO properly cache V, mu, lambda

    return nothing
end

function outer_loop_update(results::UnconstrainedResults,solver::Solver)::Void
    return nothing
end
