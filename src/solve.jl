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
function solve(solver::Solver, X0::Array{Float64,2}, U0::Array{Float64,2}; prevResults::SolverResults=ConstrainedResults())::SolverResults
    solver.opts.infeasible = true

    # If initialize zero controls if none are passed in
    if isempty(U0)
        U0 = zeros(solver.m,solver.N-1)
    end

    # Convert to a constrained problem
    if isa(solver.obj, UnconstrainedObjective)
        obj_c = ConstrainedObjective(solver.obj)
        solver.opts.unconstrained = true
        solver = Solver(solver.model, obj_c, integration=solver.integration, dt=solver.dt, opts=solver.opts)
    end

    _solve(solver,U0,X0,prevResults=prevResults)
end

function solve(solver::Solver,U0::Array{Float64,2}; prevResults::SolverResults=ConstrainedResults())::SolverResults
    _solve(solver,U0,prevResults=prevResults)
end

function solve(solver::Solver)::SolverResults
    # Generate random control sequence
    U0 = rand(solver.model.m, solver.N)
    solve(solver,U0)
end


"""
$(SIGNATURES)

Solve constrained optimization problem specified by `solver`
"""
function _solve(solver::Solver, U0::Array{Float64,2}, X0::Array{Float64,2}=Array{Float64}(0,0); prevResults::SolverResults=ConstrainedResults())::SolverResults
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

    # Initialization
    if solver.obj isa UnconstrainedObjective
        print_debug("Solving Unconstrained Problem...")
        solver.opts.iterations_outerloop = 1
        results = UnconstrainedResults(n,m,N)
        results.U .= U0

    elseif solver.obj isa ConstrainedObjective
        p = solver.obj.p # number of inequality and equality constraints
        pI = solver.obj.pI # number of inequality constraints

        if infeasible
            println("Solving Constrained Problem with Infeasible Start...")
            ui = infeasible_controls(solver,X0,U0) # generates n additional control input sequences that produce the desired infeasible state trajectory
            m += n # augment the number of control input sequences by the number of states
            p += n # increase the number of constraints by the number of additional control input sequences
            solver.opts.infeasible = true
        else
            println("Solving Constrained Problem...")
            solver.opts.infeasible = false
        end

        ## Initialize results
        results = ConstrainedResults(n,m,p,N) # preallocate memory for results

        if infeasible
            results.X .= X0 # initialize state trajectory with infeasible trajectory input
            results.U .= [U0; ui] # augment control with additional control inputs that produce infeasible state trajectory
        else
            results.U .= U0 # initialize control to control input sequence
            if !isempty(prevResults) # bootstrap previous constraint solution
                results.LAMBDA .= prevResults.LAMBDA[1:p,:]
                results.MU .= prevResults.MU[1:p,:]
                results.λN .= prevResults.λN
                results.μN .= prevResults.μN
            end
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

    ## Solver
    # Initial rollout
    if !infeasible
        X[:,1] = solver.obj.x0 # set state trajector initial conditions
        flag = rollout!(results,solver) # rollout new state trajectoy

        if !flag
            println("Bad initial control sequence, setting initial control to random")
            results.U .= rand(solver.model.m,solver.N)
            rollout!(results,solver)
        end
    end

    if solver.opts.cache
        # Initialize cache and store initial trajectories and cost
        iter = 1 # counter for total number of iLQR iterations
        results_cache = ResultsCache(solver,solver.opts.iterations*solver.opts.iterations_outerloop+1) #TODO preallocate smaller arrays
        add_iter!(results_cache, results, cost(solver, X, U))
        iter += 1
    end

    # Outer Loop
    dJ = Inf
    for k = 1:solver.opts.iterations_outerloop
        println("Outer loop $k (begin)")

        if results isa ConstrainedResults
            update_constraints!(results,solver,results.X,results.U)
        end
        J_prev = cost(solver, results, X, U)
        println("Cost ($k): $J_prev\n")

        for i = 1:solver.opts.iterations
            println("--Iteration: $k-($i)--")

            if solver.opts.cache
                t1 = time_ns() # time flag for iLQR inner loop start
            end

            # Backward pass
            calc_jacobians(results, solver)
            if solver.control_integration == :foh
                v1, v2 = backwardpass_foh!(results,solver) #TODO combine with square root
            elseif solver.opts.square_root
                v1, v2 = backwards_sqrt!(results, solver) #TODO option to help avoid ill-conditioning [see algorithm xx]
            else
                v1, v2 = backwardpass!(results, solver)
            end

            # Forward pass
            J = forwardpass!(results, solver, v1, v2)

            if solver.opts.cache
                t2 = time_ns() # time flag of iLQR inner loop end
            end

            # Update results
            X .= X_
            U .= U_
            dJ = copy(abs(J-J_prev)) # change in cost
            J_prev = copy(J)

            if solver.opts.cache
                # Store current results and performance parameters
                time = (t2-t1)/(1.0e9)
                add_iter!(results_cache, results, J, time, iter)
                iter += 1
            end

            # Check for cost convergence
            if (results isa UnconstrainedResults && dJ < solver.opts.eps) || (results isa ConstrainedResults && dJ < solver.opts.eps_intermediate)
                if solver.opts.verbose
                    println("--iLQR (inner loop) cost eps criteria met at iteration: $i\n")
                    if results isa UnconstrainedResults
                        println("Unconstrained solve complete")
                    end
                end
                break
            end
        end

        if solver.opts.cache
            results_cache.iter_type[iter-1] = 1 # flag outerloop update
        end

        ## Outer loop update for Augmented Lagrange Method parameters
        outer_loop_update(results,solver)

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
        println("Outer loop $k (end)\n")

    end

    if solver.opts.cache
        # Store final results
        results_cache.termination_index = iter-1
        results_cache.X = results.X
        results_cache.U = results.U
    end

    ## Return dynamically feasible trajectory
    if infeasible
        if solver.opts.cache
            println("Infeasible -> Feasible ")
            results_cache_2 = feasible_traj(results,solver) # using current control solution, warm-start another solve with dynamics strictly enforced
            return merge_results_cache(results_cache,results_cache_2) # return infeasible results and final enforce dynamics results
        else
            return feasible_traj(results,solver)
        end
    else
        if solver.opts.verbose
            println("***Solve Complete***")
        end
        if solver.opts.cache
            return results_cache
        else
            return results
        end
    end
end


"""
$(SIGNATURES)

Updates penalty (μ) and Lagrange multiplier (λ) parameters for Augmented Lagrange Method. λ is updated for equality and inequality constraints according to [insert equation ref] and μ is incremented by a constant term for all constraint types.
"""
function outer_loop_update(results::ConstrainedResults,solver::Solver)::Void
    p,N = size(results.C) # note I changed C to be (p,N)
    if solver.control_integration == :foh
        final_index = N
    else
        final_index = N-1
    end

    for jj = 1:final_index
        for ii = 1:p
            if ii <= solver.obj.pI
                results.LAMBDA[ii,jj] .+= results.MU[ii,jj]*min(results.C[ii,jj],0)
            else
                results.LAMBDA[ii,jj] .+= results.MU[ii,jj]*results.C[ii,jj]
            end
            results.MU[ii,jj] .+= solver.opts.mu_al_update
        end
    end
    results.λN .+= results.μN.*results.CN
    results.μN .+= solver.opts.mu_al_update
    return nothing
end

function outer_loop_update(results::UnconstrainedResults,solver::Solver)::Void
    return nothing
end
