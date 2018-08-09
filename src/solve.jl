import Base.println

"""
$(SIGNATURES)
Solve the trajectory optimization problem defined by `solver`, with `U0` as the
initial guess for the controls
"""
function solve(solver::Solver, X0::Array{Float64,2}, U0::Array{Float64,2})::SolverResults
    if isa(solver.obj, UnconstrainedObjective)
        obj_c = ConstrainedObjective(solver.obj)
        solver = Solver(solver.model, obj_c, dt=solver.dt, opts=solver.opts)
    end
    solve(solver,X0,U0)
end

function solve(solver::Solver,U0::Array{Float64,2})::SolverResults
    if isa(solver.obj, UnconstrainedObjective)
        solve_unconstrained(solver, U0)
    elseif isa(solver.obj, ConstrainedObjective)
        solve_al(solver,U0)
    end
end

function solve(solver::Solver)::SolverResults
    # Generate random control sequence
    U = rand(solver.model.m, solver.N-1)
    solve(solver,U)
end


"""
$(SIGNATURES)
Solve an unconstrained optimization problem specified by `solver`
"""
function solve_unconstrained(solver::Solver,U0::Array{Float64,2})::SolverResults
    N = solver.N; n = solver.model.n; m = solver.model.m

    if solver.obj isa UnconstrainedObjective
        X = zeros(n,N)
        U = copy(U0)
        X_ = similar(X)
        U_ = similar(U)
        K = zeros(m,n,N-1)
        d = zeros(m,N-1)
        # results = UnconstrainedResults(X,U,K,d,X_,U_)
        results = UnconstrainedResults(n,m,N)
    elseif solver.obj isa ConstrainedObjective

    end

    # Unpack results for convenience
    X = results.X # state trajectory
    U = results.U # control trajectory
    X_ = results.X_ # updated state trajectory
    U_ = results.U_ # updated control trajectory


    if solver.opts.cache
        # Initialize cache and store initial trajectories and cost
        iter = 1 # counter for total number of iLQR iterations
        results_cache = ResultsCache(solver,solver.opts.iterations+1) #TODO preallocate smaller arrays
        add_iter!(results_cache, results, cost(solver, X, U))
        iter += 1
    end

    # initial roll-out
    X[:,1] = solver.obj.x0
    rollout!(results, solver)
    J_prev = cost(solver, X, U)
    if solver.opts.verbose
        println("Initial Cost: $J_prev\n")
    end

    for i = 1:solver.opts.iterations
        if solver.opts.verbose
            println("*** Iteration: $i ***")
        end

        t1 = time_ns() # time flag for iLQR inner loop start
        # calc_jacobians!(results,solver)
        if solver.opts.square_root
            v1, v2 = backwards_sqrt(results,solver)
        else
            v1, v2 = backwardpass!(results,solver)
        end

        J = forwardpass!(results, solver, v1, v2)

        X .= X_
        U .= U_

        t2 = time_ns() # time flag of iLQR inner loop end

        if solver.opts.cache
            # Store current results and performance parameters
            time = (t2-t1)/(1.0e9)
            add_iter!(results_cache, results, J, time, iter)
            iter += 1
        end

        if abs(J-J_prev) < solver.opts.eps
            if solver.opts.verbose
                print_info("-----SOLVED-----")
                print_info("eps criteria met at iteration: $i")
            end
            break
        end
        J_prev = copy(J)
    end

    if solver.opts.cache
        # Store final results
        results_cache.termination_index = iter-1
        results_cache.X = results.X
        results_cache.U = results.U
    end

    if solver.opts.cache
        return results_cache
    else
        return results
    end
end

"""
$(SIGNATURES)

Solve constrained optimization problem using an initial control trajectory
"""
function solve_al(solver::Solver,U0::Array{Float64,2})
    solve_al(solver,zeros(solver.model.n,solver.N),U0,infeasible=false)
end

"""
$(SIGNATURES)

Solve constrained optimization problem specified by `solver`
"""
# QUESTION: Should the infeasible tag be able to be changed? What happens if we turn it off with an X0?
function solve_al(solver::Solver,X0::Array{Float64,2},U0::Array{Float64,2};infeasible::Bool=true)::SolverResults
    ## Unpack model, objective, and solver parameters
    N = solver.N # number of iterations for the solver (ie, knotpoints)
    n = solver.model.n # number of states
    m = solver.model.m # number of control inputs

    if solver.obj isa UnconstrainedObjective
        solver.opts.iterations_outerloop = 1
        results = UnconstrainedResults(n,m,N)

    elseif solver.obj isa ConstrainedObjective
        p = solver.obj.p # number of inequality and equality constraints
        pI = solver.obj.pI # number of inequality constraints

        if infeasible
            ui = infeasible_controls(solver,X0,U0) # generates n additional control input sequences that produce the desired infeasible state trajectory
            m += n # augment the number of control input sequences by the number of states
            p += n # increase the number of constraints by the number of additional control input sequences
        end

        ## Initialize results
        results = ConstrainedResults(n,m,p,N) # preallocate memory for results

        if infeasible
            #solver.obj.x0 = X0[:,1] # TODO not sure this is correct or needs to be here
            results.X .= X0 # initialize state trajectory with infeasible trajectory input
            results.U .= [U0; ui] # augment control with additional control inputs that produce infeasible state trajectory
        else
            results.U .= U0 # initialize control to control input sequence
        end

        # Diagonal indicies for the Iμ matrix (fast)
        diag_inds = CartesianIndex.(indices(results.Iμ,1),indices(results.Iμ,2))

        # Generate constraint function and jacobian functions from the objective
        c_fun, constraint_jacobian = generate_constraint_functions(solver.obj, infeasible=infeasible)

        # Evalute constraints for new trajectories
        update_constraints!(results,c_fun,pI,results.X,results.U)
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
        rollout!(results,solver) # rollout new state trajectoy
    end

    if solver.opts.cache
        # Initialize cache and store initial trajectories and cost
        iter = 1 # counter for total number of iLQR iterations
        results_cache = ResultsCache(solver,solver.opts.iterations*solver.opts.iterations_outerloop+1) #TODO preallocate smaller arrays
        add_iter!(results_cache, results, cost(solver, X, U, infeasible=infeasible))
        iter += 1
    end

    # Outer Loop
    for k = 1:solver.opts.iterations_outerloop
        # J_prev = cost(solver, results, X, U, infeasible=infeasible) # calculate cost for current trajectories and constraint violations
        J_prev = cost(solver, results, X, U, infeasible)

        if solver.opts.verbose
            println("Cost ($k): $J_prev\n")
        end

        for i = 1:solver.opts.iterations
            if solver.opts.verbose
                println("--Iteration: $k-($i)--")
            end

            if solver.opts.cache
                t1 = time_ns() # time flag for iLQR inner loop start
            end

            # Backward pass
            calc_jacobians(results, solver)
            if solver.opts.square_root
                v1, v2 = backwards_sqrt(results, solver, constraint_jacobian=constraint_jacobian, infeasible=infeasible) #TODO option to help avoid ill-conditioning [see algorithm xx]
            else
                # v1, v2 = backwardpass!(results, solver; kwargs_bp...) # standard backward pass [see insert algorithm]
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
            if dJ < solver.opts.eps
                if solver.opts.verbose
                    println("   eps criteria met at iteration: $i\n")
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
            if max_c < solver.opts.eps_constraint
                if solver.opts.verbose
                    println("\teps constraint criteria met at outer iteration: $k\n")
                end
                break
            end
        end

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
            results_cache_2 = feasible_traj(results,solver) # using current control solution, warm-start another solve with dynamics strictly enforced
            return merge_results_cache(results_cache,results_cache_2) # return infeasible results and final enforce dynamics results
        else
            return feasible_traj(results,solver)
        end
    else
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
    p,N = size(results.C)
    N += 1
    for jj = 1:N-1
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

function println(level::Symbol, msg::String)
    if level_priorities[level] ≥ level_priorities[debug_level]
        println(msg)
    end
end

print_info(msg) = println(:info,msg)
print_debug(msg) = println(:debug,msg)
