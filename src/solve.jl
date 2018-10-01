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
function _solve(solver::Solver, U0::Array{Float64,2}, X0::Array{Float64,2}=Array{Float64}(undef,0,0); prevResults::SolverResults=ConstrainedVectorResults())::Tuple{SolverResults,Dict}
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

    use_static = solver.opts.use_static

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
            results = UnconstrainedStaticResults(n,m,N)
        else
            results = UnconstrainedVectorResults(n,m,N)
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
            results = ConstrainedStaticResults(n,m,p,N)
        else
            results = ConstrainedVectorResults(n,m,p,N)
        end
        results.MU .*= solver.opts.μ1 # set initial penalty term values

        if infeasible
            copyto!(results.X, X0)  # initialize state trajectory with infeasible trajectory input
            copyto!(results.U, [U0; ui])  # augment control with additional control inputs that produce infeasible state trajectory
        else
            copyto!(results.U, U0) # initialize control to control input sequence

            # bootstrap previous constrained solution
            if !isempty(prevResults)
                println("BOOTSTRAPPING λ")
                for k = 1:solver.N
                    results.LAMBDA[k] = prevResults.LAMBDA[k][1:solver.obj.p]
                end
                results.λN .= prevResults.λN
            end
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
        X[1] = solver.obj.x0
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
    λ_second_order_idx = Vector{Int64}()

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
    sqrt_tolerance = true

    for j = 1:solver.opts.iterations_outerloop
        iter_outer = j
        if solver.opts.verbose
            println("Outer loop $j (begin)")
        end

        if is_constrained
            update_constraints!(results,solver,results.X,results.U)
            if j == 1
                results.C_prev .= deepcopy(results.C)
                results.CN_prev .= deepcopy(results.CN)
            end
        end
        c_max = 0.  # Init max constraint violation to increase scope
        J_prev = cost(solver, results, X, U)
        j == 1 ? push!(J_hist, J_prev) : nothing  # store the first cost

        if solver.opts.verbose
            println("Cost ($j): $J_prev\n")
        end

        #****************************#
        #         INNER LOOP         #
        #****************************#

        for i = 1:solver.opts.iterations
            iter_inner = i
            if solver.opts.verbose
                println("--Iteration: $j-($i)--")
            end

            if solver.opts.cache
                t1 = time_ns() # time flag for iLQR inner loop start
            end

            ### BACKWARD PASS ###
            calculate_jacobians!(results, solver)
            if solver.control_integration == :foh
                Δv = backwardpass_foh!(results,solver)
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
            X .= deepcopy(X_)
            U .= deepcopy(U_)

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
            d_grad = maximum(map((x)->maximum(abs.(x)),results.d))
            s_grad = maximum(abs.(results.s[1]))
            todorov_grad = calculate_todorov_gradient(results)

            if solver.opts.verbose
                println("d gradient: $d_grad")
                println("s gradient: $s_grad")
                println("todorov gradient $(todorov_grad)")
            end
            gradient = todorov_grad

            if (~is_constrained && gradient < solver.opts.gradient_tolerance) || (is_constrained && gradient < solver.opts.gradient_intermediate_tolerance && j != solver.opts.iterations_outerloop)
                if solver.opts.verbose
                    println("--iLQR (inner loop) cost eps criteria met at iteration: $i\n")
                    if ~is_constrained
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
            if (~is_constrained && dJ < solver.opts.cost_tolerance) || (is_constrained && dJ < solver.opts.cost_intermediate_tolerance && j != solver.opts.iterations_outerloop)
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

        # check sqrt convergence criteria for second order lagrange multiplier update
        if solver.opts.λ_second_order_update
            if c_max <= sqrt(solver.opts.constraint_tolerance) && (dJ <= sqrt(solver.opts.cost_tolerance) || gradient <= sqrt(solver.opts.gradient_tolerance))
                sqrt_tolerance = true
                push!(λ_second_order_idx, iter-1)
            end
        end

        # update multiplier and penalty terms
        outer_loop_update(results,solver,true)#sqrt_tolerance)

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
                    println("-Outer loop cost and constraint eps criteria met at outer iteration: $j\n")
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
            println("Outer loop $j (end)\n -----")
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
                 "c_max"=>c_max_hist,
                 "λ_second_order"=>λ_second_order_idx)

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
            results_feasible, stats_feasible = solve(solver,to_array(results_feasible.U),prevResults=results_feasible)

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
            append!(stats["λ_second_order"],stats_feasible["λ_second_order"])
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
    results_feasible = no_infeasible_controls_unconstrained_results(results,solver)

    # backward pass (ie, time varying lqr)
    if solver.control_integration == :foh
        Δv = backwardpass_foh!(results_feasible,solver)
    elseif solver.opts.square_root
        Δv = backwardpass_sqrt!(results_feasible, solver)
    else
        Δv = backwardpass!(results_feasible, solver)
    end

    # rollout
    forwardpass!(results_feasible,solver,Δv)
    results_feasible.X .= results_feasible.X_
    results_feasible.U .= results_feasible.U_

    # return constrained results if input was constrained
    if !solver.opts.unconstrained
        results_feasible = new_constrained_results(results_feasible,solver,results.LAMBDA,results.λN)
        update_constraints!(results_feasible,solver,results_feasible.X,results_feasible.U)
        calculate_jacobians!(results_feasible,solver)
    end

    return results_feasible
end

"""
$(SIGNATURES)
    For infeasible solve, return an unconstrained results from a prior unconstrained or constrained results
        -removes infeasible controls and infeasible components in Jacobians
"""
function no_infeasible_controls_unconstrained_results(r::SolverIterResults,solver::Solver)::UnconstrainedIterResults
    n,m,N = get_sizes(solver)
    if solver.opts.use_static
        results = UnconstrainedStaticResults(n,m,N)
    else
        results = UnconstrainedVectorResults(n,m,N)
    end
    copyto!(results.X,r.X)
    copyto!(results.xdot,r.xdot)
    copyto!(results.xmid,r.xmid)
    for k = 1:N
        results.U[k] = r.U[k][1:m]
        results.Ac[k] = r.Ac[k][1:n,1:n]
        results.Bc[k] = r.Bc[k][1:n,1:m]
        k == N ? continue : nothing
        results.fx[k] = r.fx[k][1:n,1:n]
        results.fu[k] = r.fu[k][1:n,1:m]
    end
    results
end

"""
$(SIGNATURES)
    For infeasible solve, return a constrained results from a prior unconstrained or constrained results
"""
function new_constrained_results(r::SolverIterResults,solver::Solver,λ,λN)::ConstrainedIterResults
    n,m,N = get_sizes(solver)
    p = solver.obj.p
    p_N = solver.obj.p_N
    if solver.opts.use_static
        results = ConstrainedStaticResults(n,m,p,N,p_N)
    else
        results = ConstrainedVectorResults(n,m,p,N,p_N)
    end
    copyto!(results.X,r.X)
    copyto!(results.xdot,r.xdot)
    copyto!(results.xmid,r.xmid)
    for k = 1:N
        results.U[k] = r.U[k][1:m]
        results.Ac[k] = r.Ac[k][1:n,1:n]
        results.Bc[k] = r.Bc[k][1:n,1:m]
        results.LAMBDA[k] = λ[k][1:p]
        k == N ? continue : nothing
        results.fx[k] = r.fx[k][1:n,1:n]
        results.fu[k] = r.fu[k][1:n,1:m]
        results.fv[k] = r.fv[k][1:n,1:m]
    end
    results.λN .= λN

    results
end

function λ_update(results::ConstrainedIterResults,solver::Solver,k::Int64)
    if solver.control_integration == :zoh
        # Build the Hessian of the Lagrangian and stack: constraints, Jacobians, multipliers
        n = solver.model.n
        m = solver.model.m
        p = length(results.C[1])
        pI = solver.obj.pI
        N = solver.N
        Q = solver.obj.Q
        R = getR(solver)
        Qf = solver.obj.Qf
        dt = solver.dt
        if solver.model.m != length(results.U[1])
            m += n
        end

        if k != solver.N
            # get inequality indices
            idx_inequality = [i for i = 1:pI]

            # assemble constraints
            c = results.C[k]
            cz = [results.Cx[k] results.Cu[k]]

            # assemble lagrange multipliers
            λ = results.LAMBDA[k]

            # assemble penalty matrix
            μ = results.Iμ[k]
        else
            # assemble pieces from terminal condition
            c = results.CN
            cz = results.Cx_N
            λ = results.λN
            μ = results.IμN
        end

        # first order multiplier update
        if !solver.opts.λ_second_order_update
            λ .= λ + μ*c
            if k != solver.N
                λ[idx_inequality] = max.(0.0,λ[idx_inequality])
            end
            return nothing
        end

        # second order multiplier update
        if k != solver.N
            constraint_status = ones(Bool,p)

            # check for active inequality constraints
            for i in idx_inequality
                if c[i] <= 0.0
                    constraint_status[i] = false
                end
            end
        else
            constraint_status = ones(Bool,n)
        end

        # get indices of all active constraints
        idx_active = findall(x->x==true,constraint_status)

        ## Build Hessian
        # stage costs
        if k != solver.N
            L = [dt*Q zeros(n,m); zeros(m,n) dt*R]
        else
            L = Qf
        end

        # constraints (active inequality and equality)
        L += cz[idx_active,:]'*μ[idx_active,idx_active]*cz[idx_active,:]

        B = cz[idx_active,:]*(L\cz[idx_active,:]')

        λ[idx_active] = λ[idx_active] + B\I*c[idx_active] #this is a bit mysterious to me, but I was finding on the foh that without the I, the \ was giving different results from using inv(), so I've included the I here as well

        if k != solver.N
            λ[idx_inequality] = max.(0.0,λ[idx_inequality])
        end

        return nothing
    end

    # if solver.control_integration == :foh
    #     n = solver.model.n
    #     m = solver.model.m
    #     q = n+m
    #     p = solver.obj.p
    #     pI = solver.obj.pI
    #     N = solver.N
    #     Q = solver.obj.Q
    #     R = getR(solver)
    #     Qf = solver.obj.Qf
    #     dt = solver.dt
    #     if solver.model.m != length(results.U[1])
    #         m += n
    #         p += n
    #     end
    #
    #     # initialize
    #     L = zeros((n+m)*N,(n+m)*N)
    #     c = zeros(p*N+n)
    #     cz = zeros(p*N+n,(n+m)*N)
    #     λ = zeros(p*N+n)
    #     μ = zeros(p*N+n,p*N+n)
    #
    #     # get inequality indices
    #     idx_inequality = []
    #     tmp = [i for i = 1:pI]
    #
    #     for k = 1:N
    #         # get all inequality indices
    #         idx_inequality = cat(idx_inequality,tmp .+ (k-1)*p,dims=(1,1))
    #
    #         # assemble constraints
    #         c[(k-1)*p+1:(k-1)*p+p] = results.C[k]
    #         cz[(k-1)*p+1:(k-1)*p+p,(k-1)*(n+m)+1:(k-1)*(n+m)+(n+m)] = [results.Cx[k] results.Cu[k]]
    #
    #         # assemble lagrange multipliers
    #         λ[(k-1)*p+1:(k-1)*p+p] = results.LAMBDA[k]
    #
    #         # assemble penalty matrix
    #         μ[(k-1)*p+1:(k-1)*p+p,(k-1)*p+1:(k-1)*p+p] = results.Iμ[k]
    #     end
    #     # assemble from terminal
    #     c[N*p+1:N*p+n] = results.CN
    #     cz[N*p+1:N*p+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = results.Cx_N
    #     λ[N*p+1:N*p+n] = results.λN
    #     μ[N*p+1:N*p+n,N*p+1:N*p+n] = results.IμN
    #
    #     # first order multiplier update
    #     if !solver.opts.λ_second_order_update
    #         λ .= λ + μ*c
    #         λ[idx_inequality] = max.(0.0,λ[idx_inequality])
    #
    #         # update results
    #         for k = 1:N
    #             results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
    #         end
    #         results.λN .= λ[N*p+1:N*p+n]
    #
    #         return nothing
    #     end
    #
    #     # second order multiplier update
    #     constraint_status = ones(Bool,N*p+n)
    #
    #     # get active constraints
    #     for i in idx_inequality
    #         if c[i] <= 0.0
    #             constraint_status[i] = false
    #         end
    #     end
    #
    #     # get indices of active constraints
    #     idx_active = findall(x->x==true,constraint_status)
    #
    #     ## Build Hessian
    #     # stage costs
    #     for k = 1:N-1
    #         # Unpack Jacobians, ̇x
    #         Ac1, Bc1 = results.Ac[k], results.Bc[k]
    #         Ac2, Bc2 = results.Ac[k+1], results.Bc[k+1]
    #         Ad, Bd, Cd = results.fx[k], results.fu[k], results.fv[k]
    #
    #         xm = results.xmid[k]
    #         um = (U[k] + U[k+1])/2.0
    #
    #         lxx = dt/6*Q + 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 + dt/8*Ac1)
    #         luu = dt/6*R + 4*dt/6*((dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5)
    #         lyy = dt/6*Q + 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(I/2 - dt/8*Ac2)
    #         lvv = dt/6*R + 4*dt/6*((-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)
    #
    #         lxu = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(dt/8*Bc1)
    #         lxy = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 - dt/8*Ac2)
    #         lxv = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(-dt/8*Bc2)
    #         luy = 4*dt/6*(dt/8*Bc1)'*Q*(I/2 - dt/8*Ac2)
    #         luv = 4*dt/6*((dt/8*Bc1)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)
    #         lyv = 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(-dt/8*Bc2)
    #
    #         L[(k-1)*(n+m)+1:(k-1)*(n+m)+2*(n+m),(k-1)*(n+m)+1:(k-1)*(n+m)+2*(n+m)] = [lxx lxu lxy lxv; lxu' luu luy luv; lxy' luy' lyy lyv; lxv' luv' lyv' lvv]
    #     end
    #     L[(N-1)*(n+m)+1:(N-1)*(n+m)+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = Qf
    #
    #     # active constraints
    #     L += cz[idx_active,:]'*μ[idx_active,idx_active]*cz[idx_active,:]
    #
    #     B = cz[idx_active,:]*(L\cz[idx_active,:]')
    #
    #     if solver.opts.λ_second_order_update
    #         λ[idx_active] = λ[idx_active] + B\I*c[idx_active] # this is a bit mysterious to me, but I was finding on the foh that without the I, the \ was giving different results from using inv(), so I've included the I here
    #         λ[idx_inequality] = max.(0.0,λ[idx_inequality])
    #     end
    #
    #     # update results
    #     for k = 1:N
    #         results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
    #     end
    #     results.λN .= λ[N*p+1:N*p+n]
    #
    #     return nothing
    # end
end

function λ_update(results::ConstrainedIterResults,solver::Solver)
    if solver.control_integration == :zoh
        # Build the Hessian of the Lagrangian and stack: constraints, Jacobians, multipliers
        n = solver.model.n
        m = solver.model.m
        p = length(results.C[1])
        pI = solver.obj.pI
        N = solver.N
        Q = solver.obj.Q
        R = getR(solver)
        Qf = solver.obj.Qf
        dt = solver.dt
        if solver.model.m != length(results.U[1])
            m += n
        end

        # initialize
        L = zeros((n+m)*(N-1)+n,(n+m)*(N-1)+n)
        c = zeros(p*(N-1)+n)
        cz = zeros(p*(N-1)+n,(n+m)*(N-1)+n)
        λ = zeros(p*(N-1)+n)
        μ = zeros(p*(N-1)+n,p*(N-1)+n)

        # get inequality indices
        idx_inequality = []
        tmp = [i for i = 1:pI]

        for k = 1:N-1
            # generate all inequality constraint indices
            idx_inequality = cat(idx_inequality,tmp .+ (k-1)*p,dims=(1,1))

            # assemble constraints
            c[(k-1)*p+1:(k-1)*p+p] = results.C[k]
            cz[(k-1)*p+1:(k-1)*p+p,(k-1)*(n+m)+1:(k-1)*(n+m)+(n+m)] = [results.Cx[k] results.Cu[k]]

            # assemble lagrange multipliers
            λ[(k-1)*p+1:(k-1)*p+p] = results.LAMBDA[k]

            # assemble penalty matrix
            μ[(k-1)*p+1:(k-1)*p+p,(k-1)*p+1:(k-1)*p+p] = results.Iμ[k]
        end
        # assemble pieces from terminal condition
        c[(N-1)*p+1:(N-1)*p+n] = results.CN
        cz[(N-1)*p+1:(N-1)*p+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = results.Cx_N
        λ[(N-1)*p+1:(N-1)*p+n] = results.λN
        μ[(N-1)*p+1:(N-1)*p+n,(N-1)*p+1:(N-1)*p+n] = results.IμN

        # first order multiplier update
        if !solver.opts.λ_second_order_update
            λ .= λ + μ*c
            λ[idx_inequality] = max.(0.0,λ[idx_inequality])

            # update results
            for k = 1:N-1
                results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
            end
            results.λN .= λ[(N-1)*p+1:(N-1)*p+n]
            return nothing
        end

        # second order multiplier update
        constraint_status = ones(Bool,p*(N-1)+n)

        # check for active inequality constraints
        for i in idx_inequality
            if c[i] <= 0.0
                constraint_status[i] = false
            end
        end

        # get indices of all active constraints
        idx_active = findall(x->x==true,constraint_status)

        ## Build Hessian
        # stage costs
        for k = 1:N-1
            L[(k-1)*(n+m)+1:(k-1)*(n+m)+(n+m),(k-1)*(n+m)+1:(k-1)*(n+m)+(n+m)] = [dt*Q zeros(n,m); zeros(m,n) dt*R]
        end
        L[(N-1)*(n+m)+1:(N-1)*(n+m)+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = Qf

        # constraints (active inequality and equality)
        L += cz[idx_active,:]'*μ[idx_active,idx_active]*cz[idx_active,:]

        B = cz[idx_active,:]*(L\cz[idx_active,:]')

        if solver.opts.λ_second_order_update
            λ[idx_active] = λ[idx_active] + B\I*c[idx_active] #this is a bit mysterious to me, but I was finding on the foh that without the I, the \ was giving different results from using inv(), so I've included the I here as well
            λ[idx_inequality] = max.(0.0,λ[idx_inequality])
        end

        # update results
        for k = 1:N-1
            results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
        end
        results.λN .= λ[(N-1)*p+1:(N-1)*p+n]

        return nothing
    end

    if solver.control_integration == :foh
        n = solver.model.n
        m = solver.model.m
        q = n+m
        p = length(results.C[1])
        pI = solver.obj.pI
        N = solver.N
        Q = solver.obj.Q
        R = getR(solver)
        Qf = solver.obj.Qf
        dt = solver.dt
        if solver.model.m != length(results.U[1])
            m += n
        end

        # initialize
        L = zeros((n+m)*N,(n+m)*N)
        c = zeros(p*N+n)
        cz = zeros(p*N+n,(n+m)*N)
        λ = zeros(p*N+n)
        μ = zeros(p*N+n,p*N+n)

        # get inequality indices
        idx_inequality = []
        tmp = [i for i = 1:pI]

        for k = 1:N
            # get all inequality indices
            idx_inequality = cat(idx_inequality,tmp .+ (k-1)*p,dims=(1,1))

            # assemble constraints
            c[(k-1)*p+1:(k-1)*p+p] = results.C[k]
            cz[(k-1)*p+1:(k-1)*p+p,(k-1)*(n+m)+1:(k-1)*(n+m)+(n+m)] = [results.Cx[k] results.Cu[k]]

            # assemble lagrange multipliers
            λ[(k-1)*p+1:(k-1)*p+p] = results.LAMBDA[k]

            # assemble penalty matrix
            μ[(k-1)*p+1:(k-1)*p+p,(k-1)*p+1:(k-1)*p+p] = results.Iμ[k]
        end
        # assemble from terminal
        c[N*p+1:N*p+n] = results.CN
        cz[N*p+1:N*p+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = results.Cx_N
        λ[N*p+1:N*p+n] = results.λN
        μ[N*p+1:N*p+n,N*p+1:N*p+n] = results.IμN

        # first order multiplier update
        if !solver.opts.λ_second_order_update
            λ .= λ + μ*c
            λ[idx_inequality] = max.(0.0,λ[idx_inequality])

            # update results
            for k = 1:N
                results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
            end
            results.λN .= λ[N*p+1:N*p+n]

            return nothing
        end

        # second order multiplier update
        constraint_status = ones(Bool,N*p+n)

        # get active constraints
        for i in idx_inequality
            if c[i] <= 0.0
                constraint_status[i] = false
            end
        end

        # get indices of active constraints
        idx_active = findall(x->x==true,constraint_status)

        ## Build Hessian
        # stage costs
        for k = 1:N-1
            # Unpack Jacobians, ̇x
            Ac1, Bc1 = results.Ac[k], results.Bc[k]
            Ac2, Bc2 = results.Ac[k+1], results.Bc[k+1]

            xm = results.xmid[k]
            um = (U[k] + U[k+1])/2.0

            lxx = dt/6*Q + 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 + dt/8*Ac1)
            luu = dt/6*R + 4*dt/6*((dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5)
            lyy = dt/6*Q + 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(I/2 - dt/8*Ac2)
            lvv = dt/6*R + 4*dt/6*((-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)

            lxu = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(dt/8*Bc1)
            lxy = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 - dt/8*Ac2)
            lxv = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(-dt/8*Bc2)
            luy = 4*dt/6*(dt/8*Bc1)'*Q*(I/2 - dt/8*Ac2)
            luv = 4*dt/6*((dt/8*Bc1)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)
            lyv = 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(-dt/8*Bc2)

            L[(k-1)*(n+m)+1:(k-1)*(n+m)+2*(n+m),(k-1)*(n+m)+1:(k-1)*(n+m)+2*(n+m)] = [lxx lxu lxy lxv; lxu' luu luy luv; lxy' luy' lyy lyv; lxv' luv' lyv' lvv]
        end
        L[(N-1)*(n+m)+1:(N-1)*(n+m)+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = Qf

        # active constraints
        L += cz[idx_active,:]'*μ[idx_active,idx_active]*cz[idx_active,:]

        B = cz[idx_active,:]*(L\cz[idx_active,:]')

        λ[idx_active] = λ[idx_active] + B\I*c[idx_active] # this is a bit mysterious to me, but I was finding on the foh that without the I, the \ was giving different results from using inv(), so I've included the I here
        λ[idx_inequality] = max.(0.0,λ[idx_inequality])

        # update results
        for k = 1:N
            results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
        end
        results.λN .= λ[N*p+1:N*p+n]

        return nothing
    end
end

"""
$(SIGNATURES)
    Updates penalty (μ) and Lagrange multiplier (λ) parameters for Augmented Lagrange Method. λ is updated for equality and inequality constraints
    according to [insert equation ref] and μ is incremented by a constant term for all constraint types.
        -  ALGENCAN 'uniform' update: see 'Practical Augmented Lagrangian Methods for Constrained Optimization' (Algorithm 4.1, p.33)
        - 'individual' update: see Bertsekas 'Constrained Optimization' (eq. 47, p.123)
"""
function outer_loop_update(results::ConstrainedIterResults,solver::Solver,sqrt_tolerance::Bool=false)::Nothing
    n,m,N = get_sizes(solver)
    p = length(results.C[1])  # number of constraints
    pI = solver.obj.pI  # number of inequality constraints

    if solver.control_integration == :foh
        final_index = N
    else
        final_index = N-1
    end

    ## Lagrange multiplier updates ###
    # for k = 1:final_index
    #     results.LAMBDA[k] = results.LAMBDA[k] + results.Iμ[k]*results.C[k]
    #     results.LAMBDA[k][1:pI] = max.(0.0,results.LAMBDA[k][1:pI])
    # end
    #
    # results.λN .= max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λN + results.IμN*results.CN))
    for k = 1:solver.N
        λ_update(results,solver,k)
    end
    # λ_update(results,solver)

    ### Penalty updates ###
    # 'default' penaltiy update - all penalty terms are updated (no conditions)
    if solver.opts.outer_loop_update == :default
        for k = 1:final_index
            results.MU[k] .= min.(solver.opts.μ_max, solver.opts.γ*results.MU[k])
        end
        results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)
    end


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
