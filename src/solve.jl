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
            calculate_jacobians!(results, solver)
            if solver.control_integration == :foh
                Δv = backwardpass_foh_alt!(results,solver) #TODO combine with square root
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

        outer_loop_update(results,solver,max_violation(results))

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
    Return constraints, Jacobians, and numbers relating to the current active constraints
    - inequality constraints are defined as c() < 0
"""
function get_active_constraints(c::AbstractVector,cx::AbstractMatrix,p::Int64,pI::Int64,n::Int64)
    # create a mask for inactive inequality constraints
    mask = ones(Bool,p)
    for i = 1:pI
        if c[i] <= 0.0
            mask[i] = false
        end
    end
    mask

    # get indicies of active constraints (inequality and equality)
    idx_active = findall(x->x==1,mask)
    idx_inactive = findall(x->x==0,mask)

    p_active = count(mask)
    p_inactive = p - p_active

    c_active = zeros(p_active)
    cx_active = zeros(p_active,n)
    c_active .= c[idx_active]
    cx_active .= cx[idx_active,:]

    c_active, cx_active, p_active, p_inactive, idx_active, idx_inactive
end

"""
$(SIGNATURES)
    First order update for Lagrange multipliers (sequential individual)
        -see Bertsekas 'Constrained Optimization' (p. 101)
"""
function λ_update_first_order!(results::ConstrainedResults,solver::Solver,i::Int64,j::Int64,mode::Symbol=:stage,constraint_type::Symbol=:equality)
    if mode == :stage
        if constraint_type == :equality
            results.LAMBDA[i,j] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.LAMBDA[i,j] + results.MU[i,j]*results.C[i,j])) # λ_min < λ < λ_max
        end
        if constraint_type == :inequality
            results.LAMBDA[i,j] = max(solver.opts.λ_min, min(solver.opts.λ_max, max(0.0, results.LAMBDA[i,j] + results.MU[i,j]*results.C[i,j]))) # λ_min < λ < λ_max
        end
    elseif mode == :terminal
        results.λN[i] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.λN[i] + results.μN[i].*results.CN[i]))
    end
end

"""
$(SIGNATURES)
    Second order update for Lagrange multipliers (sequential by timestep)
        -see Bertsekas 'Constrained Optimization' (eq. 24, p. 133)
        -note we use the augmented state z = [x; u]
"""
function λ_update_second_order!(results::ConstrainedResults,solver::Solver,mode::Symbol=:stage,k::Int64=0)
    # update stage multipliers
    # println("***Second order λ update***")
    if mode == :stage && solver.control_integration == :zoh
        n = solver.model.n
        m = solver.model.m

        Q = solver.obj.Q
        R = solver.obj.R
        if size(results.U,1) != m
            R = getR(solver)
            m = size(R,1)
        end
        dt = solver.dt

        pI = solver.obj.pI
        p = size(results.C,1)

        c_active, cz_active, p_active, p_inactive, idx_active, idx_inactive = get_active_constraints(results.C[:,k],[results.Cx[:,:,k] results.Cu[:,:,k]],p,pI,n+m)
        lzz = [dt*Q zeros(n,m); zeros(m,n) dt*R]
        Lzz = lzz + cz_active'*results.Iμ[idx_active,idx_active,k]*cz_active
        B = cz_active*(Lzz\cz_active')

        # Lagrange multiplier update (1st order) for inactive constraints
        results.LAMBDA[idx_inactive,k] = results.LAMBDA[idx_inactive,k] + Matrix(Diagonal(results.MU[idx_inactive,k]))*results.C[idx_inactive,k]

        # Lagrange multiplier update (2nd order) for active constraints
        results.LAMBDA[idx_active,k] = results.LAMBDA[idx_active,k] + B\results.C[idx_active,k]

        # additional criteria for inequality constraints (ie, λ for inequality constraints should not go negative)
        results.LAMBDA[1:pI,k] = max.(0.0,results.LAMBDA[1:pI,k])
    end

    ## FOH
    if mode == :nonsequential && solver.control_integration == :foh
        N = solver.N
        n = solver.model.n
        m = solver.model.m

        # infeasible
        if size(results.U,1) != m
            m += n
        end
        q = n + m
        q2 = 2*q
        Q = solver.obj.Q
        R = getR(solver)
        dt = solver.dt

        # Initialized Hessian of Lagrangian
        Lzz = zeros(q*N,q*N)

        # Collect terms from stage cost
        for i = 1:N-1
            Ac1 = results.Ac[:,:,i]
            Bc1 = results.Bc[:,:,i]
            Ac2 = results.Ac[:,:,i+1]
            Bc2 = results.Bc[:,:,i+1]

            # Expansion of stage cost L(x,u,y,v) -> dL(dx,du,dy,dv)
            Lxx = dt/6*Q + (dt/6*Q + 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 + dt/8*Ac1)) # l(x,u) and l(xm,um) terms
            Luu = dt/6*R + (dt/6*R + 4*dt/6*((dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5)) # l(x,u) and l(xm,um) terms
            Lyy = dt/6*Q + (dt/6*Q + 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(I/2 - dt/8*Ac2)) # l(y,v) and l(xm,um) terms
            Lvv = dt/6*R + (dt/6*R + 4*dt/6*((-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)) # l(y,v) and l(xm,um) terms

            Lxu = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(dt/8*Bc1)
            Lxy = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 - dt/8*Ac2)
            Lxv = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(-dt/8*Bc2)
            Luy = 4*dt/6*(dt/8*Bc1)'*Q*(I/2 - dt/8*Ac2)
            Luv = 4*dt/6*((dt/8*Bc1)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)
            Lyv = 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(-dt/8*Bc2)

            L = [Lxx Lxu Lxy Lxv;
                 Lxu' Luu Luy Luv;
                 Lxy' Luy' Lyy Lyv;
                 Lxv' Luv' Lyv' Lvv]

            Lzz[q*(i-1)+1:q*(i-1)+q2, q*(i-1)+1:q*(i-1)+q2] += L
        end

        # Collect terms from constraints
        p = size(results.C,1)
        pI = solver.obj.pI
        c_aug = zeros(p*N)
        cz_aug = zeros(p*N,q*N)
        idx_active_aug = []
        idx_inactive_aug = []

        for i = 1:N
            c_active, cz_active, p_active, p_inactive, idx_active, idx_inactive = TrajectoryOptimization.get_active_constraints(results.C[:,i],[results.Cx[:,:,i] results.Cu[:,:,i]],p,pI,q)

            Lzz[q*(i-1)+1:q*(i-1)+q, q*(i-1)+1:q*(i-1)+q] += cz_active'*results.Iμ[idx_active,idx_active,i]*cz_active

            idx = (i-1)*p .+ idx_active

            if p_active > 0
                c_aug[idx] = c_active
                cz_aug[idx,(i-1)*q+1:(i-1)*q+q] = cz_active
                idx_active_aug = cat(idx_active_aug,idx,dims=(1,1))

            elseif p_inactive > 0
                idx_inactive_aug = cat(idx_inactive_aug,(i-1)*p .+ idx_inactive,dims=(1,1))
            end
        end

        # Calculate B
        B = cz_aug[idx_active_aug,:]*(Lzz\cz_aug[idx_active_aug,:]')

        # update inactive multipliers (1st order)
        results.LAMBDA[idx_inactive_aug] = results.LAMBDA[idx_inactive_aug] + Matrix(Diagonal(results.MU[idx_inactive_aug]))*results.C[idx_inactive_aug]

        # update active multipliers (2nd order)
        results.LAMBDA[idx_active_aug] = results.LAMBDA[idx_active_aug] + B\results.C[idx_active_aug]

        for i = 1:N
            results.LAMBDA[1:pI,i] = max.(0.0,results.LAMBDA[1:pI,i])
        end

    end

    # update for terminal constraints
    if mode == :terminal
        lzz = solver.obj.Qf
        Lzz = lzz + results.Cx_N'*results.IμN*results.Cx_N
        B = results.Cx_N*(Lzz\results.Cx_N')
        results.λN .= results.λN + B\results.CN
    end
end

"""
$(SIGNATURES)
    Updates penalty (μ) and Lagrange multiplier (λ) parameters for Augmented Lagrange Method. λ is updated for equality and inequality constraints according to [insert equation ref] and μ is incremented by a constant term for all constraint types.
        -ALGENCAN 'uniform' update: see 'Practical Augmented Lagrangian Methods for Constrained Optimization' (Algorithm 4.1, p.33)
        -'individual' update: see Bertsekas 'Constrained Optimization' (eq. 47, p.123)
"""
function outer_loop_update(results::ConstrainedResults,solver::Solver,max_constraint::Float64=Inf)::Nothing
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
                if !solver.opts.λ_second_order_update || max_constraint > sqrt(solver.opts.constraint_tolerance)
                    # λ_update_first_order!(results,solver,ii,jj,:stage,:inequality)
                    results.LAMBDA[ii,jj] = max(solver.opts.λ_min, min(solver.opts.λ_max, max(0.0, results.LAMBDA[ii,jj] + results.MU[ii,jj]*results.C[ii,jj]))) # λ_min < λ < λ_max
                end

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
                if !solver.opts.λ_second_order_update || max_constraint > sqrt(solver.opts.constraint_tolerance)
                    # λ_update_first_order!(results,solver,ii,jj,:stage,:equality)
                    results.LAMBDA[ii,jj] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.LAMBDA[ii,jj] + results.MU[ii,jj]*results.C[ii,jj])) # λ_min < λ < λ_max
                end

                # penalty update for 'individual' scheme
                if solver.opts.outer_loop_update == :individual
                    if abs(results.C[ii,jj]) <= solver.opts.τ*abs(results.C_prev[ii,jj])
                        results.MU[ii,jj] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[ii,jj])
                    else
                        results.MU[ii,jj] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[ii,jj])
                    end
                end
            end
        end

        # Lagrange multiplier update (2nd order)
        if solver.opts.λ_second_order_update && max_constraint <= sqrt(solver.opts.constraint_tolerance)
            # if solver.opts.verbose
            #     println("**λ second order update**")
            # end
            λ_update_second_order!(results,solver,:stage,jj)
        end
    end

    # Lagrange multiplier (2st order) update nonsequential (foh)
    if solver.opts.λ_second_order_update && max_constraint <= sqrt(solver.opts.constraint_tolerance)
        # if solver.opts.verbose
        #     println("**λ second order update**")
        # end
        λ_update_second_order!(results,solver,:nonsequential)
    end

    # Lagrange multiplier (1st order) update for terminal state equality constraints
    if !solver.opts.λ_second_order_update || max_constraint > sqrt(solver.opts.constraint_tolerance)
        for ii = 1:solver.model.n
            # λ_update_first_order!(results,solver,ii,0,:terminal)
            results.λN[ii] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.λN[ii] + results.μN[ii].*results.CN[ii]))
        end
    end

    # Lagrange multiplier (2st order) update for terminal state equality constraints
    if solver.opts.λ_second_order_update && max_constraint <= sqrt(solver.opts.constraint_tolerance)
        # if solver.opts.verbose
        #     println("**λ second order update**")
        # end
        λ_update_second_order!(results,solver,:terminal)
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


function outer_loop_update(results::ConstrainedResultsStatic,solver::Solver,max_constraint::Float64=Inf)::Nothing
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
                if !solver.opts.λ_second_order_update || max_constraint > sqrt(solver.opts.constraint_tolerance)
                    results.LAMBDA[jj][ii] = max(solver.opts.λ_min, min(solver.opts.λ_max, max(0.0, results.LAMBDA[jj][ii] + results.MU[jj][ii]*results.C[jj][ii]))) # λ_min < λ < λ_max
                end
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
                if !solver.opts.λ_second_order_update || max_constraint > sqrt(solver.opts.constraint_tolerance)
                    results.LAMBDA[jj][ii] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.LAMBDA[jj][ii] + results.MU[jj][ii]*results.C[jj][ii])) # λ_min < λ < λ_max
                end

                # penalty update for 'individual' scheme
                if solver.opts.outer_loop_update == :individual
                    if abs(results.C[ii][jj]) <= solver.opts.τ*abs(results.C_prev[ii][jj])
                        results.MU[ii][jj] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[ii][jj])
                    else
                        results.MU[ii][jj] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[ii][jj])
                    end
                end
            end
        end
    end

    # Lagrange multiplier (1st order) update for terminal state equality constraints (note terminal is already taken care in 2nd order update)
    if !solver.opts.λ_second_order_update || max_constraint > sqrt(solver.opts.constraint_tolerance)
        for ii = 1:solver.model.n
            results.λN[ii] = max(solver.opts.λ_min, min(solver.opts.λ_max, results.λN[ii] + results.μN[ii].*results.CN[ii]))
        end
    end

    # Lagrange multiplier update (2nd order)
    if solver.opts.λ_second_order_update && max_constraint <= sqrt(solver.opts.constraint_tolerance)
        λ_update_second_order!(results, solver)
        if solver.opts.verbose
            println("**λ second order update**")
        end
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

function outer_loop_update(results::UnconstrainedResults,solver::Solver,max_constraint::Float64)::Nothing
    return nothing
end

function outer_loop_update(results::UnconstrainedResultsStatic,solver::Solver,max_constraint::Float64)::Nothing
    return nothing
end
