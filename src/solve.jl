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

        # Set initial regularization
        results.ρ[1] *= solver.opts.ρ_initial

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

        # Set initial penalty term values
        results.MU .*= solver.opts.μ1

        # Set initial regularization
        results.ρ[1] *= solver.opts.ρ_initial

        if infeasible
            copyto!(results.X, X0)  # initialize state trajectory with infeasible trajectory input
            copyto!(results.U, [U0; ui])  # augment control with additional control inputs that produce infeasible state trajectory
            calculate_derivatives!(results,solver,results.X,results.U)
            calculate_midpoints!(results,solver,results.X,results.U)
        else
            copyto!(results.U, U0) # initialize control to control input sequence

            # bootstrap previous constrained solution
            if !isempty(prevResults)
                results.ρ[1] = prevResults.ρ[1] #TODO consider if this is necessary
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
    iter = 0 # counter for total number of iLQR iterations
    iter_outer = 0
    iter_inner = 0
    time_setup = time_ns() - t_start
    J_hist = Vector{Float64}()
    c_max_hist = Vector{Float64}()
    t_solve_start = time_ns()

    #****************************#
    #         OUTER LOOP         #
    #****************************#

    dJ = Inf
    gradient = Inf
    Δv = Inf
    sqrt_tolerance = false

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

            # increment iLQR inner loop counter
            iter += 1

            ### UPDATE RESULTS ###
            X .= deepcopy(X_)
            U .= deepcopy(U_)

            dJ = copy(abs(J-J_prev)) # change in cost
            J_prev = copy(J)


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
                error("*Regularization maxed out*\n - terminating solve - ")
            end
            ################################
        end
        ### END INNER LOOP ###

        #****************************#
        #      OUTER LOOP UPDATE     #
        #****************************#

        # # check sqrt convergence criteria for second order lagrange multiplier update
        # if solver.opts.λ_second_order_update
        #     if c_max <= sqrt(solver.opts.constraint_tolerance) && (dJ <= sqrt(solver.opts.cost_tolerance) || gradient <= sqrt(solver.opts.gradient_tolerance))
        #         sqrt_tolerance = true
        #     end
        # end

        # update multiplier and penalty terms
        outer_loop_update(results,solver,false)

        #****************************#
        #    TERMINATION CRITERIA    #
        #****************************#
        # Check if maximum constraint violation satisfies termination criteria AND cost or gradient tolerance convergence
        if is_constrained
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

    if is_constrained
        # use_static ? results = ConstrainedResults(results) : nothing
    else
        # use_static ? results = UnconstrainedResults(results) : nothing
        solver.opts.iterations_outerloop = iterations_outerloop_original
    end

    # Run Stats
    stats = Dict("iterations"=>iter,
                 "major iterations"=>iter_outer,
                 "runtime"=>float(time_ns() - t_solve_start)/1e9,
                 "setup_time"=>float(time_setup)/1e9,
                 "cost"=>J_hist,
                 "c_max"=>c_max_hist)

    if ((iter_outer == solver.opts.iterations_outerloop) && (iter_inner == solver.opts.iterations)) && solver.opts.verbose
        println("*Solve reached max iterations*")
    end

    ## Return dynamically feasible trajectory
    if infeasible #&& solver.opts.solve_feasible
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
                solver_feasible = Solver(solver.model,obj_uncon,integration=solver.integration,dt=solver.dt,opts=solver.opts)
            else
                solver_feasible = solver
            end

            # resolve feasible problem with warm start
            results_feasible, stats_feasible = solve(solver_feasible,to_array(results_feasible.U),prevResults=results_feasible)

            # merge stats
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
        return results_feasible, stats

    # if feasible solve, return results
    else
        if solver.opts.verbose
            println("***Solve Complete***")
        end

        return results, stats
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
        results_feasible = new_constrained_results(results_feasible,solver,results.LAMBDA,results.λN,results.ρ)
        update_constraints!(results_feasible,solver,results_feasible.X,results_feasible.U)
        calculate_jacobians!(results_feasible,solver)
    end

    return results_feasible
end

"""
@(SIGNATURES)
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
    if solver.control_integration == :zoh && !second_order
        for k = 1:solver.N
            λ_update_1_zoh!(results,solver,k)
        end
    elseif solver.control_integration == :zoh && second_order
        for k = 1:solver.N
            λ_update_2_zoh!(results,solver,k)
        end
    end

    # FOH
    if solver.control_integration == :foh && !second_order
        for k = 1:solver.N
            λ_update_1_zoh!(results,solver,k)
        end
    elseif solver.control_integration == :foh && second_order
        λ_update_2_foh!(results,solver)
    end
end

"""@(SIGNATURES) 1st order multiplier update for zoh (sequential)"""
function λ_update_1_zoh!(results::ConstrainedIterResults,solver::Solver,k::Int64)
    if k != solver.N
        pI = solver.obj.pI
        results.LAMBDA[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.LAMBDA[k] + results.MU[k].*results.C[k]))
        results.LAMBDA[k][1:pI] = max.(0.0,results.LAMBDA[k][1:pI])
    else
        results.λN .= max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λN + results.μN.*results.CN))
    end
    return nothing
end

"""@(SIGNATURES) 1st order multiplier update for zoh (nonsequential)"""
function λ_update_1_zoh!(results::ConstrainedIterResults,solver::Solver)
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
end

"""@(SIGNATURES) 2nd order multiplier update for zoh (sequential)"""
function λ_update_2_zoh!(results::ConstrainedIterResults,solver::Solver,k::Int64)
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

    λ[idx_active] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, λ[idx_active] + B\I*c[idx_active])) #this is a bit mysterious to me, but I was finding on the foh that without the I, the \ was giving different results from using inv(), so I've included the I here as well

    if k != solver.N
        λ[idx_inequality] = max.(0.0,λ[idx_inequality])
    end

    return nothing
end

"""@(SIGNATURES) 2nd order multiplier update for zoh (nonsequential)"""
function λ_update_2_zoh!(results::ConstrainedIterResults,solver::Solver)
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

    λ[idx_active] = λ[idx_active] + B\I*c[idx_active] #this is a bit mysterious to me, but I was finding on the foh that without the I, the \ was giving different results from using inv(), so I've included the I here as well
    λ[idx_inequality] = max.(0.0,λ[idx_inequality])

    # update results
    for k = 1:N-1
        results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
    end
    results.λN .= λ[(N-1)*p+1:(N-1)*p+n]

    return nothing
end

"""@(SIGNATURES) 1st order multiplier update for foh (sequential)"""
function λ_update_1_foh!(results::ConstrainedIterResults,solver::Solver,k::Int64)
    pI = solver.obj.pI
    results.LAMBDA[k] = max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.LAMBDA[k] + results.MU[k].*results.C[k]))
    results.LAMBDA[k][1:pI] = max.(0.0,results.LAMBDA[k][1:pI])

    if k == solver.N
        results.λN .= max.(solver.opts.λ_min, min.(solver.opts.λ_max, results.λN + results.μN.*results.CN))
    end
    return nothing
end

"""@(SIGNATURES) 2nd order multiplier update for foh (nonsequential)"""
function λ_update_2_foh!(results::ConstrainedIterResults,solver::Solver)
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
        Ad, Bd, Cd = results.fx[k], results.fu[k], results.fv[k]

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

    if solver.opts.λ_second_order_update
        λ[idx_active] = λ[idx_active] + B\I*c[idx_active] # this is a bit mysterious to me, but I was finding on the foh that without the I, the \ was giving different results from using inv(), so I've included the I here
        λ[idx_inequality] = max.(0.0,λ[idx_inequality])
    end

    # update results
    for k = 1:N
        results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
    end
    results.λN .= λ[N*p+1:N*p+n]

    return nothing
end

""" @(SIGNATURES) Penalty update """
function μ_update!(results::ConstrainedIterResults,solver::Solver)
    if solver.opts.outer_loop_update == :default
        μ_update_default!(results,solver)
    elseif solver.opts.outer_loop_update == :individual
        μ_update_individual!(results,solver)
    elseif solver.opts.outer_loop_update == :sequential
        μ_update_sequential!(results,solver,:post)
    end
    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('default') - all penalty terms are updated"""
function μ_update_default!(results::ConstrainedIterResults,solver::Solver)
    # println("default penalty update")
    if solver.control_integration == :foh
        final_index = solver.N
    else
        final_index = solver.N-1
    end

    for k = 1:final_index
        results.MU[k] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[k])
    end

    results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)

    return nothing
end

"""
@(SIGNATURES)
    Penalty update scheme ('individual')- all penalty terms are updated according to ALGENCAN heuristic
    -note that
"""
function μ_update_sequential!(results::ConstrainedIterResults,solver::Solver,status::Symbol)
    p = length(results.C[1])
    pI = solver.obj.pI
    n = solver.model.n

    if solver.control_integration == :foh
        final_index = solver.N
    else
        final_index = solver.N-1
    end

    if status == :pre
        results.V_al_prev .= deepcopy(results.V_al_current)

        for k = 1:final_index
            for i = 1:p
                # inequality constraints
                if i <= pI
                    # calculate term for penalty update (see ALGENCAN ref.)
                    results.V_al_current[i,k] = min(-1.0*results.C[k][i], results.LAMBDA[k][i]/results.MU[k][i])
                end
            end
        end

    elseif status == :post
        # println("sequential penalty update")
        v1 = max(sqrt(norm2(results.C,pI+1:p) + norm2(results.CN)), norm(results.V_al_current))
        v2 = max(sqrt(norm2(results.C_prev,pI+1:p) + norm2(results.CN_prev)), norm(results.V_al_prev))

        if v1 <= solver.opts.τ*v2
            for k = 1:final_index
                results.MU[k] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[k])
            end

            results.μN .= min.(solver.opts.μ_max, solver.opts.γ_no*results.μN)

            if solver.opts.verbose
                println("no μ update\n")
            end
        else
            for k = 1:final_index
                results.MU[k] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[k])
            end

            results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)

            if solver.opts.verbose
                println("$(solver.opts.γ)x μ update\n")
            end
        end
    end

    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('individual')- all penalty terms are updated uniquely according to indiviual improvement compared to previous iteration"""
function μ_update_individual!(results::ConstrainedIterResults,solver::Solver)
    # println("individual penalty update")
    p = length(results.C[1])
    pI = solver.obj.pI
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
                    results.MU[k][i] = min(μ_max, γ_no*results.MU[k][i])
                else
                    results.MU[k][i] = min(μ_max, γ*results.MU[k][i])
                end
            else
                if abs(results.C[k][i]) <= τ*abs(results.C_prev[k][i])
                    results.MU[k][i] = min(μ_max, γ_no*results.MU[k][i])
                else
                    results.MU[k][i] = min(μ_max, γ*results.MU[k][i])
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

    # store metrics for sequential update
    if solver.opts.outer_loop_update == :sequential
        μ_update_sequential!(results,solver,:pre)
    end

    ## Lagrange multiplier updates
    λ_update!(results,solver,false)

    ## Penalty updates
    μ_update!(results,solver)

    ## Store current constraints evaluations for next outer loop update
    results.C_prev .= deepcopy(results.C)
    results.CN_prev .= deepcopy(results.CN)

    return nothing
end

# function outer_loop_update(results::ConstrainedIterResults,solver::Solver,sqrt_tolerance::Bool=false)::Nothing
#
#     n,m,N = get_sizes(solver)
#     p = length(results.C[1])  # number of constraints
#     pI = solver.obj.pI  # number of inequality constraints
#
#     if solver.control_integration == :foh
#         final_index = N
#     else
#         final_index = N-1
#     end
#
#     # store previous term for penalty update
#     if solver.opts.outer_loop_update == :uniform
#         results.V_al_prev .= deepcopy(results.V_al_current)
#     end
#
#     ### Lagrange multiplier updates ###
#     for jj = 1:final_index
#         for ii = 1:p
#             # inequality constraints
#             if ii <= pI
#                 # calculate term for penalty update (see ALGENCAN ref.)
#                 if solver.opts.outer_loop_update == :uniform
#                     results.V_al_current[ii,jj] = min(-1.0*results.C[jj][ii], results.LAMBDA[jj][ii]/results.MU[jj][ii])
#                 end
#
#                 # penalty update for 'individual' scheme
#                 if  solver.opts.outer_loop_update == :individual
#                     if max(0.0,results.C[jj][ii]) <= solver.opts.τ*max(0.0,results.C_prev[jj][ii])
#                         results.MU[jj][ii] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[jj][ii])
#                     else
#                         results.MU[jj][ii] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[jj][ii])
#                     end
#                 end
#
#             # equality constraints
#             else
#                 # penalty update for 'individual' scheme
#                 if  solver.opts.outer_loop_update == :individual
#                     if abs(results.C[jj][ii]) <= solver.opts.τ*abs(results.C_prev[jj][ii])
#                         results.MU[jj][ii] = min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[jj][ii])
#                     else
#                         results.MU[jj][ii] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[jj][ii])
#                     end
#                 end
#             end
#         end
#     end
#
#     λ_update!(results,solver,solver.opts.λ_second_order_update)
#
#
#
#     ###################################
#
#     ### Penalty updates ###
#     # 'default' penaltiy update - all penalty terms are updated (no conditions)
#     if solver.opts.outer_loop_update == :default
#         for k = 1:N
#             results.MU[k] = min.(solver.opts.μ_max, solver.opts.γ*results.MU[k])
#         end
#         results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)
#     end
#
#     # 'uniform' penalty update - see ALGENCAN reference
#     if solver.opts.outer_loop_update == :uniform
#         v1 = max(sqrt(norm2(results.C,pI+1:p) + norm2(results.CN)), norm(results.V_al_current))
#         v2 = max(sqrt(norm2(results.C_prev,pI+1:p) + norm2(results.CN_prev)), norm(results.V_al_prev))
#
#         if v1 <= solver.opts.τ*v2
#             for k = 1:N
#                 results.MU[k] .= min.(solver.opts.μ_max, solver.opts.γ_no*results.MU[k])
#             end
#             results.μN .= min.(solver.opts.μ_max, solver.opts.γ_no*results.μN)
#             if solver.opts.verbose
#                 println("no μ update\n")
#             end
#         else
#             for k = 1:N
#                 results.MU[k] .= min.(solver.opts.μ_max, solver.opts.γ*results.MU[k])
#             end
#             results.μN .= min.(solver.opts.μ_max, solver.opts.γ*results.μN)
#             if solver.opts.verbose
#                 println("$(solver.opts.γ)x μ update\n")
#             end
#         end
#     end
#
#     # 'individual' penalty update (only terminal constraints left to update)
#     if solver.opts.outer_loop_update == :individual
#         # TODO: handle general terminal constraints
#         for ii = 1:n
#             if abs(results.CN[ii]) <= solver.opts.τ*abs(results.CN_prev[ii])
#                 results.μN[ii] = min.(solver.opts.μ_max, solver.opts.γ_no*results.μN[ii])
#             else
#                 results.μN[ii] = min.(solver.opts.μ_max, solver.opts.γ*results.μN[ii])
#             end
#         end
#     end
#     #######################
#
#     ## Store current constraints evaluations for next outer loop update
#     results.C_prev .= deepcopy(results.C)
#     results.CN_prev .= deepcopy(results.CN)
#
#     return nothing
# end



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
