

# Set up
u_bound = 3.
model, obj = TrajectoryOptimization.Dynamics.pendulum!
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false

obj.Q = 1e-3*Diagonal(I,2)
obj.R = 1e-2*Diagonal(I,1)
obj.tf = 5.
model! = TrajectoryOptimization.Model(TrajectoryOptimization.Dynamics.pendulum_dynamics!,2,1) # inplace dynamics
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective

solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts)
U0 = ones(solver.model.m, solver.N)
X0 = line_trajectory(obj.x0, obj.xf,solver.N)

N = solver.N # number of iterations for the solver (ie, knotpoints)
n = solver.model.n # number of states
m = solver.model.m # number of control inputs


#****************************#
#       INITIALIZATION       #
#****************************#

p = solver.obj.p # number of inequality and equality constraints
pI = solver.obj.pI # number of inequality constraints
is_constrained = true
infeasible = false

if infeasible
    ui = infeasible_controls(solver,X0,U0) # generates n additional control input sequences that produce the desired infeasible state trajectory
    m += n # augment the number of control input sequences by the number of states
    p += n # increase the number of constraints by the number of additional control input sequences
    solver.opts.infeasible = true
    U0 = [U0; ui]
else
    solver.opts.infeasible = false
end

if is_constrained
    # res = ConstrainedResults(n,m,p,N) # preallocate memory for results
    reS = TrajectoryOptimization.ConstrainedStaticResults(n,m,p,N)
    reV = TrajectoryOptimization.ConstrainedVectorResults(n,m,p,N)
else
    # res = TrajectoryOptimization.UnconstrainedResults(n,m,N)
    reS = TrajectoryOptimization.UnconstrainedStaticResults(n,m,N)
    reV = TrajectoryOptimization.UnconstrainedVectorResults(n,m,n)
end
length(reS.U[1])
res = reV
# reS = reV


res.X[1] = solver.obj.x0
reS.X[1] = solver.obj.x0

res.MU .*= solver.opts.μ1 # set initial penalty term values
reS.MU .*= solver.opts.μ1 # set initial penalty term values
copyto!(res.U, U0) # initialize control to control input sequence
copyto!(reS.U, U0) # initialize control to control input sequence


# Diagonal indicies for the Iμ matrix (fast)
diag_inds = CartesianIndex.(axes(res.Iμ,1),axes(res.Iμ,2))

# Generate constraint function and jacobian functions from the objective
TrajectoryOptimization.update_constraints!(res,solver,res.X,res.U)
TrajectoryOptimization.update_constraints!(reS,solver,reS.X,reS.U)
res.C == to_array(reS.C)

#****************************#
#           SOLVER           #
#****************************#
# Initial rollout
if infeasible
    copyto!(res.X, X0)
    copyto!(reS.X, X0)
    flag = false
else
    flag = rollout!(res,solver) # rollout new state trajectoy
    flag = rollout!(reS,solver) # rollout new state trajectoy
    to_array(reS.X) == res.X
end

if !flag
    if solver.opts.verbose
        println("Bad initial control sequence, setting initial control to random")
    end
    results.U .= rand(solver.model.m,solver.N)
    rollout!(results,solver)
end
plot(to_array(res.X)')

J_prev = cost(solver, res, res.X, res.U)
J_prev = cost(solver, reS, reS.X, reS.U)

getR(solver)

TrajectoryOptimization.calculate_jacobians!(res, solver)
TrajectoryOptimization.calculate_jacobians!(reS, solver)
to_array(reS.fx) ≈ to_array(res.fx)
to_array(reS.fu) ≈ to_array(res.fu)

backwardpass_mintime!(res, solver)
Δv = backwardpass!(res, solver)
Δv = backwardpass!(reS, solver)
to_array(reS.K) ≈ res.K
to_array(reS.d) ≈ res.d

α = 1.
flag = rollout!(res,solver,α)
flag = rollout!(reS,solver,α)
to_array(reS.X_) ≈ res.X_
to_array(reS.U_) ≈ res.U_
J = cost(solver, res, res.X_, res.U_)
J = cost(solver, reS, reS.X_, reS.U_)

J = forwardpass!(res, solver, Δv)
J = forwardpass!(reS, solver, Δv)
to_array(reS.X_) ≈ res.X_

res.X .= res.X_
reS.X .= deepcopy(reS.X_)

res.U .= res.U_
reS.U .= deepcopy(reS.U_)
to_array(reS.X) ≈ res.X
to_array(reS.U) ≈ res.U

res.Iμ == to_array(reS.Iμ)
res.CN ≈ reS.CN
res.CN ≈ reS.CN
max_violation(res)
max_violation(reS)

# todorov_grad(res)
# todorov_grad(reS)

d_grad = maximum(map((x)->maximum(abs.(x)),reS.d))
d_grad = maximum(abs.(res.d[:]))

s_grad = maximum(abs.(res.s[1]))
s_grad = maximum(abs.(res.s[:,1]))

@btime todorov_grad = mean(map((x)->maximum(x), map((x,y)-> x./(y.+1), map((x)->abs.(x),reS.d), map((x)->abs.(x),reS.U) )))
todorov_grad = mean(maximum(abs.(res.d)./(abs.(res.U) .+ 1),dims=1))


@btime calculate_todorov_gradient(res)
@btime todo_grad(reS)

(map((x)->abs.(x),reS.U) + 1)



solver.opts.verbose = true
outer_loop_update(res,solver)
outer_loop_update(reS,solver)
res.MU ≈ to_array(reS.MU)
res.LAMBDA ≈ to_array(reS.LAMBDA)



solver.opts.use_static
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

infeasible = false
use_static = solver.opts.use_static
use_vector = true

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
        if use_vector
            results = UnconstrainedVectorResults(n,m,N)
        else
            results = UnconstrainedResults(n,m,N)
        end
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
        if use_vector
            results = ConstrainedVectorResults(n,m,p,N)
        else
            results = ConstrainedResults(n,m,p,N) # preallocate memory for results
        end
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
    if use_static || use_vector
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
ctg_gradient = Inf
Δv = Inf

for k = 1:solver.opts.iterations_outerloop
    iter_outer = k
    if solver.opts.verbose
        println("Outer loop $k (begin)")
    end

    if results isa ConstrainedResults || results isa ConstrainedIterResults
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
        if use_static || use_vector
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
        if results isa SolverVectorResults
            d_grad = maximum(map((x)->maximum(abs.(x)),results.d))
            s_grad = maximum(abs.(results.s[1]))
            todorov_grad = mean(map((x)->maximum(x), map((x,y)-> x./y, map((x)->abs.(x),results.d),map((x)->abs.(x),results.U .+ 1.0))))
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
        ctg_gradient = todorov_grad

        if (~is_constrained && gradient < solver.opts.gradient_tolerance) || (results isa ConstrainedResults && ctg_gradient < solver.opts.gradient_intermediate_tolerance && k != solver.opts.iterations_outerloop)
            if solver.opts.verbose
                println("--iLQR (inner loop) cost eps criteria met at iteration: $i\n")
                if results isa UnconstrainedResults
                    println("Unconstrained solve complete")
                end
                println("---Gradient tolerance met")
            end
            break
        # Check for gradient and constraint tolerance convergence
    elseif (is_constrained && ctg_gradient < solver.opts.gradient_tolerance  && c_max < solver.opts.constraint_tolerance)
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
        if max_c < solver.opts.constraint_tolerance && (dJ < solver.opts.cost_tolerance || ctg_gradient < solver.opts.gradient_tolerance)
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
