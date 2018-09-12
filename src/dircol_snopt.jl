
# Evaluate the equality constraint function
function eval_ceq!(gE::Base.ReshapedArray,gE_N::SubArray,solver::Solver,res::DircolResults)
    N = solver.N
    pE = size(gE,1)
    pE_N = size(gE_N,1)

    # Custom constraints
    X,U = res.X,res.U
    if pE > 0
        for k = 1:N-1
            gE[:,k] = obj.cE(X[:,k],U[:,k])
        end
    end
    if pE_N > 0
        gE_N .= obj.cE(X[:,N])
    end
end

function eval_c!(gI::Base.ReshapedArray,gI_N::SubArray,solver::Solver,res::DircolResults)
    N = solver.N
    pI = size(gI,1)
    pI_N = size(gI_N,1)

    # Custom constraints
    X,U = res.X,res.U
    if pI > 0
        for k = 1:N-1
            gE[:,k] = obj.cI(X[:,k],U[:,k])
        end
    end
    if pI_N > 0
        gE_N .= obj.cI(X[:,N])
    end
end

"""
$(SIGNATURES)
Generate the custom function to be passed into SNOPT, as well as `eval_f` and
`eval_g` used to calculate the objective and constraint functions.

# Arguments
* model: TrajectoryOptimization.Model for the dynamics
* obj: TrajectoryOptimization.ConstrainedObjective to describe the cost function
* dt: time step
* pack: tuple of important sizes (n,m,N) -> (num states, num controls, num knot points)
* method: Collocation method. Either :trapezoid or :hermite_simpson_separated
* grads: Specifies the gradient information provided to SNOPT.
    :none - returns no gradient information to SNOPT
    :auto - uses ForwardDiff to calculate gradients
    :quadratic - uses functions exploiting quadratic cost functions
"""
function gen_usrfun(solver::Solver, results::DircolResults, method::Symbol; grads=:none)::Function
    n,N = size(results.X)
    m,N_ = size(results.U_)
    NN = (n+m)N  # Number of decision variables
    dt = solver.dt

    # Count constraints
    pI = 0
    pE = (N-1)*n # Collocation constraints
    p_colloc = pE

    pI_obj, pE_obj = count_constraints(solver.obj)
    pI_c,   pE_c   = pI_obj[2], pE_obj[2]  # Number of custom stage constraints
    pI_N_c, pE_N_c = pI_obj[4], pE_obj[4]  # Number of custom terminal constraints

    pI += pI_c*(N-1) + pI_N_c  # Add custom inequality constraints
    pE += pE_c*(N-1) + pE_N_c  # Add custom equality constraints

    # Allocate Arrays
    c = zeros(pI)
    ceq = zeros(pE)

    g_colloc = view(ceq,1:p_colloc)
    gE = reshape(view(ceq,p_colloc+(1:pI_c*(N-1))),pI_c,N-1)
    gE_N = view(ceq,p_colloc+pI_c*(N-1)+(1:pE_N_c))

    gI = reshape(view(c,1:pI_c*(N-1)),pI_c,N-1)
    gI_N = view(c,pE_c+(1:pI_N_c))

    # User defined function passed to SNOPT (via Snopt.jl)
    function usrfun(Z::Vector{Float64})
        results.Z .= Z

        # Update Derivatives
        update_derivatives!(solver,results,method)

        # Get points used in integration
        get_traj_points!(solver,results,method)
        get_traj_points_derivatives!(solver,results,method)

        # Objective Function (Cost)
        J = cost(solver,results)

        # Collocation Constraints
        g_colloc .= collocation_constraints(solver, results, method)
        eval_ceq!(gE,gE_N,solver,results) # No inequality constraints for now
        eval_c!(gI,gI_N,solver,results)

        fail = false

        if grads == :none
            return J, c, ceq, fail
        else
            if pE_c > 0 || pE_N_c > 0 || pI_c > 0 || pI_N_c > 0
                error("Gradients not defined for custom constraints yets")
            end

            # Calculate dynamics jacobians at integration points
            update_jacobians!(solver,results,method)

            # Gradient of Objective
            grad_f = cost_gradient(solver,results,method)

            # Constraint Jacobian
            jacob_ceq = constraint_jacobian(solver,results,method)
            jacob_c = Float64[]

            return J, c, ceq, grad_f, jacob_c, jacob_ceq, fail
        end


    end

    function usrfun(Z::Vector{Float64},mode::Symbol)
        if mode == :ceq
            row,col = constraint_jacobian_sparsity(solver,method)
        else
            row,col = [],[]
        end
        return row,col
    end

    # Specify sparsity structure
    # gceq = constraint_jacobian_sparsity(solver,results,method)
    # gceq_c = spzeros(pE_c,NN)
    # gceq_c[:,1:NN-(n+m)] = 1
    # gceq_N = spzeros(pE_N_c,NN)
    # gceq_N[:,NN-(n+m)+1:end] = 1
    # gceq = [gceq; gceq_c; gceq_N]
    #
    # gc_c = spzeros(pI_c,NN)
    # gc_c[:,1:NN-(n+m)] = 1
    # gc_N = spzeros(pI_N_c,NN)
    # gc_N[:,NN-(n+m)+1:end] = 1
    # gc = [gc_c; gc_N]
    #
    # function usrfun(s::Symbol)
    #     if s == :gceq
    #         return gceq
    #     elseif s == :gc
    #         return gc
    #     end
    # end

    return usrfun
end


"""
$(SIGNATURES)
Solve a trajectory optimization problem with direct collocation

# Arguments
* model: TrajectoryOptimization.Model for the dynamics
* obj: TrajectoryOptimization.ConstrainedObjective to describe the cost function
* dt: time step. Used to determine the number of knot points. May be modified
    by the solver in order to achieve an integer number of knot points.
* method: Collocation method.
    :midpoint - Zero order interpolation on states and controls.
    :trapezoidal - First order interpolation on states and zero order on control.
    :hermite_simpson_separated - Hermite Simpson collocation with the midpoints
        included as additional decision variables with constraints
    :hermite_simpson - condensed verision of Hermite Simpson. Currently only
        supports grads=:auto or grads=:none (recommended)
* grads: Specifies the gradient information provided to SNOPT.
    :none - returns no gradient information to SNOPT
    :auto - uses ForwardDiff to calculate gradients
    :quadratic - uses functions exploiting quadratic cost functions
"""
function solve_snopt(solver::Solver,X0::Matrix,U0::Matrix;
        method::Symbol=:auto, grads::Symbol=:quadratic, start=:cold)

    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)

    # Create results structure
    results = DircolResults(n,m,solver.N,method)
    var0 = DircolVars(X0,U0)
    Z0 = var0.Z

    # Generate the objective/constraint function and its gradients
    usrfun = gen_usrfun(solver, results, method, grads=grads)

    # Set up the problem
    lb,ub = get_bounds(solver,method)

    # Set options
    options = Dict{String, Any}()
    options["Derivative option"] = 0
    options["Verify level"] = 1
    options["Minor feasibility tol"] = solver.opts.eps_constraint
    # options["Minor optimality  tol"] = solver.opts.eps_intermediate
    options["Major optimality  tol"] = solver.opts.eps

    # Solve the problem
    if solver.opts.verbose
        println("DIRCOL with $method")
        println("Passing Problem to SNOPT...")
    end
    row,col = constraint_jacobian_sparsity(solver,method)
    prob = Snopt.createProblem(usrfun, Z0, lb, ub, iE=row, jE=col)
    # prob = Snopt.createProblem(usrfun, Z0, lb, ub)
    prob.x = Z0
    t_eval = @elapsed z_opt, fopt, info = snopt(prob, options, start=start)
    stats = parse_snopt_summary()
    stats["info"] = info
    stats["runtime"] = t_eval
    # @time snopt(usrfun, Z0, lb, ub, options, start=start)
    # xopt, fopt, info = Z0, Inf, "Nothing"
    sol = DircolVars(z_opt,n,m,N)

    if solver.opts.verbose
        println(info)
    end
    return sol, stats, prob
end

"""
$(SIGNATURES)
MESH REFINEMENT:
Solve by warm starting with a coarse time step and warm-starting the solver
with the previous solution

# Arguments
* mesh: vector of step sizes for refinement prior to step size specified by solver.
"""
function solve_dircol(solver::Solver, X0::Matrix, U0::Matrix, mesh::Vector;
        method::Symbol=:auto, grads::Symbol=:quadratic, start=:cold)
    x_opt, u_opt = X0, U0
    stats = Dict{String,Any}("iterations"=>0, "major iterations"=>0, "objective calls"=>0, "info"=>Vector{String}())
    tic()
    for dt in mesh
        solver.opts.verbose ? println("Refining mesh at dt=$dt") : nothing
        solver_mod = Solver(solver,dt=dt)
        x_int, u_int = interp_traj(solver_mod.N, solver.obj.tf, x_opt, u_opt)
        x_opt, u_opt, f_opt, stats_run = solve_dircol(solver_mod, x_int, u_int, method=method, grads=grads, start=start)
        for key in keys(stats)
            if key == "info"
                push!(stats["info"], stats_run["info"])
            else
                stats[key] += stats_run[key]
            end
        end
        start = :warm  # Use warm starts after the first one
    end

    # Run the original time step
    x_int, u_int = interp_traj(solver.N, solver.obj.tf, x_opt, u_opt)
    x_opt, u_opt, f_opt, stats_run = solve_dircol(solver, x_int, u_int, method=method, grads=grads, start=:warm)
    for key in keys(stats)
        if key == "info"
            push!(stats["info"], stats_run["info"])
        else
            stats[key] += stats_run[key]
        end
    end
    stats["runtime"] = toq()
    return x_opt, u_opt, f_opt, stats
end

function solve_dircol(solver::Solver, mesh::Vector;
        method::Symbol=:auto, grads::Symbol=:quadratic, start=:cold)
    # Constants
    N = solver.N
    N = convert_N(N,method)

    X0, U0 = get_initial_state(solver.obj,N)
    solve_dircol(solver, X0, U0, mesh, method=method, grads=grads, start=start)
end

"""
$(SIGNATURES)
Extract important information from the SNOPT output file(s)
"""
function parse_snopt_summary(file="snopt-summary.out")
    props = Dict()

    function stash_prop(ln::String,prop::String,prop_name::String=prop)
        if contains(ln, prop)
            loc = search(ln,prop)
            val = Int(float(split(ln[loc[end]+1:end])[1]))
            props[prop_name] = val
        end
    end

    open(file) do f
        for ln in eachline(f)
            stash_prop(ln,"No. of iterations","iterations")
            stash_prop(ln,"No. of major iterations","major iterations")
            stash_prop(ln,"No. of calls to funobj","objective calls")
        end
    end
    return props
end
