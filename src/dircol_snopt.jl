
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
    function usrfun(Z)
        results.Z .= Z

        # Update Derivatives
        update_derivatives!(solver,results,method)

        # Get points used in integration
        get_traj_points!(solver,results,method)

        # Calculate dynamics jacobians at integration points
        update_jacobians!(solver,results,method)

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

            # Gradient of Objective
            grad_f = cost_gradient(solver,results,method)

            # Constraint Jacobian
            jacob_ceq = constraint_jacobian(solver,results,method)
            jacob_c = Float64[]

            return J, c, ceq, grad_f, jacob_c, jacob_ceq, fail
        end
    end

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
function solve_dircol(solver::Solver,X0::Matrix,U0::Matrix;
        method::Symbol=:auto, grads::Symbol=:quadratic, start=:cold)

    if solver.obj isa UnconstrainedObjective
        solver = Solver(solver,obj=ConstrainedObjective(solver.obj))
    end

    obj = solver.obj
    model = solver.model

    X0 = copy(X0)
    U0 = copy(U0)

    if method == :auto
        if solver.integration == :rk3_foh
            method = :hermite_simpson
        elseif solver.integration == :midpoint
            method = :midpoint
        else
            method = :hermite_simpson
        end
    end

    # Constants
    n,m,N = get_sizes(solver)
    dt = solver.dt
    N = convert_N(N,method)
    pack = (solver.model.n, solver.model.m, N)

    # Create results structure
    results = DircolResults(n,m,solver.N,method)

    if N != size(X0,2)
        solver.opts.verbose ? println("Interpolating initial guess") : nothing
        X0,U0 = interp_traj(N,obj.tf,X0,U0)
    end

    # Generate the objective/constraint function and its gradients
    usrfun = gen_usrfun(solver, results, method, grads=grads)

    # Set up the problem
    Z0 = packZ(X0,U0)
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
    z_opt, fopt, info = snopt(usrfun, Z0, lb, ub, options, start=start)
    stats = parse_snopt_summary()
    # @time snopt(usrfun, Z0, lb, ub, options, start=start)
    # xopt, fopt, info = Z0, Inf, "Nothing"
    x_opt,u_opt = unpackZ(z_opt,pack)

    if solver.opts.verbose
        println(info)
    end
    return x_opt, u_opt, fopt, stats
end

"""
$(SIGNATURES)
Automatically generate an initial guess by linearly interpolating the state
between initial and final state and settings the controls to zero.
"""
function solve_dircol(solver::Solver;
        method::Symbol=:auto, grads::Symbol=:quadratic, start=:cold)
    # Constants
    N = solver.N
    N = convert_N(N,method)

    X0, U0 = get_initial_state(solver.obj,N)
    solve_dircol(solver, X0, U0, method=method, grads=grads, start=start)
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
    stats = Dict{String,Any}("iterations"=>0, "major iterations"=>0, "objective calls"=>0)
    tic()
    for dt in mesh
        solver.opts.verbose ? println("Refining mesh at dt=$dt") : nothing
        solver_mod = Solver(solver,dt=dt)
        x_int, u_int = interp_traj(solver_mod.N, solver.obj.tf, x_opt, u_opt)
        x_opt, u_opt, f_opt, stats_run = solve_dircol(solver_mod, x_int, u_int, method=method, grads=grads, start=start)
        for key in keys(stats)
            stats[key] += stats_run[key]
        end
        start = :warm  # Use warm starts after the first one
    end
    stats["runtime"] = toq()
    # Run the original time step
    x_int, u_int = interp_traj(solver.N, solver.obj.tf, x_opt, u_opt)
    x_opt, u_opt, f_opt = solve_dircol(solver, x_int, u_int, method=method, grads=grads, start=:warm)
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
