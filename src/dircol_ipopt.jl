
function init_jacobians(solver,method)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    if method == :trapezoid || method == :hermite_simpson_separated
        A = zeros(n,n+m̄,N_)
        B = zeros(0,0,N_)
    else
        A = zeros(n,n,N_)
        B = zeros(n,m,N_)
    end
    return A,B
end

function get_nG(solver::Solver,method::Symbol)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver,method)
    m̄, = get_num_controls(solver)
    if method == :trapezoid || method == :hermite_simpson
        return 2(n+m̄)*(N-1)n
    elseif method == :hermite_simpson_separated
        return 3(n+m̄)*(N-1)n
    elseif method == :midpoint
        return (2n+m̄)*(N-1)n
    end
end

function gen_usrfun_ipopt(solver::Solver,method::Symbol)
    N,N_ = TrajectoryOptimization.get_N(solver,method)
    n,m = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    NN = N*(n+m̄)

    # Initialize Variables
    fVal = zeros(n,N)
    gX_,gU_,fVal_ = init_traj_points(solver,fVal,method)
    weights = get_weights(method,N_)
    A,B = init_jacobians(solver,method)

    #################
    # COST FUNCTION #
    #################
    function eval_f(Z)
        # vars = DircolVars(Z,n,m̄,N)
        # X,U = vars.X, vars.U
        X,U = TrajectoryOptimization.unpackZ(Z,(n,m̄,N))
        fVal = zeros(eltype(Z),n,N)
        X_ = zeros(eltype(Z),n,N_)
        U_ = zeros(eltype(Z),m̄,N_)
        X_,U_ = get_traj_points(solver,X,U,fVal,X_,U_,method,true)
        J = cost(solver,X_,U_,method)
        return J
    end

    ###########################
    # COLLOCATION CONSTRAINTS #
    ###########################
    function eval_g(Z, g)
        # vars = DircolVars(Z,n,m,N)
        # X,U = vars.X,vars.U
        X,U = unpackZ(Z,(n,m̄,N))
        fVal = zeros(eltype(Z),n,N)
        X_ = zeros(eltype(Z),n,N_)
        U_ = zeros(eltype(Z),m̄,N_)
        fVal_ = zeros(eltype(Z),n,N_)
        X_,U_ = get_traj_points(solver,X,U,fVal,X_,U_,method)
        get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
        collocation_constraints!(solver::Solver, X_, U_, fVal_, g, method::Symbol)
        # reshape(g,n*(N-1),1)
        return nothing
    end

    #################
    # COST GRADIENT #
    #################
    function eval_grad_f(Z, grad_f)
        vars = DircolVars(Z,n,m̄,N)
        X,U = vars.X, vars.U
        X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method)
        get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
        update_jacobians!(solver,X_,U_,A,B,method,true)
        cost_gradient!(solver, X_, U_, fVal, A, B, weights, grad_f, method)
        return nothing
    end

    #######################
    # CONSTRAINT JACOBIAN #
    #######################
    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure  # TODO: Do this in place
            r,c = constraint_jacobian_sparsity(solver,method)
            rows .= r
            cols .= c
        else
            vars = DircolVars(Z,n,m̄,N)
            X,U, = vars.X,vars.U
            X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method)
            get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
            update_jacobians!(solver,X_,U_,A,B,method)
            constraint_jacobian!(solver,X_,U_,fVal_,A,B,vals,method)
        end
        return nothing
    end

    return eval_f, eval_g, eval_grad_f, eval_jac_g

end

function solve_ipopt(solver::Solver, X0::Matrix{Float64}, U0::Matrix{Float64}, method::Symbol)

    # Get Constants
    N,N_ = TrajectoryOptimization.get_N(solver,method)  # N=>Number of time steps for decision variables (may differ from solver.N)
    n,m = get_sizes(solver)                             # N_=>Number of time steps for "trajectory" or integration points
    NN = N*(n+m)                                        # Total number of decision variables
    nG = TrajectoryOptimization.get_nG(solver,method)   # Number of nonzero entries in Constraint Jacobian
    nH = 0                                              # Number of nonzeros entries in Hessian

    # Pack the variables
    vars0 = DircolVars(X0,U0)
    Z0 = vars0.Z

    # Generate functions
    eval_f, eval_g, eval_grad_f, eval_jac_g = gen_usrfun_ipopt(solver,method)

    # Get Bounds
    x_L, x_U, g_L, g_U = get_bounds(solver,method)
    P = length(g_L)  # Total number of constraints

    # Create Problem
    prob = Ipopt.createProblem(NN, x_L, x_U, P, g_L, g_U, nG, nH,
        eval_f, eval_g, eval_grad_f, eval_jac_g)  # Ipopt.jl method
    prob.x =  Z0  # Set initial state

    # Set options
    dir = root_dir()
    opt_file = joinpath(dir,"ipopt.opt")
    addOption(prob,"option_file_name",opt_file)
    if solver.opts.verbose == false
        addOption(prob,"print_level",0)
    end


    # Solve
    t_eval = @elapsed status = solveProblem(prob)
    stats = parse_ipopt_summary()
    stats["info"] = Ipopt.ApplicationReturnStatus[status]
    stats["runtime"] = t_eval
    vars = DircolVars(prob.x,n,m,N)
    return vars, stats, prob
end



"""
$(SIGNATURES)
Extract important information from the SNOPT output file(s)
"""
function parse_ipopt_summary(file=joinpath(root_dir(),"logs","ipopt.out"))
    props = Dict()
    obj = Vector{Float64}()
    c_max = Vector{Float64}()
    iter_lines = false  # Flag true if it's parsing the iteration summary lines


    function stash_prop(ln::String,prop::String,prop_name::String=prop,vartype::Type=Float64)
        if occursin(prop,ln)
            loc = findfirst(prop,ln)
            if vartype <: Real
                val = convert(vartype,parse(Float64,split(ln)[end]))
                props[prop_name] = val
                return true
            end
        end
        return false
    end

    function store_itervals(ln::String)
        if iter_lines
            vals = split(ln)
            if length(vals) == 10 && vals[1] != "iter" && vals[1] != "Restoration"
                push!(obj, parse(Float64,vals[2]))
                push!(c_max, parse(Float64,vals[3]))
            end
        end
    end


    open(file) do f
        for ln in eachline(f)
            stash_prop(ln,"Number of Iterations","iterations",Int64) ? iter_lines = false : nothing
            stash_prop(ln,"Total CPU secs in IPOPT (w/o function evaluations)","self time")
            stash_prop(ln,"Total CPU secs in NLP function evaluations","function time")
            stash_prop(ln,"Number of objective function evaluations","objective calls",Int64)
            length(ln) > 0 && split(ln)[1] == "iter" && iter_lines == false ? iter_lines = true : nothing
            store_itervals(ln)
        end
    end
    props["cost"] = obj
    props["c_max"] = c_max
    return props
end

function write_ipopt_options(
        optfile=joinpath(root_dir(),"ipopt.opt"),
        outfile=joinpath(root_dir(),"logs","ipopt.out"))
    f = open(optfile,"w")
    println(f,"# IPOPT Options for TrajectoryOptimization.jl\n")
    println(f,"# Use Quasi-Newton methods to avoid the need for the Hessian")
    println(f,"hessian_approximation limited-memory\n")
    println(f,"# Output file")
    println(f,"file_print_level 5")
    # println(f,"output_file $(wrap_quotes(outfile))")
    println(f,"output_file"*" "*"\""*"$(outfile)"*"\"")
    close(f)
end

function wrap_quotes(s::String)
    "\"" * s * "\""
end







#
# function eval_f2(Z)
#     vars = DircolVars(Z,n,m,N)
#     X,U = vars.X, vars.U
#     cost(solver,X,U,weights)
# end
#
# function eval_f3(Z)
#     results.Z .= Z
#     vars = DircolVars(Z,n,m,N)
#     update_derivatives!(solver,results,method)
#     get_traj_points!(solver,results,method)
#     cost(solver,results)
# end
#
#
# function eval_g1(Z, g)
#     X,U = unpackZ(Z)
#     g = reshape(g,n,N-1)
#     f1, f2 = zeros(n),zeros(n)
#     for k = 1:N-1
#         solver.fc(f1,X[:,k+1],U[:,k+1])
#         solver.fc(f2,X[:,k],U[:,k])
#         g[:,k] = dt*( f1 + f2 )/2 - X[:,k+1] + X[:,k]
#     end
#     # reshape(g,n*(N-1),1)
#     return nothing
# end
#
# function eval_g2(Z, g)
#     results.Z .= Z
#     update_derivatives!(solver,results,method)
#     get_traj_points!(solver,results,method)
#     get_traj_points_derivatives!(solver,results,method)
#     g .= collocation_constraints(solver::Solver,results, method::Symbol)
#     # reshape(g,n*(N-1),1)
#     return nothing
# end
#
# function eval_grad_f1(Z, grad_f)
#     results.Z .= Z
#     update_derivatives!(solver,results,method)
#     get_traj_points!(solver,results,method)
#     get_traj_points_derivatives!(solver,results,method)
#     grad_f .= cost_gradient(solver,results,method)
#     return nothing
# end
#
# function eval_jac_g1(Z, mode, rows, cols, vals)
#     if mode == :Structure
#         constraint_jacobian_sparsity(method,solver)
#     else
#         vars = DircolVars(Z,n,m,N)
#         X,U, = vars.X,vars.U
#         X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method)
#         get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
#         update_jacobians!(solver,X_,U_,A,B,method)
#         jacob_g = constraint_jacobian(solver,X_,U_,A,B,method)
#     end
# end
