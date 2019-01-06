
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
        nC = 2(n+m̄)*(N-1)n
    elseif method == :hermite_simpson_separated
        nC = 3(n+m̄)*(N-1)n
    elseif method == :midpoint
        nC = (2n+m̄)*(N-1)n
    end
    solver.state.minimum_time ? nE = 2(N-2) : nE = 0

    # Custom constraints
    pI,pE = count_constraints(solver.obj, :custom)
    nP = (N-1)*(n+m̄)*(pE[1] + pI[1]) + n*(pE[2] + pI[2])

    nG = nC + nE + nP
    return nG, (collocation=nC, dt=nE, custom=nP)
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

    # Generate custom constraint functions
    if solver.state.constrained
        custom_constraints!, custom_constraint_jacobian!, custom_jacobian_sparsity = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
        pI_c, pE_c = count_constraints(solver.obj, :custom)
        PI_c = (N-1)pI_c[1] + pI_c[2]  # Total custom inequality constraints
        PE_c = (N-1)pE_c[1] + pE_c[2]  # Total custom equality constraints
        P_c = PI_c + PE_c              # Total custom constraints

        # Partition constraints
        n_colloc = (N-1)n  # Number of collocation constraints
        g_colloc = 1:n_colloc           # Collocation constraints
        g_custom = n_colloc.+(1:P_c)    # Custom constraints
        g_dt = (n_colloc+P_c).+(1:N-2)  # dt constraints (minimum time)

        nG,Gpart = get_nG(solver,method)
        jac_g_colloc = 1:Gpart.collocation
        jac_g_custom = Gpart.collocation .+ (1:Gpart.custom)
        jac_g_dt = Gpart.collocation + Gpart.custom + 1:nG
    end



    #################
    # COST FUNCTION #
    #################
    function eval_f(Z)
        # vars = DircolVars(Z,n,m̄,N)
        # X,U = vars.X, vars.U
        X,U = unpackZ(Z,(n,m̄,N))
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
        get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal, method)
        collocation_constraints!(solver::Solver, X_, U_, fVal_, view(g,g_colloc), method::Symbol)
        if solver.state.constrained
            custom_constraints!(view(g,g_custom),X,U)
            if solver.state.minimum_time
                dt_constraints!(solver, view(g,g_dt), view(U,m̄,1:N))
            end
        end
        return nothing
    end

    #################
    # COST GRADIENT #
    #################
    function eval_grad_f(Z, grad_f)
        # vars = DircolVars(Z,n,m,N)
        # X,U = vars.X,vars.U
        X,U = unpackZ(Z,(n,m̄,N))
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
            nG,Gpart = get_nG(solver,method)
            jacob_colloc = collocation_constraint_jacobian_sparsity(solver,method)
            jacob_custom = custom_jacobian_sparsity(Gpart.collocation)
            jacob_dt = time_step_constraint_jacobian_sparsity(solver,Gpart.collocation + Gpart.custom)
            jacob_g = [jacob_colloc;
                       jacob_custom;
                       jacob_dt]

            r,c,inds = findnz(jacob_g)
            v = sortperm(inds)
            rows .= r[v]
            cols .= c[v]
        else
            # vars = DircolVars(Z,n,m,N)
            # X,U = vars.X,vars.U
            X,U = unpackZ(Z,(n,m̄,N))
            X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method)
            get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
            update_jacobians!(solver,X_,U_,A,B,method)
            collocation_constraint_jacobian!(solver, X_,U_,fVal_, A,B, view(vals, jac_g_colloc), method)
            if solver.state.constrained
                custom_constraint_jacobian!(view(vals, jac_g_custom), X,U)
            end
            if solver.state.minimum_time
                time_step_constraint_jacobian!(view(vals, jac_g_dt), solver)
            end
        end
        return nothing
    end

    return eval_f, eval_g, eval_grad_f, eval_jac_g

end

function solve_ipopt(solver::Solver, X0::Matrix{Float64}, U0::Matrix{Float64}, method::Symbol; options::Dict{String,T}=Dict{String,Any}()) where T
    X0 = copy(X0)
    U0 = copy(U0)


    # Get Constants
    N,N_ = TrajectoryOptimization.get_N(solver,method)  # N=>Number of time steps for decision variables (may differ from solver.N)
    n,m = get_sizes(solver)                             # N_=>Number of time steps for "trajectory" or integration points
    m̄, = get_num_controls(solver)
    NN = N*(n+m̄)                                        # Total number of decision variables
    nG, = TrajectoryOptimization.get_nG(solver,method)   # Number of nonzero entries in Constraint Jacobian
    nH = 0                                              # Number of nonzeros entries in Hessian

    # Pack the variables
    Z0 = packZ(X0,U0)
    @assert length(Z0) == NN

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
    for (opt,val) in pairs(options)
        addOption(prob,opt,val)
    end
    if solver.opts.verbose == false
        addOption(prob,"print_level",0)
    end


    # Solve
    t_eval = @elapsed status = solveProblem(prob)
    stats = parse_ipopt_summary()
    stats["info"] = Ipopt.ApplicationReturnStatus[status]
    stats["runtime"] = t_eval
    vars = DircolVars(prob.x,n,m̄,N)

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
            if length(vals) == 10 && vals[1] != "iter" && vals[1] != "Restoration" && vals[2] != "iteration"
                push!(obj, parse(Float64,vals[2]))
                push!(c_max, parse(Float64,vals[3]))
            end
        end
    end


    open(file) do f
        for ln in eachline(f)
            stash_prop(ln,"Number of Iterations..","iterations",Int64) ? iter_lines = false : nothing
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
