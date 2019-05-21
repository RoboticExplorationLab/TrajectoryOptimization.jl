using Ipopt

function remove_bounds!(prob::Problem)
    n,m,N = size(prob)
    bounds = [BoundConstraint(n,m) for k = 1:prob.N]

    # Initial Time step
    if :bound ∈ labels(prob.constraints[1])
        bnd_init = remove_bounds!(prob.constraints[1])[1]
    else
        bnd_init = bounds[1]
    end
    bounds[1] = BoundConstraint(n,m, x_min=prob.x0, u_min=bnd_init.u_min,
                                     x_max=prob.x0, u_max=bnd_init.u_max)

    # All time steps
    for k = 2:prob.N
        bnd = remove_bounds!(prob.constraints[k])
        if !isempty(bnd)
            bounds[k] = bnd[1]::BoundConstraint
        end
    end

    # Terminal time step
    if :goal ∈ labels(prob.constraints[N])
        goal = pop!(prob.constraints[N])
        xf = zeros(n)
        evaluate!(xf, goal, zero(xf))
        term_bound = BoundConstraint(n,m, x_min=-xf, u_min=bounds[N-1].u_min,
                                          x_max=-xf, u_max=bounds[N-1].u_max)
        bounds[N] = term_bound::BoundConstraint
    end
    return bounds
end

function get_bounds(prob::Problem, bounds::Vector{<:BoundConstraint})
    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    Z = Primals(prob, true)

    Z.equal ? uN = N : uN = N-1
    x_U = [zeros(n) for k = 1:N]
    x_L = [zeros(m) for k = 1:N]
    u_U = [zeros(m) for k = 1:uN]
    u_L = [zeros(m) for k = 1:uN]
    for k = 1:uN
        x_U[k] = bounds[k].x_max
        x_L[k] = bounds[k].x_min
        u_U[k] = bounds[k].u_max
        u_L[k] = bounds[k].u_min
    end
    if !Z.equal
        x_U = bounds[N].x_max
        x_L = bounds[N].x_min
    end
    z_U = Primals(x_U,u_U)
    z_L = Primals(x_L,u_L)

    # Constraints
    p = num_constraints(prob)
    g_U = [PartedVector(prob.constraints[k]) for k = 1:N-1]
    g_L = [PartedVector(prob.constraints[k]) for k = 1:N-1]
    push!(g_U, PartedVector(prob.constraints[N], :terminal))
    push!(g_L, PartedVector(prob.constraints[N], :terminal))
    for k = 1:N
        if p[k] > 0
            g_L[k].inequality .= -Inf
        end
    end
    g_U = vcat(zeros(p_colloc), g_U...)
    g_L = vcat(zeros(p_colloc), g_L...)

    convertInf!(z_U.Z)
    convertInf!(z_L.Z)
    convertInf!(g_U)
    convertInf!(g_L)
    return z_U.Z, z_L.Z, g_U, g_L
end

function solve(prob::Problem{T,Continuous}, opts::DIRCOLSolverOptions{T}) where T<:AbstractFloat
    prob = copy(prob)
    bnds = remove_bounds!(prob)
    n,m,N = size(prob)
    p = num_constraints(prob)
    p_colloc = num_colloc(prob)
    P = p_colloc + sum(p)
    NN = N*(n+m)
    nG = num_colloc(prob)*2*(n + m) + sum(p[1:N-1])*(n+m) + p[N]*n
    nH = 0

    Z0 = Primals(prob,true)
    z_U, z_L, g_U, g_L = get_bounds(prob, bnds)

    solver = DIRCOLSolver(prob, opts)
    eval_f, eval_g, eval_grad_f, eval_jac_g = TrajectoryOptimization.gen_ipopt_functions(prob, solver)

    problem = createProblem(NN, z_L, z_U, P, g_L, g_U, nG, nH,
        eval_f, eval_g, eval_grad_f, eval_jac_g)

    opt_file = joinpath(TrajectoryOptimization.root_dir(),"ipopt.opt");
    addOption(problem,"option_file_name",opt_file)
    if opts.verbose == false
        addOption(problem,"print_level",0)
    end
    problem.x = copy(Z0.Z)

    # Solve
    t_eval = @elapsed status = solveProblem(problem)
    stats = parse_ipopt_summary()
    stats[:info] = Ipopt.ApplicationReturnStatus[status]
    stats[:runtime] = t_eval
    merge!(solver.stats, stats)

    sol = Primals(problem.x, n, m)
    return sol, solver, problem
end

function gen_ipopt_functions(prob::Problem{T}, solver::DIRCOLSolver) where T
    n,m,N = size(prob)
    NN = N*(n+m)
    p_colloc = num_colloc(prob)
    p = num_constraints(prob)
    P = p_colloc + sum(p)
    dt = prob.dt

    part_f = create_partition2(prob.model)
    part_z = create_partition(n,m,N,N)
    pcum = cumsum(p)

    jac_structure = spzeros(P, NN)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)

    function eval_f(Z)
        X,U = unpack(Z,part_z)
        cost(prob.obj, X, U)
    end

    function eval_grad_f(Z, grad_f)
        X,U = unpack(Z, part_z)
        cost_gradient!(grad_f, prob, X, U)
    end

    function eval_g(Z, g)
        X,U = unpack(Z,part_z)
        g_colloc = view(g,1:p_colloc)
        g_custom = view(g,p_colloc+1:length(g))

        collocation_constraints!(g_colloc, prob, solver, X, U)
        update_constraints!(g_custom, prob, solver, X, U)
    end


    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure
            copyto!(rows,r)
            copyto!(cols,c)
        else
            X,U = unpack(Z, part_z)

            nG_colloc = p_colloc * 2(n+m)
            jac_colloc = view(vals, 1:nG_colloc)
            collocation_constraint_jacobian!(jac_colloc, prob, solver, X, U)

            # General constraint jacobians
            jac_custom = view(vals, nG_colloc+1:length(vals))
            constraint_jacobian!(jac_custom, prob, solver, X, U)
        end

        return nothing
    end
    return eval_f, eval_g, eval_grad_f, eval_jac_g
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


    function stash_prop(ln::String,prop::String,prop_name::Symbol=prop,vartype::Type=Float64)
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
            stash_prop(ln,"Number of Iterations..",:iterations,Int64) ? iter_lines = false : nothing
            stash_prop(ln,"Total CPU secs in IPOPT (w/o function evaluations)",:self_time)
            stash_prop(ln,"Total CPU secs in NLP function evaluations",:function_time)
            stash_prop(ln,"Number of objective function evaluations",:objective_calls,Int64)
            length(ln) > 0 && split(ln)[1] == "iter" && iter_lines == false ? iter_lines = true : nothing
            store_itervals(ln)
        end
    end
    props[:cost] = obj
    props[:c_max] = c_max
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
