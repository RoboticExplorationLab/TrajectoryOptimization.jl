using Ipopt

function solve_ipopt(prob::Problem{T,Continuous}, opts::DIRCOLSolverOptions{T}) where T<:AbstractFloat
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
    eval_f, eval_g, eval_grad_f, eval_jac_g = TrajectoryOptimization.gen_dircol_functions(prob, solver)

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


"""
$(SIGNATURES)
Extract important information from the Ipopt output file(s)
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

function write_ipopt_options()
    if !isfile(joinpath(root_dir(),"logs","ipopt.out"))
        mkdir(joinpath(root_dir(),"logs"))
    end
    outfile=joinpath(root_dir(),"logs","ipopt.out")
    optfile=joinpath(root_dir(),"ipopt.opt")

    f = open(optfile,"w")
    println(f,"# IPOPT Options for TrajectoryOptimization.jl\n")
    println(f,"# Use Quasi-Newton methods to avoid the need for the Hessian")
    println(f,"hessian_approximation limited-memory\n")
    println(f,"# Output file")
    println(f,"file_print_level 5")
    println(f,"output_file"*" "*"\""*"$(outfile)"*"\"")
    close(f)
end
