using Ipopt

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
