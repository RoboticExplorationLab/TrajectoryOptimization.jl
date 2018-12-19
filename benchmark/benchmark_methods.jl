
function retune_params(suite)
    tune!(suite)
    BenchmarkTools.save(paramsfile, params(suite))
end

function save_benchmark(results, stats, tags::Vector{<:Any}=[])
    # Add any custom tags (includes date by default)
    append!(suite.tags, tags)
    append!(stats.tags, tags)

    # Load in history
    suite_history, stats_history = load_benchmark_history()

    # Add current to history
    key = string(today())
    if haskey(suite_history, key)
        print("A benchmark has already been run today. Enter:\n  0 - to Cancel\n  1 - to Overwrite\n  \t Another key name\n INPUT > ")
        input = readline()
        if input == "0"
            return nothing
        elseif input != "1"
            key = input
        end
    end

    suite_history[key] = results
    stats_history[key] = stats

    # Save to file
    printstyled("Saving to file",color=:blue)
    BenchmarkTools.save(histfile, suite_history, stats_history)
    return nothing
end

function set_baseline_benchmark(tag::String)
    suite_history, stats_history = load_benchmark_history()
    results = suite_history[tag]
    stats = stats_history[tag]
    set_baseline_benchmark(results, stats)
end

function set_baseline_benchmark(results::BenchmarkGroup, stats::BenchmarkGroup)
    suite_history, stats_history = load_benchmark_history()
    suite_history["baseline"] = results
    stats_history["baseline"] = stats
    BenchmarkTools.save(histfile, suite_history, stats_history)
    return nothing
end

function get_baseline_benchmark()
    suite_history, stats_history = load_benchmark_history()
    suite_history["baseline"], stats_history["baseline"]
end

function load_benchmark_history()
    suite_history, stats_history = BenchmarkTools.load(histfile)
end

function baseline_comparison(results, stats; f=BenchmarkTools.median, styled=true)
    base_res, base_stats = get_baseline_benchmark()
    benchmark_comparison(results, stats, base_res, base_stats)
end

function baseline_comparison()
    results = run(suite)
    baseline_comparison(results, stats)
end

function benchmark_comparison(r1,s1,r2,s2; f=BenchmarkTools.median, styled=true)
    r_comp = compare_results(r1, r2,f=f)
    s_comp = compare_stats(s1, s2)
    if [r[1] for r in r_comp] == [s[1] for s in s_comp]
    end
    for (r,s) in zip(r_comp,s_comp)
        if r[1] == s[1]  # Check to make sure the runs match
            # Merge the Dictionaries
            merge!(r[2], s[2])
        end
    end
    print_comparison_table(r_comp, styled=styled)
end

function compare_results(r1::BenchmarkGroup, r2::BenchmarkGroup; f=BenchmarkTools.median)
    tmp = leaves(r1)
    run_list1 = [t[1] for t in tmp]
    tmp = leaves(r2)
    run_list2 = [t[1] for t in tmp]
    run_list = intersect(run_list1, run_list2)

    j = judge(f(r1),f(r2))

    comp = []
    for run in run_list
        vals = [f(r1[run]), f(r2[run])]
        t = j[run]
        stats_comp = Dict()
        stats_comp["time"] =  (join([prettytime(val.time) for val in vals], " ← ") * " (" * prettydiff(time(ratio(t))) * ")", t.time)
        stats_comp["memory"] = (join([prettymemory(val.memory) for val in vals], " ← ") * " (" * prettydiff(memory(ratio(t))) * ")", t.memory)
        stats_comp["gctime"] = (join([prettytime(val.gctime) for val in vals], " ← ") * " (" * prettytime(vals[1].gctime - vals[2].gctime) * ")",
            check_improvement(vals[1].gctime, vals[2].gctime, Inf))
        stats_comp["allocs"] = (join([format(val.allocs, autoscale=:metric, precision=2) for val in vals], " ← ") * " (" * format(vals[1].allocs - vals[2].allocs, signed=true, autoscale=:metric, precision=3) * ")",
            check_improvement(vals[1].allocs, vals[2].allocs))
        push!(comp, (run, stats_comp))
    end
    return sort(comp)
end

function check_improvement(v1,v2,tol=0.05)
    diff = v1-v2
    if diff > tol*v2
        return :regression
    elseif diff < -tol*v2
        return :improvement
    else
        return :invariant
    end
end


prettytimesigned(x::Real) = (x > 0 ? "+" : "") * prettytime(x)
prettyjudgetime(t::BenchmarkTools.TrialJudgement) = prettydiff(time(ratio(t))) * " => " * string(time(t)) * " (" * prettypercent(params(t).time_tolerance) * " tolerance)"
prettyjudgememory(t::BenchmarkTools.TrialJudgement) = prettydiff(memory(ratio(t))) * " => " * string(memory(t)) * " (" * prettypercent(params(t).memory_tolerance) * " tolerance)"

function compare_stats(s1::BenchmarkGroup, s2::BenchmarkGroup)
    stat_list = ["iterations","c_max","cost","setup_time","major iterations"]
    tmp = leaves(s1[@tagged "iterations"])
    run_list1 = [t[1] for t in tmp]
    tmp = leaves(s2[@tagged "iterations"])
    run_list2 = [t[1] for t in tmp]
    run_list = intersect(run_list1, run_list2)
    run_list = [run[1:end-1] for run in run_list]

    width = 8
    comp = []
    for run in run_list
        stat_comp = Dict()
        for stat in stat_list
            v1 = s1[run][stat]
            v2 = s2[run][stat]
            diff = v1 - v2
            str = trim_entry(v1, width, pad=false, positivespace=false) * " ← " * trim_entry(v2, width, pad=false, positivespace=false) * " (" * trim_entry(diff, width, pad=false, positivespace=false, signed=true) * ")"
            stat_comp[stat] = (str, check_improvement(v1,v2))
        end
        push!(comp,(run, stat_comp))
    end
    return sort(comp)
end

function loop_groups(s1::BenchmarkGroup, s2::BenchmarkGroup)
    k1 = keys(s1)
    k2 = keys(s2)
    both_keys = intersect(k1,k2)
    for key in both_keys
        if s1[key] isa BenchmarkGroup && s2[key] isa BenchmarkGroup
            loop_groups(s1[key], s2[key])
        else
            @show key, s1[key], s2[key]
        end
    end
end


function print_comparison_table(r_comp::Vector{Any}; styled=true)
    cols = ["time","memory","allocs","iterations","major iterations","c_max","cost"]

    ids = [join(run[1], ", ") for run in r_comp]
    change = [[run[2][key][2] for key in cols] for run in r_comp]
    change = hcat(change...)
    table = [[rpad(run[2][key][1],20) for key in cols] for run in r_comp]
    table = hcat(table...)
    table = vcat(reshape(ids,1,length(ids)),table)

    table = hcat(["IDs"; cols], table)
    col_widths = maximum(length.(table), dims=2)
    n_stats,n_runs = size(table)
    for j = 1:n_runs, i=1:n_stats
        table[i,j] = rpad(table[i,j],col_widths[i])
    end

    header = "| " * join(table[:,1]," | ") * " |";
    hline = "|" * join(["-"^(width+2) for width in col_widths],"|") * "|";

    if styled
        printstyled(header,bold=true); println();
        println(hline)
        colormap = Dict(:invariant=>:grey, :regression=>:red, :improvement=>:green)
        for i = 2:n_runs
            print("| ")
            printstyled(table[1,i],bold=true)
            for j = 2:n_stats
                print(" | ")
                printstyled(table[j,i],color=colormap[change[j-1,i-1]])
            end
            println(" |")
        end
    else
        println(header)
        println(hline)
        for i = 2:n_runs
            println("| " * join(table[:,i]," | ") * " |")
        end
    end
end
