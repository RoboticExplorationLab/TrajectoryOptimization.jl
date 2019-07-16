"""
$(SIGNATURES)
Extract important information from the SNOPT output file(s)
"""
function parse_snopt_summary(file=joinpath(pwd(),"snopt.out"))
    props = Dict()
    obj = Vector{Float64}()
    c_max = Vector{Float64}()
    next_line = false

    function stash_prop(ln::String,prop::String,prop_name::String=prop,vartype::Type=Float64)
        if occursin(prop,ln)
            loc = findfirst(prop,ln)
            if vartype <: Real
                val = split(ln[loc[end]+1:end])[1]
                val = convert(vartype,parse(Float64,val))
                props[prop_name] = val
                return true
            end
        end
        return false
    end

    function store_itervals(ln::String)
        vals = split(ln)

        if vals[6][1] == "("[1]

            idx = -1
            for i = 2:length(vals[6])
                if vals[6][i] == ")"[1] && idx == -1
                    idx = i-1
                    break
                end
            end
            _c = vals[6][2:idx]

        else
            _c = vals[6]
        end
        push!(obj, parse(Float64,vals[8])) # Merit function
        push!(c_max, parse(Float64,_c))

        next_line = false
    end

    open(file) do f
        for ln in eachline(f)
            stash_prop(ln,"No. of iterations","iterations",Int64)
            stash_prop(ln,"No. of major iterations","major iterations",Int64)
            stash_prop(ln,"No. of calls to funobj","objective calls",Int64)
            stash_prop(ln,"Nonlinear constraint violn","c_max")
            if next_line
                store_itervals(ln)
            end

            if length(ln) > 0 && split(ln)[1] == "Itns" && split(ln)[2] == "Major"
                next_line = true
            end
        end
    end
    props[:cost] = obj
    props[:c_max] = c_max
    return props
end
