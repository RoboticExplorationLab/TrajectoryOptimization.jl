using Combinatorics, IterTools

function create_partition(lengths::Vector{Int})
    num = 0
    partition = UnitRange{Int}[]
    for i = 1:length(lengths)
        push!(partition, (1:lengths[i]) .+ num)
        num += lengths[i]
    end
    return partition
end

function create_partition(lengths::Vector{Int}, names::NTuple{N,Symbol} where N)
    if length(lengths) == length(names)
        partition = create_partition(lengths)
        named_part = NamedTuple{names}(partition)
        return named_part
    else
        throw(ArgumentError("lengths must be the same"))
    end
end

function create_partition(sizes::Vector{NTuple{N,Int}} where N)
    num = 0
    partition = NTuple{N,UnitRange{Int}}[]
    for i = 1:length(sizes)
        push!(partition, (1:lengths[1]) .+ num)
        num += lengths[1]
    end
    return partition
end

function create_partition2(lengths::Vector{Int})
    part1 = create_partition(lengths)
    partition = NTuple{2,UnitRange{Int}}[]
    for (rng1,rng2) in Iterators.product(part1,part1)
        push!(partition, (rng1,rng2))
    end
    return CartesianIndices.(partition)
end

function create_partition2(lengths::Vector{Int}, names::NTuple{N,Symbol} where N)
    if length(lengths) == length(names)
        partition = create_partition2(lengths)
        names_all = vec(collect(Iterators.product(names,names)))
        names_all = Tuple([Symbol(string(a) * string(b)) for (a,b) in names_all])
        named_part = NamedTuple{names_all}(partition)
        return named_part
    else
        throw(ArgumentError("lengths must be the same"))
    end
end
