mutable struct Forensics
    result::Array{SolverResults,1}
    cost::Array{Float64,1}
    time::Array{Float64,1}
    iter_type::Array{Int64,1}
    termination_index::Int64

    function Forensics(result, cost, time, iter_type, termination_index)
        new(result, cost, time, iter_type, termination_index)
    end
end

function Forensics(max_iter::Int)
    result = Array{SolverResults}(max_iter)
    cost = zeros(max_iter)
    time = zeros(max_iter)
    iter_type = zeros(max_iter)
    termination_index = 0
    Forensics(result, cost, time, iter_type, termination_index)
end

function merge_forensics(f1::Forensics,f2::Forensics)
    n1 = f1.termination_index
    n2 = f2.termination_index
    println("n1: $n1, n2: $n2")

    F = Forensics(n1+n2)
    F.result[1:n1] = f1.result[1:n1]
    F.result[n1+1:end] = f2.result[1:n2]
    F.cost[1:n1] = f1.cost[1:n1]
    F.cost[n1+1:end] = f2.cost[1:n2]
    F.time[1:n1] = f1.time[1:n1]
    F.time[n1+1:end] = f2.time[1:n2]
    F.iter_type[1:n1] = f1.iter_type[1:n1]
    F.iter_type[n1+1:end] = f2.iter_type[1:n2]
    F.termination_index = n1+n2
    F
end
