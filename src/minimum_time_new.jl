function total_time(prob::Problem{T}) where T
    m̄ = prob.model.m + 1
    tt = 0.0
    try
        tt = sum([prob.U[k][m̄] for k = 1:prob.N-1])
    catch
        tt = prob.dt*(prob.N-1)
    end
    return tt
end
