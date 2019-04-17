cost(prob::Problem, solver::DIRCOLSolver) = cost(prob.cost, solver.Z.X, solver.Z.U, prob.dt)

function collocation_constraints!(g, prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}) where T
    n,m,N = size(prob)
    @assert isodd(N)
    dt = prob.dt

    iLow = 1:2:N-1
    iMid = iLow .+ 1
    iUpp = iMid .+ 1

    for k = 1:N-1
        g[:,k]
    collocation = - X[:,iUpp] + X[:,iLow] + dt[:,iLow].*(fVal[:,iLow] + 4*fVal[:,iMid] + fVal[:,iUpp])/6

end

function traj_points!(prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}) where T
    n,m,N = size(prob)
    Xm = solver.X_
    X,U = solver.Z.X, solver.Z.U
    for k = 1:N
        Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
    end
end

function dynamics!(prob::Problem, solver::DirectSolver)
    for k = 1:prob.N-1
        evaluate!(solver.fval[k], prob.model, prob.X[k], prob.U[k], prob.dt)
    end
end

function update_constraints!(g, prob::Problem, Z)

end

function cost_gradient!(grad_f, prob::Problem, Z)

end
