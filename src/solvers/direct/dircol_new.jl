cost(prob::Problem, solver::DIRCOLSolver) = cost(prob.cost, solver.Z.X, solver.Z.U, prob.dt)

function constraints!(g, prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}) where T
    g_colloc = view(g,1:n*(N-1))
    collocation_constraints!(g_colloc, prob, solver)
end

function collocation_constraints!(g, prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}) where T
    n,m,N = size(prob)
    @assert isodd(N)
    dt = prob.dt

    # Reshape the contraint vector
    g_ = reshape(g,n,N-1)

    # Pull out values
    fVal = solver.fVal
    X = solver.Z.X
    U = solver.Z.U
    Xm = solver.X_
    fValm = zero(fVal[1])

    for k = 1:N-1
        Um = (U[k] + U[k+1])*0.5
        evaluate!(fValm, prob.model, Xm[k], Um) # dynamics at the midpoint
        g_[:,k] = -X[k+1] + X[k] + dt*(fVal[k] + 4*fValm + fVal[k+1])/6
    end
end

function traj_points!(prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}) where T
    n,m,N = size(prob)
    dt = prob.dt
    Xm = solver.X_
    fVal = solver.fVal
    X,U = solver.Z.X, solver.Z.U
    for k = 1:N-1
        Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
    end
end

function dynamics!(prob::Problem, solver::DirectSolver)
    for k = 1:prob.N
        evaluate!(solver.fVal[k], prob.model, solver.Z.X[k], solver.Z.U[k], prob.dt)
    end
end

function update_constraints!(g, prob::Problem, Z)
    n,m,N = size(prob)
    p,pN = num_stage_constraints(prob), num_terminal_constraints(prob)
    P = (N-1)*p
    X,U = Z.X, Z.U

    g_stage = reshape(view(g,1:P), p, N-1)
    g_term = view(g,P+1:length(g))

    for k = 1:N-1
        evaluate!(g_stage[:,k], prob.constraints, X[k], U[k])
    end
    evaluate!(g_term, prob.constraints, X[N])
end

function cost_gradient!(grad_f, prob::Problem, Z)

end

get_N(prob::Problem, solver::DIRCOLSolver) = get_N(prob.N, solver.opts.method)
