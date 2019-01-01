"""
$(SIGNATURES)
    Calculate state midpoint using cubic spline
"""
function cubic_midpoint(x1::AbstractVector,dx1::AbstractVector,x2::AbstractVector,dx2::AbstractVector,dt::Float64)
    0.5*x1 + dt/8.0*dx1 + 0.5*x2 - dt/8.0*dx2
end

"""
$(SIGNATURES)
    Calculate state midpoints (xm)
"""
function calculate_midpoints!(results::SolverVectorResults, solver::Solver, X=results.X, U=results.U)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt
    for k = 1:N-1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing
        results.xm[k] = cubic_midpoint(X[k],results.dx[k],X[k+1],results.dx[k+1],dt)
        results.um[k] = 0.5*(U[k] + U[k+1])
    end
end

"""
$(SIGNATURES)
    Calculate state derivatives (dx)
"""
function calculate_derivatives!(results::SolverVectorResults, solver::Solver, X=results.X, U=results.U)
    n,m,N = get_sizes(solver)
    for k = 1:N
        solver.fc(results.dx[k],X[k],U[k][1:m])
    end
end
