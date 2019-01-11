"""
$(SIGNATURES)
    Determine if solving a minimum time problem
"""
function is_min_time(solver::Solver)
    if solver.dt == 0 && solver.N > 0
        return true
    end
    return false
end

"""
$(SIGNATURES)
    Determine an initial dt for minimum time solver
"""
function get_initial_dt(solver::Solver)
    if is_min_time(solver)
        if solver.opts.minimum_time_dt_estimate > 0.0
            dt = solver.opts.minimum_time_dt_estimate
        elseif solver.opts.minimum_time_tf_estimate > 0.0
            dt = solver.opts.minimum_time_tf_estimate / (solver.N - 1)
            if dt > solver.opts.max_dt
                dt = solver.opts.max_dt
                @warn "Specified minimum_time_tf_estimate is greater than max_dt. Capping at max_dt"
            end
        else
            dt  = solver.opts.max_dt / 2
        end
    else
        dt = solver.dt
    end
    return dt
end

function total_time(solver::Solver, results::SolverVectorResults)
    if is_min_time(solver)
        m̄,mm = get_num_controls(solver)
        T = sum([u[m̄]^2 for u in results.U[1:solver.N-1]])
    else
        T = solver.dt*(solver.N-1)
    end
    return T::Float64
end

function total_time(solver::Solver, results::DircolVars)
    if is_min_time(solver)
        m̄, = get_num_controls(solver)
        T = sum(results.U[m̄,1:N-1])
    else
        T = solver.dt*(solver.N-1)
    end
end

function get_time(solver::Solver)
    range(0,stop=solver.obj.tf,length=solver.N)
end
