####################################
### METHODS FOR INFEASIBLE START ###
####################################

"""
$(SIGNATURES)
    Calculate infeasible controls to produce an infeasible state trajectory
"""
function infeasible_controls(solver::Solver,X0::Array{Float64,2},u::Array{Float64,2})
    ui = zeros(solver.model.n,solver.N-1) # initialize
    m = solver.model.m
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    x = zeros(solver.model.n,solver.N)
    x[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        solver.state.minimum_time ? dt = u[m̄,k]^2 : nothing

        solver.fd(view(x,:,k+1),x[:,k],u[1:m,k], dt)

        ui[:,k] = X0[:,k+1] - x[:,k+1]
        x[:,k+1] += ui[:,k]
    end
    ui
end

function infeasible_controls(solver::Solver,X0::Array{Float64,2})
    u = zeros(solver.model.m,solver.N-1)
    if solver.state.minimum_time
        dt = get_initial_dt(solver)
        u_dt = ones(1,solver.N-1)
        u = [u; u_dt]
    end
    infeasible_controls(solver,X0,u)
end

"""
$(SIGNATURES)
    Infeasible start solution is run through time varying LQR to track state and control trajectories
"""
function get_feasible_trajectory(results0::SolverIterResults,solver::Solver)::SolverIterResults
    # Need to copy to avoid writing over original results
    results = copy(results0)

    get_feasible_trajectory!(results,solver)
    return results
end

function get_feasible_trajectory!(results::SolverIterResults,solver::Solver)::Nothing
    remove_infeasible_controls!(results,solver)

    # backward pass - project infeasible trajectory into feasible space using time varying lqr
    Δv = backwardpass!(results, solver)

    # forward pass
    forwardpass!(results,solver,Δv,cost(solver, results, results.X, results.U))

    # update trajectories
    copyto!(results.X, results.X_)
    copyto!(results.U, results.U_)

    # return constrained results if input was constrained
    if !solver.state.unconstrained_original_problem
        update_constraints!(results,solver,results.X,results.U)
        update_jacobians!(results,solver)
    else
        solver.state.constrained = false
    end

    return nothing
end

"""
$(SIGNATURES)
Linear interpolation trajectory between initial and final state(s)
"""
function line_trajectory(solver::Solver, method=:trapezoid)::Array{Float64,2}
    N, = get_N(solver,method)
    line_trajectory(solver.obj.x0,solver.obj.xf,N)
end

function line_trajectory(x0::Array{Float64,1},xf::Array{Float64,1},N::Int64)::Array{Float64,2}
    x_traj = zeros(size(x0,1),N)
    t = range(0,stop=N,length=N)
    slope = (xf-x0)./N
    for i = 1:size(x0,1)
        x_traj[i,:] = slope[i].*t
    end
    x_traj
end


"""
$(SIGNATURES)
Generate a control trajectory that holds a mechanism in the configuration q
"""
function hold_trajectory(solver, mech::Mechanism, q)
    state = MechanismState(mech)
    nn = num_positions(state)
    set_configuration!(state, q[1:nn])
    vd = zero(state.q)
    u0 = dynamics_bias(state)

    n,m,N = get_sizes(solver)
    if length(q) > m
        throw(ArgumentError("system must be fully actuated to hold an arbitrary position ($(length(q)) should be > $m)"))
    end
    U0 = zeros(m,N)
    for k = 1:N
        U0[:,k] = u0
    end
    return U0
end



hold_trajectory(solver::Solver, q) = hold_trajectory(solver, solver.model.mech, q)
hold_trajectory(solver::Solver) = hold_trajectory(solver, solver.model.mech, solver.obj.x0)
