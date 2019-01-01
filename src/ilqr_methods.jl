# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Important methods used by the forward and backward iLQR passes
#
#     GENERAL METHODS
#         rollout!: Compute state trajectory X given controls U
#         cost: Compute the cost
#         calculate_jacobians!: Compute jacobians
#     CONSTRAINTS:
#         update_constraints!: Update constraint values and handle activation of
#             inequality constraints
#         generate_constraint_functions: Given a ConstrainedObjective, generate
#             the constraint function and its jacobians
#         max_violation: Compute the maximum constraint violation
#     INFEASIBLE START:
#         infeasible_controls: Compute the augmented (infeasible) controls
#             required to meet the specified trajectory
#         line_trajectory: Generate a linearly interpolated state trajectory
#             between start and end
#         feasible_traj: Finish an infeasible start solve by removing the
#             augmented controls
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



########################################
###         GENERAL METHODS          ###
########################################

"""
$(SIGNATURES)
    Determine if the solver is solving a minimum time problem
"""
function is_min_time(solver::Solver)
    if solver.dt == 0 && solver.N > 0
        return true
    end
    return false
end

"""
$(SIGNATURES)
    Get number of controls, accounting for minimum time and infeasible start
    Output
    - m̄ = number of non infeasible controls. Augmented by one if time is included as a control for minimum time problems.
    - mm = total number of controls
"""
function get_num_controls(solver::Solver)
    n,m = get_sizes(solver)
    m̄ = m
    solver.opts.minimum_time ? m̄ += 1 : nothing
    solver.opts.infeasible ? mm = m̄ + n : mm = m̄
    return m̄, mm
end

"""
$(SIGNATURES)
    Get true number of constraints, accounting for minimum time and infeasible start constraints
"""
function get_num_constraints(solver::Solver)
    if solver.opts.constrained
        pIs = solver.obj.pIs
        pIsN = solver.obj.pIsN
        pIc = solver.obj.pIc
        pEs = solver.obj.pEs
        pEsN = solver.obj.pEsN
        pEc = solver.obj.pEc

        if is_min_time(solver)
            pIc += 2
            pEc += 1
        end

        solver.opts.infeasible ? pEc += solver.model.n : nothing

        return pIs, pIsN, pIc, pEs, pEsN, pEc
    else
        return 0,0,0,0,0,0
    end
end

function get_initial_dt(solver::Solver)
    if is_min_time(solver)
        if solver.opts.minimum_time_dt_estimate > 0.0
            if solver.opts.minimum_time_tf_estimate > 0
                @warn "dt estimate taking precedence over tf estimate"
            end
            dt = opts.minimum_time_dt_estimate
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

"""
$(SIGNATURES)
Roll out the dynamics for a given control sequence (initial)
Updates `res.X` by propagating the dynamics, using the controls specified in
`res.U`.
"""
function rollout!(res::SolverVectorResults, solver::Solver)
    status = rollout!(res.X, res.U, solver)

    # Calculate state derivatives and midpoints
    if solver.control_integration == :foh
        calculate_derivatives!(res, solver, res.X, res.U)
        calculate_midpoints!(res, solver, res.X, res.U)
    end

    # Update constraints
    update_constraints!(res,solver,res.X,res.U)
    return status
end

function rollout(solver::Solver, U::Matrix)
    n,m,N = get_sizes(solver)
    X = [zeros(n) for k=1:N]
    rollout!(X, to_dvecs(U), solver)
    return to_array(X)
end

function rollout!(X::Matrix, U::Matrix, solver::Solver)
    X_vecs = to_dvecs(X)
    status = rollout!(X_vecs, to_dvecs(U), solver)
    X .= to_array(X_vecs)
    return status
end

function rollout!(X::Vector, U::Vector, solver::Solver)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    X[1] = solver.obj.x0
    for k = 1:N-1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        if solver.control_integration == :foh
            solver.fd(X[k+1], X[k], U[k][1:m], U[k+1][1:m], dt) # get new state
        else
            solver.fd(X[k+1], X[k], U[k][1:m], dt)
        end

        solver.opts.infeasible ? X[k+1] += U[k][m̄+1:m̄+n] : nothing

        # Check that rollout has not diverged
        if ~(norm(X[k+1],Inf) < solver.opts.max_state_value && norm(U[k],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    return true
end

"""
$(SIGNATURES)
Roll out the dynamics using the gains and optimal controls computed by the
backward pass
Updates `res.X` by propagating the dynamics at each timestep, by applying the
gains `res.K` and `res.d` to the difference between states
Will return a flag indicating if the values are finite for all time steps.
"""
function rollout!(res::SolverVectorResults,solver::Solver,alpha::Float64)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[1] = solver.obj.x0;

    if solver.control_integration == :foh
        b = res.b
        du = alpha*d[1]
        U_[1] = U[1] + du
        dv = zero(du)
    end

    for k = 2:N
        δx = X_[k-1] - X[k-1]

        if solver.control_integration == :foh
            dv = K[k]*δx + b[k]*du + alpha*d[k]
            U_[k] = U[k] + dv
            solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
            solver.fd(X_[k], X_[k-1], U_[k-1][1:m], U_[k][1:m], dt)
            du = dv
        else
            U_[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]
            solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
            solver.fd(X_[k], X_[k-1], U_[k-1][1:m], dt)
        end

        solver.opts.infeasible ? X_[k] += U_[k-1][m̄.+(1:n)] : nothing

        # Check that rollout has not diverged
        if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    # Calculate state derivatives and midpoints
    if solver.control_integration == :foh
        calculate_derivatives!(res, solver, X_, U_)
        calculate_midpoints!(res, solver, X_, U_)
    end

    # Update constraints
    update_constraints!(res,solver,X_,U_)

    return true
end

"""
$(SIGNATURES)
Quadratic stage cost (with goal state)
"""
function stage_cost(x,u,Q::AbstractArray{Float64,2},R::AbstractArray{Float64,2},xf::Vector{Float64},c::Float64=0)::Union{Float64,ForwardDiff.Dual}
    0.5*(x - xf)'*Q*(x - xf) + 0.5*u'*R*u + c
end

function stage_cost(obj::Objective, x::Vector, u::Vector)::Float64
    0.5*(x - obj.xf)'*obj.Q*(x - obj.xf) + 0.5*u'*obj.R*u + obj.c
end

function ℓ(x,u,Q,R,xf=zero(x))
    0.5*(x - xf)'*Q*(x - xf) + 0.5*u'*R*u
end


"""
$(SIGNATURES)
Compute the unconstrained cost
"""
function cost(solver::Solver,vars::DircolVars)
    cost(solver,vars.X,vars.U)
end

function _cost(solver::Solver,res::SolverVectorResults,X=res.X,U=res.U)
    # pull out solver/objective values
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    obj = solver.obj
    Q = obj.Q; R = obj.R; xf::Vector{Float64} = obj.xf; Qf::Matrix{Float64} = obj.Qf
    dt = solver.dt

    J = 0.0
    for k = 1:N-1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing
        if solver.control_integration == :foh
            xm = res.xm[k]
            um = res.um[k]
            J += dt*(1/6*ℓ(X[k],U[k][1:m],Q,R,xf) + 4/6*ℓ(xm,um[1:m],Q,R,xf) + 1/6*ℓ(X[k+1],U[k+1][1:m],Q,R,xf)) # Simpson quadrature (integral approximation) for foh stage cost
        else
            J += dt*ℓ(X[k],U[k][1:m],Q,R,xf)
        end
        solver.opts.minimum_time ? J += solver.opts.R_minimum_time*dt : nothing
        solver.opts.infeasible ? J += 0.5*solver.opts.R_infeasible*U[k][m̄.+(1:n)]'*U[k][m̄.+(1:n)] : nothing
    end

    J += 0.5*(X[N] - xf)'*Qf*(X[N] - xf)

    return J
end

""" $(SIGNATURES) Compute the Constraints Cost """
function cost_constraints(solver::Solver, res::ConstrainedIterResults)
    N = solver.N
    pIs, pIsN, pIc, pEs, pEsN, pEc = get_num_constraints(solver)

    J = 0.0

    for k = 1:N
        # state constraints from k=2 - k=N
        if k != 1
            (k < N && pIs > 0) || (k == N && pIsN > 0) ? J += 0.5*res.gs[k]'*res.Iμs[k]*res.gs[k] + res.λs[k]'*res.gs[k] : nothing
            (k < N && pEs > 0) || (k == N && pEsN > 0) ? J += 0.5*res.hs[k]'*res.Iνs[k]*res.hs[k] + res.κs[k]'*res.hs[k] : nothing
        end
        # control constraints from k=1 - k=N-1 (foh k=N)
        if k != N || solver.control_integration == :foh
            pIc > 0 ? J += 0.5*res.gc[k]'*res.Iμc[k]*res.gc[k] + res.λc[k]'*res.gc[k] : nothing
            pEc > 0 ? J += 0.5*res.hc[k]'*res.Iνc[k]*res.hc[k] + res.κc[k]'*res.hc[k] : nothing
        end
    end

    return J
end

function cost_constraints(solver::Solver, res::UnconstrainedIterResults)
    return 0.
end


function cost(solver::Solver, res::SolverIterResults, X=res.X, U=res.U)
    _cost(solver,res,X,U) + cost_constraints(solver,res)
end

"""
$(SIGNATURES)
Calculate Jacobians prior to the backwards pass
Updates both dyanmics and constraint jacobians, depending on the results type.
"""
function calculate_jacobians!(res::ConstrainedIterResults, solver::Solver)::Nothing
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    for k = 1:N-1
        if solver.control_integration == :foh
            res.fdx[k], res.fdu[k], res.fdv[k] = solver.Fd(res.X[k], res.U[k], res.U[k+1])
            res.fcx[k], res.fcu[k][:,1:m] = solver.Fc(res.X[k], res.U[k][1:m])
        else
            res.fdx[k], res.fdu[k] = solver.Fd(res.X[k], res.U[k])
        end

        # TODO these jacobians are not changing and only need to be updated if custom constraints were used
        k != 1 ? solver.gsx(res.gsx[k],res.X[k]) : nothing
        solver.gcu(res.gcu[k],res.U[k])
        k != 1 ? solver.hsx(res.hsx[k],res.X[k]) : nothing
        solver.hcu(res.hcu[k],res.U[k])

        if solver.opts.minimum_time && k < N-1
            solver.opts.infeasible ? idx = n+1 : idx = 1
            res.hcu[k][idx,m̄] = 1
        end
    end

    solver.gsx(res.gsx[N],res.X[N])
    solver.hsNx(res.hsx[N],res.X[N])

    if solver.control_integration == :foh
        res.fcx[N], res.fcu[N][:,1:m] = solver.Fc(res.X[N], res.U[N][1:m])

        solver.gcu(res.gcu[N],res.U[N])
        solver.hcu(res.hcu[N],res.U[N])

        if solver.opts.minimum_time
            res.gcu[N][m̄,:] .= 0.0
            res.gcu[N][m̄+m̄,:] .= 0.0

            solver.opts.infeasible ? idx = n+1 : idx = 1
            res.hcu[N-1][idx,:] .= 0.0
            res.hcu[N][idx,:] .= 0.0
        end
    end
    return nothing
end

function calculate_jacobians!(res::UnconstrainedIterResults, solver::Solver)::Nothing
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    for k = 1:N-1
        if solver.control_integration == :foh
            res.fdx[k], res.fdu[k], res.fdv[k] = solver.Fd(res.X[k], res.U[k], res.U[k+1])
            res.fcx[k], res.fcu[k][:,1:m] = solver.Fc(res.X[k], res.U[k][1:m])
        else
            res.fdx[k], res.fdu[k] = solver.Fd(res.X[k], res.U[k])
        end
    end
    if solver.control_integration == :foh
        res.fcx[N], res.fcu[N][:,1:m] = solver.Fc(res.X[N], res.U[N][1:m])
    end

    return nothing
end

########################################
### METHODS FOR CONSTRAINED PROBLEMS ###
########################################

"""
$(SIGNATURES)
Evalutes all inequality and equality constraints (in place) for the current state and control trajectories
    A Novel Augmented Lagrangian Approach for Inequalities and Convergent Any-Time Non-Central Updates (Toussaint)
"""
function update_constraints!(res::ConstrainedIterResults, solver::Solver, X=res.X, U=res.U)::Nothing
    n,m,N = get_sizes(solver)
    pIs, pIsN, pIc, pEs, pEsN, pEc = get_num_constraints(solver)
    m̄,mm = get_num_controls(solver)
    solver.opts.infeasible ? idx = n+1 : idx = 1

    # Update constraints
    for k = 1:N
        if k != 1
            k != N ? solver.gs(res.gs[k],res.X[k]) : solver.gsN(res.gs[N],res.X[N])
            k != N ? solver.hs(res.hs[k],res.X[k]) : solver.hsN(res.hs[N],res.X[N])
        end
        if k != N || solver.control_integration == :foh
            solver.gc(res.gc[k],res.U[k])
            solver.hc(res.hc[k],res.U[k])
        end

        if solver.opts.minimum_time
            if k < N-1
                res.hcu[k][idx] = U[k][m̄] - U[k+1][m̄]
            end
            if k == N
                res.gc[k][m̄] = 0.0
                res.gc[k][m̄+m̄] = 0.0
            end
        end

        # Get active constraint set
        get_active_set!(res,solver,pIs,pIc,k)

        if k != 1
            res.Iμs[k] = Diagonal(res.gs_active_set[k].*res.μs[k])
            res.Iνs[k] = Diagonal(res.νs[k])
        end
        if k != N || solver.control_integration == :foh
            res.Iμc[k] = Diagonal(res.gc_active_set[k].*res.μc[k])
            res.Iνc[k] = Diagonal(res.νc[k])
        end
    end

    return nothing
end

function update_constraints!(res::UnconstrainedIterResults, solver::Solver, X=res.X, U=res.U)::Nothing
    return nothing
end

function get_active_set!(results::ConstrainedIterResults,solver::Solver,pIs::Int,pIc::Int,k::Int)
    # Inequality constraints
    if k != 1
        for j = 1:pIs
            if results.gs[k][j] > -solver.opts.active_constraint_tolerance || results.λs[k][j] > 0.0
                results.gs_active_set[k][j] = 1
            else
                results.gs_active_set[k][j] = 0
            end
        end
    end
    if k != solver.N || solver.control_integration == :foh
        for j = 1:pIc
            if results.gc[k][j] > -solver.opts.active_constraint_tolerance || results.λc[k][j] > 0.0
                results.gc_active_set[k][j] = 1
            else
                results.gc_active_set[k][j] = 0
            end
        end
    end

    return nothing
end

"""
$(SIGNATURES)
    Compute the maximum constraint violation. Inactive inequality constraints are
    not counted (masked by the penalty matrix).
"""
function max_violation(results::ConstrainedIterResults)
    a = maximum(norm.(map((x)->x.>0., results.Iμs) .* results.gs, Inf))
    b = maximum(norm.(map((x)->x.>0., results.Iμc) .* results.gc, Inf))
    c = maximum(norm.(results.hs, Inf))
    d = maximum(norm.(results.hc, Inf))
    return max(a,b,c,d)
end

function max_violation(results::UnconstrainedIterResults)
    return 0.0
end

function evaluate_trajectory(solver::Solver, X, U)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    pIs, pIsN, pIc, pEs, pEsN, pEc = get_num_constraints(solver)
    results = init_results(solver,X,U)
    calculate_midpoints!(results, solver)
    calculate_derivatives!(results, solver)
    calculate_jacobians!(results, solver)
    update_constraints!(results, solver)
    return results
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
####################################
### METHODS FOR INFEASIBLE START ###
####################################

"""
$(SIGNATURES)
Additional controls for producing an infeasible state trajectory
"""
function infeasible_controls(solver::Solver,X0::Array{Float64,2},u::Array{Float64,2})
    ui = zeros(solver.model.n,solver.N) # initialize
    m = solver.model.m
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    x = zeros(solver.model.n,solver.N)
    x[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        solver.opts.minimum_time ? dt = u[m̄,k]^2 : nothing
        if solver.control_integration == :foh
            solver.fd(view(x,:,k+1),x[:,k],u[1:m,k],u[1:m,k+1], dt)
        else
            solver.fd(view(x,:,k+1),x[:,k],u[1:m,k], dt)
        end
        ui[:,k] = X0[:,k+1] - x[:,k+1]
        x[:,k+1] += ui[:,k]
    end
    ui
end

function infeasible_controls(solver::Solver,X0::Array{Float64,2})
    u = zeros(solver.model.m,solver.N)
    if solver.opts.minimum_time
        dt = get_initial_dt(solver)
        u_dt = ones(1,solver.N)
        u = [u; u_dt]
    end
    infeasible_controls(solver,X0,u)
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
Generates the correctly sized input trajectory, tacking on infeasible and minimum
time controls, if required. Will interpolate the initial trajectory as needed.
# Arguments
* X0: Matrix of initial states. May be empty. If empty and the infeasible flag is set in the solver, it will initialize a linear interpolation from start to goal state.
* U0: Matrix of initial controls. May either be only the dynamics controls, or include infeasible and minimum time controls (as necessary).
"""
function get_initial_trajectory(solver::Solver, X0::Matrix{Float64}, U0::Matrix{Float64})
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    if size(U0,1) ∉ [m,mm]
        ArgumentError("Size of U0 must be either include only plant controls or all expected controls (infeasible + minimum time)")
    end

    if N != size(U0,2)
        @info "Interpolating initial guess"
        X0,U0 = interp_traj(N,solver.obj.tf,X0,U0)
    end

    if solver.opts.minimum_time
        solver.opts.infeasible ? sep = " and " : sep = " with "
        solve_string = sep * "minimum time..."

        # Initialize controls with sqrt(dt)
        if size(U0,1) == m
            U_init = [U0; ones(1,size(U0,2))*sqrt(get_initial_dt(solver))]
        end
    else
        solve_string = "..."
        U_init = U0
    end

    if solver.opts.infeasible
        solve_string =  "Solving Constrained Problem with Infeasible Start" * solve_string

        # Generate infeasible controls
        if size(U0,1) == m
            ui = infeasible_controls(solver,X0,U_init)  # generates n additional control input sequences that produce the desired infeasible state trajectory
            U_init = [U_init; ui]  # augment control with additional control inputs that produce infeasible state trajectory
        end

        # Assign state trajectory
        if isempty(X0)
            X_init = line_trajectory(solver)
        else
            X_init = X0
        end
    else
        solve_string = "Solving Constrained Problem" * solve_string
        X_init = zeros(n,N)
    end
    @info solve_string

    return X_init, U_init
end

"""
$(SIGNATURES)
    Regularization update scheme
        - see "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
"""
function regularization_update!(results::SolverResults,solver::Solver,status::Symbol=:increase)
    if status == :increase # increase regularization
        # @logmsg InnerLoop "Regularization Increased"
        results.dρ[1] = max(results.dρ[1]*solver.opts.ρ_factor, solver.opts.ρ_factor)
        results.ρ[1] = max(results.ρ[1]*results.dρ[1], solver.opts.ρ_min)
        if results.ρ[1] > solver.opts.ρ_max
            @warn "Max regularization exceeded"
        end
    elseif status == :decrease # decrease regularization
        results.dρ[1] = min(results.dρ[1]/solver.opts.ρ_factor, 1.0/solver.opts.ρ_factor)
        results.ρ[1] = results.ρ[1]*results.dρ[1]*(results.ρ[1]*results.dρ[1]>solver.opts.ρ_min)
    end
end

function get_time(solver::Solver)
    range(0,stop=solver.obj.tf,length=solver.N)
end
