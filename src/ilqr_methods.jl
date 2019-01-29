"""
$(SIGNATURES)
Get number of (solver) controls, accounting for minimum time and infeasible start
# Output
- m̄:  number of non infeasible controls (ie, system controls + time control if minimum time). System controls augmented by one if time is included as a control for minimum time problems.
- mm: total number of solver controls
"""
function get_num_controls(solver::Solver)
    n,m = get_sizes(solver)
    m̄ = m
    solver.state.minimum_time ? m̄ += 1 : nothing
    solver.state.infeasible ? mm = m̄ + n : mm = m̄
    return m̄, mm
end

"""
$(SIGNATURES)
Get number of (solver) stats, accounting for minimum time
# Output
- n̄:  number of non infeasible states (ie, system states + time state if minimum time). System state augmented by one if time is included as a state for minimum time problems.
- nn: total number of solver states
"""
function get_num_states(solver::Solver)
    n,m = get_sizes(solver)
    n̄ = n
    solver.state.minimum_time ? n̄ += 1 : nothing

    # TODO for now:
    nn = n̄
    return n̄, nn
end

"""
$(SIGNATURES)
    Compute the optimal control problem cost
"""
function cost(solver::Solver,vars::DircolVars)
    cost(solver,vars.X,vars.U)
end

function cost(solver::Solver, X::AbstractMatrix, U::AbstractMatrix)
    cost(solver, to_dvecs(X), to_dvecs(U))
end

function cost(solver::Solver,X,U)
    n,m,N = get_sizes(solver)
    J = 0.0
    costfun = solver.obj.cost
    for k = 1:N-1
        J += stage_cost(costfun,X[k][1:n],U[k][1:m])*solver.dt
    end
    J += stage_cost(costfun, X[N][1:n])
end

"""
$(SIGNATURES)
    Compute the optimal control problem unconstrained cost,
    including minimum time and infeasible controls
"""
function _cost(solver::Solver{M,Obj},res::SolverVectorResults,X=res.X,U=res.U) where {M, Obj<:Objective}
    # pull out solver/objective values
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    costfun = solver.obj.cost
    dt = solver.dt
    xf = solver.obj.xf

    J = 0.0
    for k = 1:N-1
        # Get dt if minimum time
        solver.state.minimum_time ? dt = U[k][m̄]^2 : nothing

        # Stage cost
        J += (stage_cost(costfun,X[k][1:n],U[k][1:m]))*dt

        # Minimum time cost
        solver.state.minimum_time ? J += solver.opts.R_minimum_time*dt : nothing

        # Infeasible control cost
        solver.state.infeasible ? J += 0.5*solver.opts.R_infeasible*U[k][m̄.+(1:n)]'*U[k][m̄.+(1:n)] : nothing
    end

    # Terminal Cost
    J += stage_cost(costfun, X[N][1:n])

    return J
end

"""
$(SIGNATURES)
    Compute the Augmented Lagrangian constraints cost
"""
function cost_constraints(solver::Solver, res::ConstrainedIterResults)
    N = solver.N
    J = 0.0
    for k = 1:N
        J += 0.5*res.C[k]'*res.Iμ[k]*res.C[k] + res.λ[k]'*res.C[k]
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
    Calculate dynamics and constraint Jacobians (perform prior to the backwards pass)
"""
function update_jacobians!(res::ConstrainedIterResults, solver::Solver, jacobians::Symbol=:both)::Nothing
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    dt = solver.dt

    for k = 1:N-1
        # Update discrete dynamics Jacobians
        jacobians in [:dynamics,:both] ? solver.Fd(res.fdx[k],res.fdu[k], res.X[k][1:n], res.U[k]) : nothing

        # Update constraint Jacobians
        if !solver.state.fixed_constraint_jacobians && jacobians in [:constraints,:both]

             solver.c_jacobian(res.Cx[k], res.Cu[k], res.X[k],res.U[k])

            # Minimum time special case
            if solver.state.minimum_time
                # No equality constraint at first time step
                if k == 1
                    res.Cu[k][p,m̄] = 0.
                    res.Cx[k][p,n̄] = 0.
                end
                # Jacobian for x[n̄]_k+1 = u[m̄]_k
                res.fdu[k][n̄,m̄] = 1.
            end
        end
    end

    # Update terminal constraint Jacobian
    (!solver.state.fixed_terminal_constraint_jacobian && jacobians in [:constraints,:both]) ? solver.c_jacobian(view(res.Cx[N],1:p_N,1:n), res.X[N][1:n]) : nothing

    return nothing
end

function update_jacobians!(res::UnconstrainedIterResults, solver::Solver, jacobians::Symbol=:none)::Nothing
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    for k = 1:N-1
        # Update discrete dynamics Jacobians
        solver.Fd(res.fdx[k],res.fdu[k], res.X[k][1:n], res.U[k])
    end

    return nothing
end

function evaluate_trajectory(solver::Solver, X, U)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    results = init_results(solver,X,U)
    update_jacobians!(results, solver)
    update_constraints!(results, solver)
    return results
end

"""
$(SIGNATURES)
Interpolate a trajectory using cubic interpolation
"""
function interp_traj(N::Int,tf::Float64,X::Matrix,U::Matrix)::Tuple{Matrix,Matrix}
    if isempty(X)
        X2 = X
    else
        X2 = interp_rows(N,tf,X)
    end
    U2 = interp_rows(N-1,tf,U)
    return X2, U2
end

"""
$(SIGNATURES)
Interpolate the rows of a matrix using cubic interpolation
"""
function interp_rows(N::Int,tf::Float64,X::Matrix)::Matrix
    n,N1 = size(X)
    t1 = range(0,stop=tf,length=N1)
    t2 = collect(range(0,stop=tf,length=N))
    X2 = zeros(n,N)
    for i = 1:n
        interp_cubic = CubicSplineInterpolation(t1, X[i,:])
        X2[i,:] = interp_cubic(t2)
    end
    return X2
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
    n̄,nn = get_num_states(solver)

    if size(U0,1) ∉ [m,mm]
        ArgumentError("Size of U0 must be either: system controls OR all expected solver controls (system + infeasible + minimum time)")
    end

    if N-1 != size(U0,2)
        @info "Interpolating initial guess"
        X0,U0 = interp_traj(N,solver.obj.tf,X0,U0)
    end

    if solver.state.minimum_time
        solver.state.infeasible ? sep = " and " : sep = " with "
        solve_string = sep * "Minimum Time..."

        # Initialize controls with sqrt(dt)
        if size(U0,1) == m
            U_init = [U0; ones(1,N-1)*sqrt(get_initial_dt(solver))]
        end
    else
        solve_string = "..."
        U_init = U0
    end

    if solver.state.infeasible
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
        if solver.state.constrained
            solve_string = "Solving Constrained Problem" * solve_string
        else
            solve_string = "Solving Unconstrained Problem" * solve_string
        end
        X_init = zeros(n,N)
    end

    @info solve_string

    if solver.state.minimum_time
        X_init = [X_init; zeros(1,N)]
    end

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
        results.dρ[1] = max(results.dρ[1]*solver.opts.bp_reg_increase_factor, solver.opts.bp_reg_increase_factor)
        results.ρ[1] = max(results.ρ[1]*results.dρ[1], solver.opts.bp_reg_min)
        if results.ρ[1] > solver.opts.bp_reg_max
            @warn "Max regularization exceeded"
        end
    elseif status == :decrease # decrease regularization
        results.dρ[1] = min(results.dρ[1]/solver.opts.bp_reg_increase_factor, 1.0/solver.opts.bp_reg_increase_factor)
        results.ρ[1] = results.ρ[1]*results.dρ[1]*(results.ρ[1]*results.dρ[1]>solver.opts.bp_reg_min)
    end
end

"""
$(SIGNATURES)
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function gradient_todorov(res::SolverVectorResults)
    N = length(res.X)
    maxes = zeros(N)
    for k = 1:N-1
        maxes[k] = maximum(abs.(res.d[k])./(abs.(res.U[k]).+1))
    end
    mean(maxes)
end

"""
$(SIGNATURES)
    Calculate the infinity norm of the gradient using feedforward term d (from δu = Kδx + d)
"""
function gradient_feedforward(res::SolverVectorResults)
    norm(res.d,Inf)
end

"""
$(SIGNATURES)
    Gradient of the Augmented Lagrangian: ∂L/∂u = Quuδu + Quxδx + Qu + Cu'λ + Cu'IμC
"""
function gradient_AuLa(results::SolverIterResults,solver::Solver)
    N = solver.N
    X = results.X; X_ = results.X_; U = results.U; U_ = results.U_

    Qx = results.bp.Qx; Qu = results.bp.Qu; Qxx = results.bp.Qxx; Quu = results.bp.Quu; Qux = results.bp.Qux

    gradient_norm = 0. # ℓ2-norm

    for k = 1:N-1
        δx = X_[k] - X[k]
        δu = U_[k] - U[k]
        tmp = Quu[k]*δu + Qux[k]*δx + Qu[k]

        gradient_norm += sum(tmp.^2)
    end

    return sqrt(gradient_norm)
end

function update_gradient(results,solver)
    if solver.opts.gradient_type == :todorov
        gradient = gradient_todorov(results)
    elseif solver.opts.gradient_type == :AuLa
        gradient = gradient_AuLa(results,solver)
    elseif solver.opts.gradient_type == :feedforward
        gradient = gradient_feedforward(results)
    end
    return gradient
end
