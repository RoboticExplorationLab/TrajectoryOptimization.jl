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
    Get true number of constraints, accounting for minimum time and infeasible start constraints
    -p: total number of stage constraints (state and control)
    -pI: number of inequality stage constraints (state and control)
    -pE: number of equality stage constraints (stage and control)
"""
function get_num_constraints(solver::Solver)
    if solver.state.constrained
        if solver.obj isa ConstrainedObjective
            p = solver.obj.p
            pI = solver.obj.pI
            pE = p - pI
        else
            p,pI,pE = 0,0,0
        end
        if is_min_time(solver)
            pI += 2
            pE += 1
        end
        solver.state.infeasible ? pE += solver.model.n : nothing
        p = pI + pE
        return p, pI, pE
    else
        return 0,0,0
    end
end

"""
$(SIGNATURES)
    Determine an initial dt for minimum time solver
"""

function get_initial_dt(solver::Solver)
    if is_min_time(solver)
        if solver.state.minimum_time_dt_estimate > 0.0
            dt = opts.minimum_time_dt_estimate
        elseif solver.state.minimum_time_tf_estimate > 0.0
            dt = solver.state.minimum_time_tf_estimate / (solver.N - 1)
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
Simulate system dynamics for a given control trajectory U
Updates X by propagating the dynamics, using the controls specified in
U
"""
function rollout!(res::SolverVectorResults, solver::Solver)
    status = rollout!(res.X, res.U, solver)

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

        # Get dt is minimum time
        solver.state.minimum_time ? dt = U[k][m̄]^2 : nothing

        # Propagate dynamics forward
        solver.fd(X[k+1], X[k], U[k][1:m], dt)

        # Add infeasible controls
        solver.state.infeasible ? X[k+1] += U[k][m̄+1:m̄+n] : nothing

        # Check that rollout has not diverged
        if ~(norm(X[k+1],Inf) < solver.opts.max_state_value && norm(U[k],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    return true
end

"""
$(SIGNATURES)
Simulate system dynamics using new control trajectory comprising
feedback gains K and feedforward gains d from backward pass
and previous control trajectory.
Line search option using alpha

flag indicates values are finite for all time steps.
"""
function rollout!(res::SolverVectorResults,solver::Solver,alpha::Float64)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[1] = solver.obj.x0;

    for k = 2:N
        # Calculate state trajectory difference
        δx = X_[k-1] - X[k-1]

        # Calculate updated control
        U_[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]

        # Get dt if minimum time
        solver.state.minimum_time ? dt = U_[k-1][m̄]^2 : nothing

        # Propagate dynamics
        solver.fd(X_[k], X_[k-1], U_[k-1][1:m], dt)

        # Add infeasible controls
        solver.state.infeasible ? X_[k] += U_[k-1][m̄.+(1:n)] : nothing

        # Check that rollout has not diverged
        if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    # Update constraints
    update_constraints!(res,solver,X_,U_)

    return true
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

function cost(solver::Solver,X::AbstractVector,U::AbstractVector)
    N = solver.N
    J = 0.0
    costfun = solver.obj.cost
    for k = 1:N-1
        J += stage_cost(costfun,X[k],U[k])*solver.dt
    end
    J += stage_cost(costfun, X[N])
end

"""
$(SIGNATURES)
    Compute the optimal control problem unconstrained cost,
    including minimum time and infeasible controls
"""

function _cost(solver::Solver{Obj},res::SolverVectorResults,X=res.X,U=res.U) where Obj <: Union{ConstrainedObjective, UnconstrainedObjective}
    # pull out solver/objective values
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    costfun = solver.obj.cost
    dt = solver.dt
    xf = solver.obj.xf

    J = 0.0
    for k = 1:N-1
        # Get dt if minimum time
        solver.state.minimum_time ? dt = U[k][m̄]^2 : nothing

        # Stage cost
        J += (stage_cost(costfun,X[k],U[k][1:m]))*dt

        # Minimum time cost
        solver.state.minimum_time ? J += solver.opts.R_minimum_time*dt : nothing

        # Infeasible control cost
        solver.state.infeasible ? J += 0.5*solver.opts.R_infeasible*U[k][m̄.+(1:n)]'*U[k][m̄.+(1:n)] : nothing
    end

    # Terminal Cost
    J += stage_cost(costfun, X[N])

    return J
end

"""
$(SIGNATURES)
    Compute the Augmented Lagrangian constraints cost
"""
function cost_constraints(solver::Solver, res::ConstrainedIterResults)
    N = solver.N
    J = 0.0
    for k = 1:N-1
        J += 0.5*res.C[k]'*res.Iμ[k]*res.C[k] + res.λ[k]'*res.C[k]
    end

    J += 0.5*res.CN'*res.IμN*res.CN + res.λN'*res.CN

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
function calculate_jacobians!(res::ConstrainedIterResults, solver::Solver)::Nothing
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    for k = 1:N-1
        # Update discrete dynamics Jacobians
        res.fdx[k], res.fdu[k] = solver.Fd(res.X[k], res.U[k])

        # Update constraint Jacobians
        solver.c_jacobian(res.Cx[k], res.Cu[k], res.X[k],res.U[k])

        # Minimum time special case
        if solver.state.minimum_time && k < N-1
            res.Cu[k][end,m̄] = 1
        end
    end

    # Update terminal constraint Jacobian
    solver.c_jacobian(res.Cx_N, res.X[N])

    return nothing
end

function calculate_jacobians!(res::UnconstrainedIterResults, solver::Solver)::Nothing
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    for k = 1:N-1
        # Update discrete dynamics Jacobians
        res.fdx[k], res.fdu[k] = solver.Fd(res.X[k], res.U[k])
    end

    return nothing
end

########################################
### METHODS FOR CONSTRAINED PROBLEMS ###
########################################

"""
$(SIGNATURES)
Evalutes all inequality and equality constraints (in place) for the current
state and control trajectories
    see: A Novel Augmented Lagrangian Approach for Inequalities and Convergent
    Any-Time Non-Central Updates (Toussaint)
"""
function update_constraints!(res::ConstrainedIterResults, solver::Solver, X=res.X, U=res.U)::Nothing
    N = solver.N
    p,pI,pE = get_num_constraints(solver)
    m̄,mm = get_num_controls(solver)
    c_fun = solver.c_fun

    for k = 1:N-1
        # Update constraints
        c_fun(res.C[k], X[k], U[k])

        # Minimum time special case
        if solver.state.minimum_time
            if k < N-1
                res.C[k][end] = U[k][m̄] - U[k+1][m̄]
            end
        end

        # Get active constraint set
        get_active_set!(res,solver,p,pI,k)

        # Update penality-indicator matrices based on active set
        res.Iμ[k] = Diagonal(res.active_set[k].*res.μ[k])
    end

    # Terminal constraint
    c_fun(res.CN,X[N])
    res.IμN .= Diagonal(res.μN)  # NOTE: Assuming all terminal constraints are equality constraints
    return nothing # TODO allow for more general terminal constraint
end

function update_constraints!(res::UnconstrainedIterResults, solver::Solver, X=res.X, U=res.U)::Nothing
    return nothing
end

"""
$(SIGNATURES)
    Determine active set for inequality constraints
"""
function get_active_set!(results::ConstrainedIterResults,solver::Solver,p::Int,pI::Int,k::Int)
    # Inequality constraints
    for j = 1:pI
        if results.C[k][j] > -solver.opts.active_constraint_tolerance || results.λ[k][j] > 0.0
            results.active_set[k][j] = 1
        else
            results.active_set[k][j] = 0
        end
    end
    # Equality constraints
    for j = pI+1:p
        results.active_set[k][j] = 1
    end
    return nothing
end

"""
$(SIGNATURES)
    Count the number of constraints of each type from an objective
"""
function count_constraints(obj::ConstrainedObjective, constraints::Symbol=:all)
    n,m = get_sizes(obj)
    p = obj.p # number of constraints
    pI = obj.pI # number of inequality and equality constraints
    pE = p-pI # number of equality constraints

    pI_c = obj.pI_custom
    pE_c = obj.pE_custom

    u_min_active = isfinite.(obj.u_min)
    u_max_active = isfinite.(obj.u_max)
    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    pI_u_max = count(u_max_active)
    pI_u_min = count(u_min_active)
    pI_u = pI_u_max + pI_u_min

    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min

    pI_c = pI - pI_x - pI_u
    pE_c = pE

    p_N = obj.p_N
    pI_N = obj.pI_N
    pE_N = p_N - pI_N
    pI_N_c = pI_N
    if obj.use_goal_constraint
        pE_N_c = pE_N - n
    else
        pE_N_c = pE_N
    end
    if constraints == :all
        return (pI, pI_c, pI_N, pI_N_c), (pE, pE_c, pE_N, pE_N_c)
    elseif constraints == :custom
        return (pI_c, pI_N_c), (pE_c, pE_N_c)
    elseif constraints == :total
        return (pI, pI_N), (pE, pE_N)
    end

end

"""
$(SIGNATURES)
    Generate the Jacobian of a general nonlinear constraint function
        -constraint function must be inplace
        -automatic differentition via ForwardDiff.jl
"""
function generate_general_constraint_jacobian(c::Function,p::Int,p_N::Int,n::Int64,m::Int64)::Function
    c_aug! = f_augmented!(c,n,m)
    J = zeros(p,n+m)
    S = zeros(n+m)
    cdot = zeros(p)
    F(J,cdot,S) = ForwardDiff.jacobian!(J,c_aug!,cdot,S)

    function c_jacobian(cx,cu,x,u)
        S[1:n] = x
        S[n+1:n+m] = u
        F(J,cdot,S)
        cx[1:p,1:n] = J[1:p,1:n]
        cu[1:p,1:m] = J[1:p,n+1:n+m]
    end

    if p_N > 0
        J_N = zeros(p_N,n)
        xdot = zeros(p_N)
        F_N(J_N,xdot,x) = ForwardDiff.jacobian!(J_N,c,xdot,x) # NOTE: terminal constraints can only be dependent on state x_N
        function c_jacobian(cx,x)
            F_N(J_N,xdot,x)
            cx .= J_N
        end
    end

    return c_jacobian
end

"""
$(SIGNATURES)
Generate the constraints function C(x,u) and a function to compute the jacobians
Cx, Cu = Jc(x,u) from a `ConstrainedObjective` type. Automatically stacks inequality
and equality constraints and takes jacobians of custom functions with `ForwardDiff`.
Stacks the constraints as follows:
[upper control inequalities
 (√dt upper bound)
 lower control inequalities
 (√dt lower bound)
 upper state inequalities
 lower state inequalities
 general inequalities
 general equalities
 (control equalities for infeasible start)
 (dt - dt+1)]
"""
function generate_constraint_functions(obj::ConstrainedObjective; max_dt::Float64=1.0, min_dt::Float64=1e-2)
    n,m = get_sizes(obj)

    # Key: I=> inequality,   E=> equality
    #     _c=> custom   (lack)=> box constraint
    #     _N=> terminal (lack)=> stage

    min_time = obj.tf == 0

    p = obj.p # number of constraints
    pI, pI_c, pI_N, pI_N_c = obj.pI, obj.pI_custom, obj.pI_N, obj.pI_N_custom
    pE, pE_c, pE_N, pE_N_c = p-obj.pI, obj.pE_custom, obj.p_N - obj.pI_N, obj.pE_N_custom
    m̄ = m
    min_time ? m̄ += 1 : nothing
    labels = String[]

    # Append on min time bounds
    u_max = obj.u_max
    u_min = obj.u_min
    if min_time
        u_max = [u_max; sqrt(max_dt)]
        u_min = [u_min; sqrt(min_dt)]
    end

    # Mask for active (state|control) constraints
    u_min_active = isfinite.(u_min)
    u_max_active = isfinite.(u_max)
    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    # Inequality on control
    pI_u_max = count(u_max_active)
    pI_u_min = count(u_min_active)
    pI_u = pI_u_max + pI_u_min
    function c_control_limits!(c,x,u)
        c[1:pI_u_max] = (u - u_max)[u_max_active]
        c[pI_u_max+1:pI_u_max+pI_u_min] = (u_min - u)[u_min_active]
    end

    lbl_u_min = ["control (lower bound)" for i = 1:pI_u_min]
    lbl_u_max = ["control (upper bound)" for i = 1:pI_u_max]
    if min_time
        lbl_u_min[end] = "* √dt (lower bound)"
        lbl_u_max[end] = "* √dt (upper bound)"
    end

    # Inequality on state
    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min
    function c_state_limits!(c,x,u)
        c[1:pI_x_max] = (x - obj.x_max )[x_max_active]
        c[pI_x_max+1:pI_x_max+pI_x_min] = (obj.x_min - x)[x_min_active]
    end
    lbl_x_max = ["state (upper bound)" for i = 1:pI_x_max]
    lbl_x_min = ["state (lower bound)" for i = 1:pI_x_min]


    # Update pI
    pI = pI_x + pI_u + pI_c

    # Form inequality constraint
    function cI!(c,x,u)
        c_control_limits!(view(c,1:pI_u),x,u)
        c_state_limits!(view(c,(1:pI_x).+pI_u),x,u)
        if pI_c > 0
            obj.cI(view(c,(1:pI_c).+pI_u.+pI_x),x,u)
        end
    end
    lbl_cI = ["custom inequality" for i = 1:pI_c]
    lbl_cE = ["custom equality" for i = 1:pE_c]

    # Construct labels
    c_labels = [lbl_u_max; lbl_u_min; lbl_x_max; lbl_x_min; lbl_cI; lbl_cE]


    # Augment functions together
    function c_function!(c,x,u,y=zero(x),v=zero(u))::Nothing
        infeasible = length(u) != m̄
        cI!(view(c,1:pI),x,u[1:m̄])
        if pE_c > 0
            obj.cE(view(c,(1:pE_c).+pI),x,u[1:m])
        end
        if infeasible
            c[pI.+pE_c.+(1:n)] = u[m̄.+(1:n)]
        end
        return nothing
    end

    # Terminal Constraint
    # TODO make this more general
    iI = 1:pI_N
    iE = pI_N .+ (1:pE_N)
    function c_function!(c,x)
        if obj.use_goal_constraint
            c[1:n] = x - obj.xf
        else
            c[iI] = obj.cI_N(c,x)
            c[iE] = obj.cE_N(c,x)
        end
    end

    ### Jacobians ###
    # Declare known Jacobians
    In = Matrix(I,n,n)
    cx_control_limits = zeros(pI_u,n)
    cx_state_limits = zeros(pI_x,n)
    cx_state_limits[1:pI_x_max, :] = In[x_max_active,:]
    cx_state_limits[pI_x_max+1:end,:] = -In[x_min_active,:]

    Im = Matrix(I,m̄,m̄)
    cu_control_limits = zeros(pI_u,m̄)
    cu_control_limits[1:pI_u_max,:] = Im[u_max_active,:]
    cu_control_limits[pI_u_max+1:end,:] = -Im[u_min_active,:]
    cu_state_limits = zeros(pI_x,m̄)

    if pI_c > 0
        cI_custom_jacobian! = generate_general_constraint_jacobian(obj.cI, pI_c, pI_N_c, n, m)
    end
    if pE_c > 0
        cE_custom_jacobian! = generate_general_constraint_jacobian(obj.cE, pE_c, 0, n, m)  # QUESTION: Why is pE_N_c = 0?
    end

    cx_infeasible = zeros(n,n)
    cu_infeasible = In

    function c_jacobian!(cx::AbstractMatrix, cu::AbstractMatrix, x::AbstractArray,u::AbstractArray,y=zero(x),v=zero(u))
        infeasible = length(u) != m̄
        let m = m̄
            cx[1:pI_u, 1:n] = cx_control_limits
            cx[(1:pI_x).+pI_u, 1:n] = cx_state_limits

            cu[1:pI_u, 1:m] = cu_control_limits
            cu[(1:pI_x).+pI_u, 1:m] = cu_state_limits
        end

        if pI_c > 0
            cI_custom_jacobian!(view(cx,pI_x+pI_u+1:pI_x+pI_u+pI_c,1:n), view(cu,pI_x+pI_u+1:pI_x+pI_u+pI_c,1:m), x, u[1:m])
        end
        if pE_c > 0
            cE_custom_jacobian!(view(cx,pI_x+pI_u+pI_c+1:pI_x+pI_u+pI_c+pE_c,1:n), view(cu,pI_x+pI_u+pI_c+1:pI_x+pI_u+pI_c+pE_c,1:m), x, u[1:m])
        end

        if infeasible
            cx[pI+pE_c+1:pI+pE_c+n,1:n] = cx_infeasible
            cu[pI+pE_c+1:pI+pE_c+n,m̄+1:m̄+n] = cu_infeasible
        end
    end

    cx_N = In  # Jacobian of final state
    function c_jacobian!(j::AbstractArray,x::AbstractArray)
        j .= cx_N
    end

    return c_function!, c_jacobian!, c_labels
end

generate_constraint_functions(obj::UnconstrainedObjective; max_dt::Float64=1.0,min_dt=1.0e-2) = (c,x,u)->nothing, (cx,cu,x,u)->nothing, String[]

"""
$(SIGNATURES)
Compute the maximum constraint violation. Inactive inequality constraints are
not counted (masked by the Iμ matrix).
"""
function max_violation(results::ConstrainedIterResults)
    if size(results.CN,1) != 0
        return max(maximum(norm.(map((x)->x.>0, results.Iμ) .* results.C, Inf)), norm(results.CN,Inf))
    else
        return maximum(norm.(map((x)->x.>0, results.Iμ) .* results.C, Inf))
    end
end

function max_violation(results::UnconstrainedIterResults)
    return 0.0
end

function constraint_violations(results::ConstrainedIterResults)
    if size(results.CN,1) != 0
        return map((x)->x.>0, results.Iμ) .* results.C
    else
        return maximum(norm.(map((x)->x.>0, results.Iμ) .* results.C, Inf))
    end
end

function evaluate_trajectory(solver::Solver, X, U)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
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
    Calculate infeasible controls to produce an infeasible state trajectory
"""
function infeasible_controls(solver::Solver,X0::Array{Float64,2},u::Array{Float64,2})
    ui = zeros(solver.model.n,solver.N) # initialize
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
    u = zeros(solver.model.m,solver.N)
    if solver.state.minimum_time
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

    if solver.state.minimum_time
        solver.state.infeasible ? sep = " and " : sep = " with "
        solve_string = sep * "minimum time..."

        # Initialize controls with sqrt(dt)
        if size(U0,1) == m
            U_init = [U0; ones(1,size(U0,2))*sqrt(get_initial_dt(solver))]
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
