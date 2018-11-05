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

"$(SIGNATURES) determine if the solver is solving a minimum time problem"
function is_minimum_time(solver::Solver)
    if solver.dt == 0 && solver.N > 0
        return true
    end
    return false
end

"""
$(SIGNATURES)
Get number of controls, accounting for minimum time and infeasible start
# Output
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
    p = solver.obj.p
    pI = solver.obj.pI
    pE = p - pI
    if solver.opts.minimum_time
        pI += 2
        pE += 1
    end
    if solver.opts.infeasible
        pE += solver.model.n
    end
    p = pI + pE
    return p, pI, pE
end

function get_initial_dt(solver::Solver)
    if solver.opts.minimum_time
        if solver.opts.minimum_time_tf_estimate > 0.0
            dt = solver.opts.minimum_time_tf_estimate / (solver.N - 1)
            if dt > solver.opts.max_dt
                dt = solver.opts.max_dt
                @warn "Specified min_time_init is greater than max_dt. Capping at max_dt"
            end
        else
            dt  = 0.5*solver.opts.max_dt + 0.5*solver.opts.min_dt
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
    X, U = res.X, res.U
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

        if solver.opts.infeasible
            X[k+1] += U[k][m̄+1:m̄+n]
        end

        # Check that rollout has not diverged
        if ~(norm(X[k+1],Inf) < solver.opts.max_state_value && norm(U[k],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    # Calculate state derivatives and midpoints
    if solver.control_integration == :foh
        calculate_derivatives!(res,solver,X,U)
        calculate_midpoints!(res,solver,X,U)
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
        dv = zeros(mm)
        du = alpha*d[1]
        U_[1] = U[1] + du
    end

    for k = 2:N
        δx = X_[k-1] - X[k-1]

        solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing

        if solver.control_integration == :foh
            dv = K[k]*δx + b[k]*du + alpha*d[k]
            U_[k] = U[k] + dv
            solver.fd(X_[k], X_[k-1], U_[k-1][1:m], U_[k][1:m], dt)
            du = dv
        else
            U_[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]
            solver.fd(X_[k], X_[k-1], U_[k-1][1:m], dt)
        end

        if solver.opts.infeasible
            X_[k] += U_[k-1][m̄.+(1:n)]
        end

        # Check that rollout has not diverged
        if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    # Calculate state derivatives and midpoints
    if solver.control_integration == :foh
        calculate_derivatives!(res, solver, X_, U_)
        calculate_midpoints!(res, solver, X_,U_)
    end

    return true
end

"""
$(SIGNATURES)
    Quadratic stage cost (with goal state)
"""
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
    Q = obj.Q; R = obj.R; xf::Vector{Float64} = obj.xf; Qf::Matrix{Float64} = obj.Qf;
    dt = solver.dt

    J = 0.0

    # Stage costs
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

    # Terminal cost
    J += ℓ(X[N],zeros(m),Qf,zeros(m,m),xf)

    return J
end

""" $(SIGNATURES) Compute the Constraints Cost """
function cost_constraints(solver::Solver, res::ConstrainedIterResults)
    N = solver.N
    J = 0.0
    for k = 1:N-1
        J += 0.5*res.C[k]'*res.Iμ[k]*res.C[k] + res.λ[k]'*res.C[k]
    end

    if solver.control_integration == :foh
        J += 0.5*res.C[N]'*res.Iμ[N]*res.C[N] + res.λ[N]'*res.C[N]
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
    Calculate state midpoint using cubic spline
"""
function cubic_midpoint(x1::AbstractVector,dx1::AbstractVector,x2::AbstractVector,dx2::AbstractVector,dt::Float64)
    0.5*x1 + dt/8.0*dx1 + 0.5*x2 - dt/8.0*dx2
end

"""
$(SIGNATURES)
    Calculate state and control midpoints
"""
function calculate_midpoints!(results::SolverVectorResults, solver::Solver, X, U)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    for k = 1:solver.N-1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing
        results.xm[k] = cubic_midpoint(X[k],results.dx[k],X[k+1],results.dx[k+1],dt)
        results.um[k] = 0.5*results.U[k] + 0.5*results.U[k+1]
    end
end

"""
$(SIGNATURES)
    Calculate state derivatives (xdot)
"""
function calculate_derivatives!(results::SolverVectorResults, solver::Solver, X, U)
    n,m,N = get_sizes(solver)
    for k = 1:N
        solver.fc(results.dx[k],X[k],U[k][1:m])
    end
end

"""
$(SIGNATURES)
    Calculate Jacobians prior to the backwards pass
    -updates dynamics and constraint jacobians, depending on the results type
"""
function calculate_jacobians!(res::ConstrainedIterResults, solver::Solver)::Nothing
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    for k = 1:N-1
        if solver.control_integration == :foh
            res.fdx[k], res.fdu[k], res.fdv[k] = solver.Fd(res.X[k], res.U[k], res.U[k+1])
            res.fcx[k], res.fcu[k][:,1:m] = solver.Fc(res.X[k], res.U[k])
        else
            res.fdx[k], res.fdu[k] = solver.Fd(res.X[k], res.U[k])
        end
        solver.c_jacobian(res.Cx[k], res.Cu[k], res.X[k],res.U[k]) # NOTE: in the foh case, c(x,u,y,u) but the jacobian is not dependent on values from y or v
    end

    if solver.control_integration == :foh
        res.fcx[N], res.fcu[N][:,1:m] = solver.Fc(res.X[N], res.U[N])
        solver.c_jacobian(res.Cx[N], res.Cu[N], res.X[N],res.U[N])

        # Zero out minimum time constraints that are not present at N-1, N timesteps
        if solver.opts.minimum_time
            res.Cu[N][m̄,:] .= 0.0
            res.Cu[N][m̄+m̄,:] .= 0.0
            res.Cu[N-1][end,:] .= 0.0
            res.Cu[N][end,:] .= 0.0
        end
    end

    solver.c_jacobian(res.Cx_N, res.X[N])
    return nothing
end

function calculate_jacobians!(res::UnconstrainedIterResults, solver::Solver)::Nothing
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    for k = 1:N-1
        if solver.control_integration == :foh
            res.fdx[k], res.fdu[k], res.fdv[k] = solver.Fd(res.X[k], res.U[k], res.U[k+1])
            res.fcx[k], res.fcu[k][:,1:m] = solver.Fc(res.X[k], res.U[k])
        else
            res.fdx[k], res.fdu[k] = solver.Fd(res.X[k], res.U[k])
        end
    end
    if solver.control_integration == :foh
        res.fcx[N], res.fcu[N][:,1:m] = solver.Fc(res.X[N], res.U[N])
    end

    return nothing
end

########################################
### METHODS FOR CONSTRAINED PROBLEMS ###
########################################

"""
$(SIGNATURES)
Evalutes all inequality and equality constraints (in place) for the current state and control trajectories
"""
function update_constraints!(res::ConstrainedIterResults, solver::Solver, X::Array=res.X, U::Array=res.U)::Nothing
    N = solver.N
    p,pI,pE = get_num_constraints(solver)
    m̄,mm = get_num_controls(solver)
    c = solver.c
    if solver.control_integration == :foh
        final_index = N
    else
        final_index = N-1
    end

    for k = 1:final_index
        # Evaluate constraint
        if solver.control_integration == :foh && k <= N-1
            c(res.C[k], X[k], U[k], X[k+1], U[k+1])
        else
            c(res.C[k], X[k], U[k])
        end
        # Zero out minimum time constraints that are not present at N-1, N timesteps
        if solver.opts.minimum_time
            if k > N-2
                res.C[k][end] = 0.0
            end
            if k == N
                res.C[k][m̄] = 0.0
                res.C[k][m̄+m̄] = 0.0
            end
        end

        # Inequality constraints [see equation ref]
        for j = 1:pI
            if res.C[k][j] >= 0.0 || res.λ[k][j] > 0.0
                res.Iμ[k][j,j] = res.μ[k][j] # active (or previously active) inequality constraints are penalized
            else
                res.Iμ[k][j,j] = 0. # inactive inequality constraints are not penalized
            end
        end
        # Equality constraints
        for j = pI+1:p
            res.Iμ[k][j,j] = res.μ[k][j] # equality constraints are penalized
        end
    end
    # Terminal constraint
    c(res.CN,X[N])
    res.IμN .= Diagonal(res.μN)  # NOTE: Assuming all terminal constraints are equality constraints
    return nothing # TODO allow for more general terminal constraint
end

function update_constraints!(res::UnconstrainedIterResults, solver::Solver, X::Array=res.X, U::Array=res.U)::Nothing
    return nothing
end

"""
$(SIGNATURES)
    Count the number of constraints of each type from an objective
"""
function count_constraints(obj::ConstrainedObjective)
    n = size(obj.Q,1)
    p = obj.p # number of constraints
    pI = obj.pI # number of inequality and equality constraints
    pE = p-pI # number of equality constraints

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
    if obj.use_terminal_constraint
        pE_N_c = pE_N - n
    else
        pE_N_c = pE_N
    end

    return (pI, pI_c, pI_N, pI_N_c), (pE, pE_c, pE_N, pE_N_c)
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
        cx .= J[1:p,1:n]
        cu .= J[1:p,n+1:n+m]
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
 (dt upper bound)
 lower control inequalities
 (dt > 0)
 upper state inequalities
 lower state inequalities
 general inequalities
 general equalities
 (control equalities for infeasible start)
 (dt - dt+1)]
"""
function generate_constraint_functions(obj::ConstrainedObjective; max_dt::Float64=1.0, min_dt::Float64=1e-3)
    m = size(obj.R,1) # number of control inputs
    n = length(obj.x0) # number of states

    # Key: I=> inequality,   E=> equality
    #     _c=> custom   (lack)=> box constraint
    #     _N=> terminal (lack)=> stage

    min_time = obj.tf == 0

    pI_obj, pE_obj = count_constraints(obj)
    p = obj.p # number of constraints
    pI, pI_c, pI_N, pI_N_c = pI_obj
    pE, pE_c, pE_N, pE_N_c = pE_obj

    m̄ = m
    min_time ? m̄ += 1 : nothing

    # Append on min time bounds
    u_max = obj.u_max
    u_min = obj.u_min
    # if min_time
    #     u_max = [u_max; sqrt(max_dt)]
    #     u_min = [u_min; sqrt(min_dt)]
    # end

    # Mask for active (state|control) constraints
    u_min_active = isfinite.(u_min)
    u_max_active = isfinite.(u_max)
    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    # Inequality on control
    pI_u_max = count(u_max_active)
    pI_u_min = count(u_min_active)
    pI_u = pI_u_max + pI_u_min
    if min_time
        pI_u += 2
    end
    function c_control_limits!(c,x,u)
        c[1:pI_u_max] = (u - u_max)[u_max_active]
        c[min_time+pI_u_max+1:min_time+pI_u_max+pI_u_min] = (u_min - u)[u_min_active]
    end

    # Inequality on state
    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min
    function c_state_limits!(c,x,u)
        c[1:pI_x_max] = (x - obj.x_max )[x_max_active]
        c[pI_x_max+1:pI_x_max+pI_x_min] = (obj.x_min - x)[x_min_active]
    end

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

    # Augment functions together
    function c_function!(c,x,u,y=zero(x),v=zero(u))::Nothing
        infeasible = length(u) != m̄
        cI!(view(c,1:pI),x,u[1:m])
        if pE_c > 0
            obj.cE(view(c,(1:pE_c).+pI),x,u[1:m])
        end
        if min_time
            c[m̄] = u[m̄] - sqrt(max_dt)
            c[m̄+m̄] = sqrt(min_dt) - u[m̄]
            c[end] = u[m̄] - v[m̄]
        end
        if infeasible
            c[pI.+pE_c.+(1:n)] = u[m̄.+(1:n)]
        end
        return nothing
    end

    # Terminal Constraint
    # TODO make this more general
    function c_function!(c,x)
        c[1:n] = x - obj.xf
    end

    ### Jacobians ###
    # Declare known Jacobians
    In = Matrix(I,n,n)
    cx_control_limits = zeros(pI_u,n)
    cx_state_limits = zeros(pI_x,n)
    cx_state_limits[1:pI_x_max, :] = In[x_max_active,:]
    cx_state_limits[pI_x_max+1:end,:] = -In[x_min_active,:]

    Im = Matrix(I,m,m)
    cu_control_limits = zeros(pI_u,m)
    cu_control_limits[1:pI_u_max,:] = Im[u_max_active,:]
    cu_control_limits[min_time+pI_u_max+1:min_time+pI_u_max+pI_u_min,:] = -Im[u_min_active,:]
    cu_state_limits = zeros(pI_x,m)

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

        cx[1:pI_u, 1:n] = cx_control_limits
        cx[(1:pI_x).+pI_u, 1:n] = cx_state_limits

        cu[1:pI_u, 1:m] = cu_control_limits
        cu[(1:pI_x).+pI_u, 1:m] = cu_state_limits

        if pI_c > 0
            cI_custom_jacobian!(view(cx,pI_x+pI_u+1:pI_x+pI_u+pI_c,1:n), view(cu,pI_x+pI_u+1:pI_x+pI_u+pI_c,1:m), x, u[1:m])
        end
        if pE_c > 0
            cE_custom_jacobian!(view(cx,pI_x+pI_u+pI_c+1:pI_x+pI_u+pI_c+pE_c,1:n), view(cu,pI_x+pI_u+pI_c+1:pI_x+pI_u+pI_c+pE_c,1:m), x, u[1:m])
        end
        if min_time
            cu[m̄,m̄] = 1
            cu[m̄+m̄,m̄] = -1
            cu[end,m̄] = 1
        end
        if infeasible
            cx[pI_x+pI_u+pI_c+pE_c+1:pI_x+pI_u+pI_c+pE_c+n,1:n] = cx_infeasible
            cu[pI_x+pI_u+pI_c+pE_c+1:pI_x+pI_u+pI_c+pE_c+n,m̄+1:m̄+n] = cu_infeasible
        end
    end

    cx_N = In  # Jacobian of final state
    function c_jacobian!(j::AbstractArray,x::AbstractArray)
        j .= cx_N
    end

    return c_function!, c_jacobian!
end

generate_constraint_functions(obj::UnconstrainedObjective; max_dt::Float64=1.0, min_dt::Float64=1e-5) = (x,u)->nothing, (x,u)->nothing

"""
$(SIGNATURES)
Compute the maximum constraint violation. Inactive inequality constraints are
not counted (masked by the Iμ matrix). For speed, the diagonal indices can be
precomputed and passed in.
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


function total_time(solver::Solver, results::SolverVectorResults)
    if is_min_time(solver)
        m̄,mm = get_num_controls(solver)
        T = sum([u[m̄]^2 for u in results.U[1:solver.N-1]])
    else
        T = solver.dt*(solver.N-1)
    end
    return T::Float64
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
    x = zeros(solver.model.n,solver.N)
    x[:,1] = solver.obj.x0
    dt = solver.dt
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
    Regularization update scheme
        - see "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
"""
function regularization_update!(results::SolverResults,solver::Solver,status::Symbol=:increase)
    if status == :increase # increase regularization
        # @logmsg InnerLoop "Regularization Increased"
        results.dρ[1] = max(results.dρ[1]*solver.opts.ρ_factor, solver.opts.ρ_factor)
        results.ρ[1] = max(results.ρ[1]*results.dρ[1], solver.opts.ρ_min)
        if results.ρ[1] > solver.opts.ρ_max
            error("Max regularization exceeded")
        end
    elseif status == :decrease # decrease regularization
        # @info "Regularzation Decreased"
        results.dρ[1] = min(results.dρ[1]/solver.opts.ρ_factor, 1.0/solver.opts.ρ_factor)
        results.ρ[1] = results.ρ[1]*results.dρ[1]*(results.ρ[1]*results.dρ[1]>solver.opts.ρ_min)
    end
end
