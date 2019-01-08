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

function get_num_terminal_constraints(solver::Solver)
    if solver.state.constrained
        if solver.obj isa ConstrainedObjective
            p_N = solver.obj.p_N
            pI_N = solver.obj.pI_N
            pE_N = p_N - pI_N
        else
            p_N,pI_N,pE_N = 0,0,0
        end
        return p_N,pI_N,pE_N
    else
        return 0,0,0
    end
end

"""
$(SIGNATURES)
Evalutes all inequality and equality constraints (in place) for the current
state and control trajectories
    see: A Novel Augmented Lagrangian Approach for Inequalities and Convergent
    Any-Time Non-Central Updates (Toussaint)
"""
function update_constraints!(res::ConstrainedIterResults, solver::Solver, X=res.X, U=res.U)::Nothing
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)


    c_fun = solver.c_fun

    for k = 1:N-1
        # Update constraints
        c_fun(res.C[k], X[k], U[k])

        # Minimum time special case
        if solver.state.minimum_time
            if k == 1
                res.C[k][p] = 0.0
            end
        end

        # Get active constraint set
        get_active_set!(res,solver,p,pI,k)

        # Update penality-indicator matrices based on active set
        res.Iμ[k] = Diagonal(res.active_set[k].*res.μ[k])
    end

    # Terminal constraint
    c_fun(res.C[N],X[N][1:n])
    get_active_set!(res,solver,p_N,pI_N,N)

    res.Iμ[N] = Diagonal(res.active_set[N].*res.μ[N])

    return nothing
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
    if obj.use_xf_equality_constraint
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
function generate_general_constraint_jacobian(c::Function,p::Int,n::Int64,m::Int64)::Function
    c_aug! = f_augmented!(c,n,m)
    J = zeros(p,n+m)
    S = zeros(n+m)
    cdot = zeros(p)
    F(J,cdot,S) = ForwardDiff.jacobian!(J,c_aug!,cdot,S)

    function c_jacobian(cx,cu,x,u)
        S[1:n] = x[1:n]
        S[n+1:n+m] = u[1:m]
        F(J,cdot,S)
        cx[1:p,1:n] = J[1:p,1:n]
        cu[1:p,1:m] = J[1:p,n+1:n+m]
    end
    return c_jacobian
end

function generate_general_constraint_jacobian(c::Function,p::Int,n::Int64)::Function
    J_N = zeros(p,n)
    xdot = zeros(p)
    F_N(J_N,xdot,x) = ForwardDiff.jacobian!(J_N,c,xdot,x) # NOTE: terminal constraints can only be dependent on state x_N
    function c_jacobian(cx,x)
        F_N(J_N,xdot,x)
        cx .= J_N
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
 (dt equality)]
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
    n̄ = n
    if min_time
         m̄ += 1
         n̄ += 1
    end
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
        c[1:pI_u_max] = (u[1:m̄] - u_max)[u_max_active]
        c[pI_u_max+1:pI_u_max+pI_u_min] = (u_min - u[1:m̄])[u_min_active]
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
        c[1:pI_x_max] = (x[1:n] - obj.x_max )[x_max_active]
        c[pI_x_max+1:pI_x_max+pI_x_min] = (obj.x_min - x[1:n])[x_min_active]
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
    function c_function!(c,x,u)::Nothing
        infeasible = length(u) != m̄

        cI!(view(c,1:pI),x,u[1:m̄])
        if pE_c > 0
            obj.cE(view(c,(1:pE_c).+pI),x,u[1:m])
        end
        if infeasible
            c[pI.+pE_c.+(1:n)] = u[m̄.+(1:n)]
        end
        if min_time
            c[pI+pE_c+(n*infeasible)+1] = u[m̄] - x[n̄]
        end
        return nothing
    end

    # Terminal Constraint
    """
    [cI_N_custom;
     xN-xf;
     cI_E_custom]
    """
    iI = 1:pI_N
    iE = pI_N .+ (1:pE_N)
    function c_function!(c,x)
        if obj.pI_N_custom > 0
            c[iI] = obj.cI_N(c,x)
        end
        if obj.use_xf_equality_constraint
            c[pI_N .+ (1:n)] = x - obj.xf
        elseif obj.pE_N_custom > 0
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
        cI_custom_jacobian! = generate_general_constraint_jacobian(obj.cI, pI_c, n, m)
    end
    if pE_c > 0
        cE_custom_jacobian! = generate_general_constraint_jacobian(obj.cE, pE_c, n, m)
    end

    cx_infeasible = zeros(n,n)
    cu_infeasible = In

    function c_jacobian!(cx::AbstractMatrix, cu::AbstractMatrix, x::AbstractArray,u::AbstractArray)
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
        if min_time
            cx[pI+pE_c+(n*infeasible)+1,n̄] = -1.0
            cu[pI+pE_c+(n*infeasible)+1,m̄] = 1.0
        end
    end

    # Terminal Constraint
    """
    [JI_N_custom;
     In;
     JI_E_custom]
    """

    if obj.pI_N_custom > 0
        cI_N_custom_jacobian! = generate_general_constraint_jacobian(obj.cI_N, obj.pI_N_custom, n)
    end
    if obj.pE_N_custom > 0
        cE_N_custom_jacobian! = generate_general_constraint_jacobian(obj.cE_N, obj.pE_N_custom, n)
    end

    function c_jacobian!(j::AbstractArray,x::AbstractArray)
        if obj.pI_N_custom > 0
            cI_N_custom_jacobian!(view(j,iI,1:n),x)
        end
        if obj.use_xf_equality_constraint
            j[pI_N .+ (1:n),1:n] = In
        elseif obj.pE_N_custom > 0
            cE_N_custom_jacobian!(view(j,iE,1:n),x)
        end
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
    return maximum(norm.(map((x)->x.>0, results.Iμ) .* results.C, Inf))
end

function max_violation(results::UnconstrainedIterResults)
    return 0.0
end
