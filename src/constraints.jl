using Test

"""
$(SIGNATURES)
    Generate state inequality constraints gs(x) <= 0
    [x - xmax
     xmin - x
     gs_custom(x)]
"""
function generate_state_inequality_constraints(obj::ConstrainedObjective)
    pIs = obj.pIs

    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min

    function state_limits!(c,x)
        c[1:pI_x_max] = (x - obj.x_max )[x_max_active]
        c[pI_x_max+1:pI_x_max+pI_x_min] = (obj.x_min - x)[x_min_active]
    end

    function gs!(c,x)
        state_limits!(view(c,1:pI_x),x)
        if pIs > pI_x
            obj.gs_custom(view(c,pI_x+1:pIs),x)
        end
    end

    return gs!
end

"""
$(SIGNATURES)
    Generate state equality constraints hs(x) = 0
    [hs_custom(x)]
"""
function generate_state_equality_constraints(obj::ConstrainedObjective)
    pEs = obj.pEs
    function hs!(c,x)
        if pEs > 0
            obj.hs_custom(view(c,1:pEs),x)
        end
    end
    return hs!
end

"""
$(SIGNATURES)
    Generate state terminal (inequality) constraint
    [x - xmax
     xmin - x
     gs_custom(x)
     gsN_custom(x)]
"""
function generate_terminal_state_inequality_constraints(obj::ConstrainedObjective)
    pIs = obj.pIs
    pIsN = obj.pIsN

    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min

    function state_limits!(c,x)
        c[1:pI_x_max] = (x - obj.x_max )[x_max_active]
        c[pI_x_max+1:pI_x_max+pI_x_min] = (obj.x_min - x)[x_min_active]
    end

    function gsN!(c,x)
        state_limits!(view(c,1:pI_x),x)
        if pIs > pI_x
            obj.gs_custom(view(c,pI_x+1:pIs),x)
        end
        if pIsN > pIs
            obj.gsN_custom(view(c,pIs+1:pIsN),x)
        end
    end
    return gsN!
end

"""
$(SIGNATURES)
    Generate state terminal (equality) constraint
    [xn - xf
     hs_custom(x)]
"""
function generate_terminal_state_equality_constraints(obj::ConstrainedObjective)
    n = length(obj.x0) # number of states
    pEsN = obj.pEsN

    function hs_terminal!(c,x)
        if obj.use_terminal_state_equality_constraint
            c[1:n] = x - obj.xf

            if pEsN > n
                obj.hs_custom(view(c,n+1:pEsN),x)
            end
        else
            if pEsN > 0
                obj.hs_custom(view(c,1:pEsN),x)
            end
        end
    end
    return hs_terminal!
end

"""
$(SIGNATURES)
    Generate control inequality constraints gc(x) <= 0
    [u - umax
     h - sqrt(dt_max)
     umin - u
     sqrt(dt_min) - h
     gc_custom(u)]
"""
function generate_control_inequality_constraints(obj::ConstrainedObjective; max_dt::Float64=1.0, min_dt::Float64=1.0e-3)
    pIc = copy(obj.pIc)
    m = size(obj.R,1) # number of control inputs
    min_time = obj.tf == 0
    m̄ = m

    if min_time
        m̄ += 1
        pIc += 2
    end

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

    # Inequality on control
    pI_u_max = count(u_max_active)
    pI_u_min = count(u_min_active)
    pI_u = pI_u_max + pI_u_min
    function control_limits!(c,u)
        c[1:pI_u_max] = (u[1:m̄] - u_max)[u_max_active]
        c[pI_u_max+1:pI_u_max+pI_u_min] = (u_min - u[1:m̄])[u_min_active]
    end

    function gc!(c,u)
        control_limits!(view(c,1:pI_u),u[1:m̄])
        if pIc > pI_u
            obj.gc_custom(view(c,pI_u+1:pIc),u[1:m])
        end
    end
    return gc!
end

"""
$(SIGNATURES)
    Generate control equality constraints hc(x) = 0
    [u_infeasible
     h - h'
     hc_custom(x)]
"""
function generate_control_equality_constraints(obj::ConstrainedObjective)
    pEc = copy(obj.pEc)
    m = size(obj.R,1) # number of control inputs
    n = length(obj.x0) # number of states

    min_time = obj.tf == 0
    m̄ = m

    if min_time
        m̄ += 1
    end

    function hc!(c,u)
        infeasible = length(u) != m̄
        if infeasible
            c[1:n] = u[m̄+1:m̄+n]
        end

        if infeasible && min_time
            idx = n+1
        elseif infeasible && !min_time
            idx = n
        elseif !infeasible && min_time
            idx = 1
        else
            idx = 0
        end
        if pEc > 0
            obj.hc_custom(view(c,idx+1:idx+pEc),u[1:m])
        end
        # note: minimum time equality constraint is updated in update constraints
    end

    return hc!
end

"""
$(SIGNATURES)
    Generate coupled state and control inequality constraints gsc(x,u) <= 0
    [gsc_custom(x)]
"""
function generate_coupled_inequality_constraints(obj::ConstrainedObjective)
    (c,x)->nothing
end

"""
$(SIGNATURES)
    Generate coupled state and control equality constraints hsc(x,u) = 0
    [hsc_custom(x)]
"""
function generate_coupled_equality_constraints(obj::ConstrainedObjective)
    (c,x)->nothing
end


# Jacobians
"""
$(SIGNATURES)
    Generate state inequality constraints dgs/dx
"""
function generate_state_inequality_constraint_jacobian(obj::ConstrainedObjective)
    n = length(obj.x0) # number of states
    pIs = obj.pIs

    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min

    In = 1.0*Matrix(I,n,n)

    p_custom = pIs - pI_x
    if p_custom > 0
        J_custom = zeros(p_custom,n)
        g = zeros(p_custom)
        F_custom(J,g,z) = ForwardDiff.jacobian!(J,obj.gs_custom,g,z)
    end

    function gsx!(J,x)
        J[1:pI_x_max, 1:n] = In[x_max_active,:]
        J[pI_x_max+1:pI_x,1:n] = -In[x_min_active,:]
        if p_custom > 0
            F_custom(J_custom,g,x)
            J[pI_x+1:pIs,1:n] = J_custom
        end
    end

    return gsx!
end

"""
$(SIGNATURES)
    Generate state equality constraint Jacobian dhs/dx

"""
function generate_state_equality_constraint_jacobian(obj::ConstrainedObjective)
    n = length(obj.x0) # number of states
    pEs = obj.pEs

    if pEs > 0
        J_custom = zeros(pEs,n)
        h = zeros(pEs)
        F_custom(J,h,z) = ForwardDiff.jacobian!(J,obj.hs_custom,h,z)
    end
    function hsx!(J,x)
        if pEs > 0
            F_custom(J_custom,h,x)
            J[1:pEs,1:n] = J_custom
        end
    end
    return hsx!
end

"""
$(SIGNATURES)
    Generate terminal state inequality constraints dgsN/dx
"""
function generate_terminal_state_inequality_constraint_jacobian(obj::ConstrainedObjective)
    n = length(obj.x0) # number of states
    pIs = obj.pIs
    pIsN = obj.pIsN

    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min

    In = 1.0*Matrix(I,n,n)

    p_custom = pIs - pI_x
    if p_custom > 0
        J_custom = zeros(p_custom,n)
        g = zeros(p_custom)
        F_custom(J,g,z) = ForwardDiff.jacobian!(J,obj.gs_custom,g,z)
    end

    p_custom2 = pIsN - pIs
    if p_custom2 > 0
        J_custom2 = zeros(p_custom2,n)
        g2 = zeros(p_custom2)
        F_custom2(J2,g2,z2) = ForwardDiff.jacobian!(J2,obj.gsN_custom,g2,z2)
    end

    function gsNx!(J,x)
        J[1:pI_x_max, 1:n] = In[x_max_active,:]
        J[pI_x_max+1:pI_x,1:n] = -In[x_min_active,:]
        if p_custom > 0
            F_custom(J_custom,g,x)
            J[pI_x+1:pIs,1:n] = J_custom
        end
        if p_custom2 > 0
            F_custom2(J_custom2,g2,x)
            J[pIs+1:pIsN,1:n] = J_custom2
        end
    end

    return gsNx!
end

"""
$(SIGNATURES)
    Generate state terminal (equality) constraint Jacobian dhsN/dx
"""
function generate_terminal_state_equality_constraint_jacobian(obj::ConstrainedObjective)
    n = length(obj.x0) # number of states
    pEsN = obj.pEsN
    In = 1.0*Matrix(I,n,n)

    p_custom = pEsN - n
    if p_custom > 0
        J_custom = zeros(p_custom,n)
        h = zeros(p_custom)
        F_custom(J,h,z) = ForwardDiff.jacobian!(J,obj.hs_custom,h,z)
    end

    function hsx_terminal!(J,x)
        J[1:n,1:n] = In
        if p_custom > 0
            F_custom(J_custom,h,x)
            J[n+1:pEsN,1:n] = J_custom
        end
    end
    return hsx_terminal!
end

"""
$(SIGNATURES)
    Generate control inequality constraint Jacobian dgc/du

"""
function generate_control_inequality_constraint_jacobian(obj::ConstrainedObjective)
    pIc = copy(obj.pIc)
    m = size(obj.R,1) # number of control inputs
    min_time = obj.tf == 0
    m̄ = m

    if min_time
        m̄ += 1
        pIc += 2
    end

    # Append on min time bounds
    u_max = obj.u_max
    u_min = obj.u_min
    if min_time
        u_max = [u_max; 1.]
        u_min = [u_min; 1.]
    end

    # Mask for active (state|control) constraints
    u_min_active = isfinite.(u_min)
    u_max_active = isfinite.(u_max)

    # Inequality on control
    pI_u_max = count(u_max_active)
    pI_u_min = count(u_min_active)
    pI_u = pI_u_max + pI_u_min

    Im = 1.0*Matrix(I,m̄,m̄)

    p_custom = pIc - pI_u
    if p_custom > 0
        J_custom = zeros(p_custom,m)
        g = zeros(p_custom)
        F_custom(J,g,z) = ForwardDiff.jacobian!(J,obj.gc_custom,g,z)
    end

    function gcu!(J,u)
        J[1:pI_u_max,1:m̄] = Im[u_max_active,1:m̄]
        J[pI_u_max+1:pI_u,1:m̄] = -Im[u_min_active,1:m̄]
        if p_custom > 0
            F_custom(J_custom,g,u[1:m])
            J[pI_u+1:pIc,1:m] = J_custom
        end
    end
    return gcu!
end

"""
$(SIGNATURES)
    Generate control equality constraints Jacobian dhc/du
"""
function generate_control_equality_constraint_jacobian(obj::ConstrainedObjective)
    pEc = copy(obj.pEc)
    m = size(obj.R,1) # number of control inputs
    n = length(obj.x0) # number of states

    min_time = obj.tf == 0
    m̄ = m

    if min_time
        m̄ += 1
    end

    In = 1.0*Matrix(I,n,n)

    if pEc > 0
        J_custom = zeros(pEc,m)
        h = zeros(pEc)
        F_custom(J,h,z) = ForwardDiff.jacobian!(J,obj.hc_custom,h,z)
    end

    function hcu!(J,u)
        infeasible = length(u) != m̄
        if infeasible
            J[1:n,m̄+1:m̄+n] = In

            if min_time
                J[n+1,m̄] = 1.
            end
        else
            if min_time
                J[1,m̄] = 1.
            end
        end

        if infeasible && min_time
            idx = n+1
        elseif infeasible && !min_time
            idx = n
        elseif !infeasible && min_time
            idx = 1
        else
            idx = 0
        end
        if pEc > 0
            F_custom(J_custom,h,u[1:m])
            J[idx+1:idx+pEc,1:m] = J_custom
        end
        # note: minimum time equality constraint is updated in update constraints
    end

    return hcu!
end


function generate_state_inequality_constraints(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_state_equality_constraints(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_terminal_state_inequality_constraints(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_terminal_state_equality_constraints(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_control_inequality_constraints(obj::UnconstrainedObjective; max_dt::Float64=1.0, min_dt::Float64=1.0e-3)
    (c,x)->nothing
end

function generate_control_equality_constraints(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_coupled_inequality_constraints(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_coupled_equality_constraints(obj::UnconstrainedObjective)
    (c,x)->nothing
end


function generate_state_inequality_constraint_jacobian(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_state_equality_constraint_jacobian(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_terminal_state_inequality_constraint_jacobian(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_terminal_state_equality_constraint_jacobian(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_control_inequality_constraint_jacobian(obj::UnconstrainedObjective)
    (c,x)->nothing
end

function generate_control_equality_constraint_jacobian(obj::UnconstrainedObjective)
    (c,x)->nothing
end

##
"""
$(SIGNATURES)
    Generate the Jacobian of a general (coupled) nonlinear constraint function
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
