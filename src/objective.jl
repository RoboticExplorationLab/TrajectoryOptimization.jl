import Base.copy

#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end

"""
$(TYPEDEF)
Cost function of the form
    xₙᵀ Qf xₙ + qfᵀxₙ + ∫ ( xᵀQx + uᵀRu + q⁠ᵀx + rᵀu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
mutable struct QuadraticCost{TM,TH,TV,T} <: CostFunction
    Q::TM                 # Quadratic stage cost for states (n,n)
    R::TM                 # Quadratic stage cost for controls (m,m)
    H::TH                 # Quadratic Cross-coupling for state and controls (n,m)
    q::TV                 # Linear term on states (n,)
    r::TV                 # Linear term on controls (m,)
    c::T                  # constant term
    Qf::TM                # Quadratic final cost for terminal state (n,n)
    qf::TV                # Linear term on terminal state (n,)
    cf::T                 # constant term (terminal)
    function QuadraticCost(Q::TM, R::TM, H::TH, q::TV, r::TV, c::T, Qf::TM, qf::TV, cf::T) where {TM, TH, TV, T}
        if !isposdef(R)
            err = ArgumentError("R must be positive definite")
            throw(err)
        end
        if !ispossemidef(Q)
            err = ArgumentError("Q must be positive semi-definite")
            throw(err)
        end
        if !ispossemidef(Qf)
            err = ArgumentError("Qf must be positive semi-definite")
            throw(err)
        end
        new{TM,TH,TV,T}(Q,R,H,q,r,c,Qf,qf,cf)
    end
end

function LQRCost(Q,R,Qf,xf)
    H = zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = zeros(size(R,1))
    c = 0.5*xf'*Q*xf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf
    return QuadraticCost(Q, R, H, q, r, c, Qf, qf, cf)
end

function taylor_expansion(cost::QuadraticCost, x::AbstractVector{Float64}, u::AbstractVector{Float64})
    m = get_sizes(cost)[2]
    return cost.Q, cost.R, cost.H, cost.Q*x + cost.q, cost.R*u[1:m] + cost.r
end

function taylor_expansion(cost::QuadraticCost, xN::AbstractVector{Float64})
    return cost.Qf, cost.Qf*xN + cost.qf
end

function stage_cost(cost::QuadraticCost, x::AbstractVector, u::AbstractVector)
    0.5*x'cost.Q*x + 0.5*u'*cost.R*u + cost.q'x + cost.r'u + cost.c
end

function stage_cost(cost::QuadraticCost, xN::AbstractVector)
    0.5*xN'cost.Qf*xN + cost.qf'*xN + cost.cf
end

function get_sizes(cost::QuadraticCost)
    return size(cost.Q,1), size(cost.R,1)
end

function copy(cost::QuadraticCost)
    return QuadraticCost(copy(cost.Q), copy(cost.R), copy(cost.H), copy(cost.q), copy(cost.r), copy(cost.c), copy(cost.Qf), copy(cost.qf), copy(cost.cf))
end


"""
$(TYPEDEF)
Cost function of the form
    ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf
"""
struct GenericCost <: CostFunction
    ℓ::Function
    ℓ_aug::Function
    ℓf::Function
    n::Int
    m::Int
    inds::NamedTuple
    hess::Matrix{Float64}  # Pre-allocated hessian
    grad::Vector{Float64}  # Pre-allocated gradient
    z::Vector{Float64}     # Pre-allocated augmented state and control

    function GenericCost(ℓ::Function, ℓf::Function, n::Int, m::Int)
        linds = LinearIndices(zeros(n+m,n+m))
        xinds = 1:n
        uinds = n .+(1:m)
        inds = (x=xinds, u=uinds, xx=linds[xinds,xinds], uu=linds[uinds,uinds], ux=linds[uinds,xinds])
        function ℓ_aug(z)
            x = view(z,xinds)
            u = view(z,uinds)
            ℓ(x,u)
        end
        hess = zeros(n+m,n+m)
        grad = zeros(n+m)
        z = zeros(n+m)
        new(ℓ,ℓ_aug,ℓf,n,m,inds,hess,grad,z)
    end
end

function taylor_expansion(cost::GenericCost, x::AbstractVector{Float64}, u::AbstractVector{Float64})
    inds = cost.inds
    z = cost.z
    z[inds.x] = x
    z[inds.u] = u
    ForwardDiff.gradient!(cost.grad, cost.ℓ_aug, z)
    ForwardDiff.hessian!(cost.hess, cost.ℓ_aug, z)

    q = view(cost.grad,inds.x)
    r = view(cost.grad,inds.u)
    Q = view(cost.hess,inds.xx)
    R = view(cost.hess,inds.uu)
    H = view(cost.hess,inds.ux)
    return Q,R,H,q,r
end

function taylor_expansion(cost::GenericCost, xN::AbstractVector{Float64})
    qf = ForwardDiff.gradient(cost.ℓf, xN)
    Qf = ForwardDiff.hessian(cost.ℓf, xN)
    return Qf,qf
end

stage_cost(cost::GenericCost, x::AbstractVector{Float64}, u::AbstractVector{Float64}) = cost.ℓ(x,u)
stage_cost(cost::GenericCost, xN::AbstractVector{Float64}) = cost.ℓf(xN)

get_sizes(cost::GenericCost) = cost.n, cost.m
copy(cost::GenericCost) = GenericCost(copy(cost.ℓ,cost.ℓ,cost.n,cost.m))


"""
$(TYPEDEF)
Generic type for Objective functions, which are currently strictly Quadratic
"""
abstract type Objective end

"""
$(TYPEDEF)
Defines an objective for an unconstrained optimization problem.
xf does not have to specified. It is provided for convenience when used as part of the cost function (see LQRObjective function)
If tf = 0, the objective is assumed to be minimum-time.
"""
struct UnconstrainedObjective{C} <: Objective
    cost::C
    tf::Float64          # Final time (sec). If tf = 0, the problem is set to minimum time
    x0::Vector{Float64}  # Initial state (n,)
    xf::Vector{Float64}  # (optional) Final state (n,)

    function UnconstrainedObjective(cost::C,tf::Float64,x0,xf=Float64[]) where {C}
        if !isempty(xf) && length(xf) != length(x0)
            throw(ArgumentError("x0 and xf must be the same length"))
        end
        if tf < 0
            throw(ArgumentError("tf must be non-negative"))
        end
        new{C}(cost,tf,x0,xf)
    end
end

"""
$(SIGNATURES)
Minimum time constructor for unconstrained objective
"""
function UnconstrainedObjective(cost::CostFunction,tf::Symbol,x0,xf)
    if tf == :min
        UnconstrainedObjective(cost,0.0,x0,xf)
    else
        err = ArgumentError(":min is the only recognized Symbol for the final time")
        throw(err)
    end
end

function copy(obj::UnconstrainedObjective)
    UnconstrainedObjective(copy(obj.cost),copy(obj.tf),copy(obj.x0),copy(obj.xf))
end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                      CONSTRAINED OBJECTIVE                                   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function is_inplace_function(c::Function, input...)
    q = 100
    iter = 1

    vals = ones(q)
    while iter < 5
        try
            c(vals,input...)
            return true
        catch e
            if e isa MethodError
                return false
            elseif e isa BoundsError
                q *= 10
            else
                throw(e)
            end
            iter += 1
        end
    end
    return false
end

function count_inplace_output(c::Function, input...)
    q0 = 100
    iter = 1

    q = q0
    vals = NaN*(ones(q))
    while iter < 5
        try
            c(vals,input...)
            break
        catch e
            if e isa BoundsError
                q *= 10
                iter += 1
                vals = NaN*(ones(q))
            else
                throw(e)
            end
        end
    end
    p = count(isfinite.(vals))

    return p
end

"""
$(TYPEDEF)
Define a quadratic objective for a constrained optimization problem.

# Constraint formulation
* Equality constraints: `f(x,u) = 0`
* Inequality constraints: `f(x,u) ≥ 0`

"""
struct ConstrainedObjective{C} <: Objective
    cost::C
    tf::Float64           # Final time (sec). If tf = 0, the problem is set to minimum time
    x0::Array{Float64,1}  # Initial state (n,)
    xf::Array{Float64,1}  # Final state (n,)

    # Control Constraints
    u_min::Array{Float64,1}  # Lower control bounds (m,)
    u_max::Array{Float64,1}  # Upper control bounds (m,)

    # State Constraints
    x_min::Array{Float64,1}  # Lower state bounds (n,)
    x_max::Array{Float64,1}  # Upper state bounds (n,)

    # Custom constraints
    cI::Function  # inequality constraint function (inplace)
    cE::Function  # equality constraint function (inplace)

    # Terminal Constraints
    cI_N::Function          # custom terminal inequality constraint
    cE_N::Function          # custom teriminal equality constraint
    use_goal_constraint::Bool  # Use terminal state constraint (true) or terminal cost (false) # TODO I don't think this is used

    p::Int   # Total number of stage constraints
    pI::Int  # Number of inequality constraints
    p_N::Int  # Number of terminal constraints
    pI_N::Int  # Number of terminal inequality constraints

    pI_custom::Int   # Number of custom inequality constraints
    pE_custom::Int   # Number of custom equality constraints
    pI_N_custom::Int # Nubmer of custom terminal inequality constraints
    pE_N_custom::Int # Number of custom terminal equality constraints

    function ConstrainedObjective(cost::C,tf::Real,x0,xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        cI_N, cE_N,
        use_goal_constraint) where {C}

        n,m = get_sizes(cost)
        x = rand(n)
        u = rand(m)

        # Make general inequality/equality constraints inplace
        is_inplace_function(cI,x,u) ? nothing : error("Custom inequality constraints are not inplace")
        pI_custom = count_inplace_output(cI,x,u)

        is_inplace_function(cE,x,u) ? nothing : error("Custom equality constraints are not inplace")
        pE_custom = count_inplace_output(cE,x,u)

        is_inplace_function(cI_N,x) ? nothing : error("Custom terminal inequality constraints are not inplace")
        pI_N_custom = count_inplace_output(cI_N,x)

        is_inplace_function(cE_N,x) ? nothing : error("Custom terminal equality constraints are not inplace")
        pE_N_custom = count_inplace_output(cE_N,x)

        # Validity Tests
        u_max, u_min = _validate_bounds(u_max,u_min,m)
        x_max, x_min = _validate_bounds(x_max,x_min,n)

        # Stage Constraints
        pI = 0
        pE = 0
        pI += count(isfinite, u_min)
        pI += count(isfinite, u_max)
        pI += count(isfinite, x_min)
        pI += count(isfinite, x_max)

        pI += pI_custom
        pE += pE_custom

        p = pI + pE

        # Terminal Constraints
        pI_N = pI_N_custom
        if use_goal_constraint
            if pI_N_custom > 0 || pE_N_custom > 0
                throw(ArgumentError("Can't specify custom terminal constraints with a goal constrait"))
            end
            pE_N = n
        else
            pE_N = 0
        end
        pE_N += pE_N_custom
        p_N = pI_N + pE_N


        new{C}(cost::C, float(tf), x0,xf, u_min, u_max, x_min, x_max, cI, cE, cI_N, cE_N, use_goal_constraint,
            p, pI, p_N, pI_N, pI_custom, pE_custom, pI_N_custom, pE_N_custom)
    end
end

function ConstrainedObjective(cost::C,tf::Symbol,x0,xf,
    u_min, u_max,
    x_min, x_max,
    cI, cE,
    cI_N, cE_N,
    use_goal_constraint) where {C}
    if tf == :min
        ConstrainedObjective(cost,0.0,x0,xf,
            u_min, u_max,
            x_min, x_max,
            cI, cE,
            cI_N, cE_N,
            use_goal_constraint)
    else
        err = ArgumentError(":min is the only recognized Symbol for the final time")
        throw(err)
    end
end


"""
$(SIGNATURES)

Construct a ConstrainedObjective with defaults.

Create a ConstrainedObjective, specifying only the needed fields. All others
will be set to their default, constrained values.

# Constraint formulation
* Equality constraints: `f(x,u) = 0`
* Inequality constraints: `f(x,u) ≥ 0`

# Arguments
* u_min, u_max, x_min, x_max: Upper and lower bounds that can accept either a single scalar or
a vector of size (m,). A scalar will be copied to all states or controls. Values
can be ±Inf.
* cI, cE: Functions for inequality and equality constraints. Must be of the form
`c = f(x,u)`, where `c` is of size (pI_c,) or (pE_c,).
* cI_N, cE_N: Functions for terminal constraints. Must be of the from `c = f(x)`,
where `c` is of size (pI_c_N,) or (pE_c_N,).
"""
function ConstrainedObjective(cost,tf,x0,xf;
    u_min=-ones(get_sizes(cost)[2])*Inf, u_max=ones(get_sizes(cost)[2])*Inf,
    x_min=-ones(get_sizes(cost)[1])*Inf, x_max=ones(get_sizes(cost)[1])*Inf,
    cI=null_constraint, cE=null_constraint,
    cI_N=null_constraint, cE_N=null_constraint,
    use_goal_constraint=true)

    ConstrainedObjective(cost,tf,x0,xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        cI_N, cE_N,
        use_goal_constraint)
end


"$(SIGNATURES) Construct a ConstrainedObjective from an UnconstrainedObjective"
function ConstrainedObjective(obj::UnconstrainedObjective; tf=obj.tf, kwargs...)
    ConstrainedObjective(obj.cost, tf, obj.x0, obj.xf; kwargs...)
end

function copy(obj::ConstrainedObjective)
    ConstrainedObjective(copy(obj.cost),copy(obj.tf),copy(obj.x0),copy(obj.xf),
        u_min=copy(obj.u_min), u_max=copy(obj.u_max), x_min=copy(obj.x_min), x_max=copy(obj.x_max),
        cI=obj.cI, cE=obj.cE,
        use_goal_constraint=obj.use_goal_constraint)
end


"""
$(SIGNATURES)
Updates constrained objective values and returns a new objective.

Only updates the specified fields, all others are copied from the previous
Objective.
"""
function update_objective(obj::ConstrainedObjective;
    cost=obj.cost, tf=obj.tf, x0=obj.x0, xf = obj.xf,
    u_min=obj.u_min, u_max=obj.u_max, x_min=obj.x_min, x_max=obj.x_max,
    cI=obj.cI, cE=obj.cE, cI_N=obj.cI_N, cE_N=obj.cE_N,
    use_goal_constraint=obj.use_goal_constraint)

    ConstrainedObjective(cost,tf,x0,xf,
        u_min=u_min, u_max=u_max,
        x_min=x_min, x_max=x_max,
        cI=cI, cE=cE,
        cI_N=cI_N, cE_N=cE_N,
        use_goal_constraint=use_goal_constraint)

end


null_constraint(c,x,u) = nothing
null_constraint(c,x) = nothing


get_sizes(obj::Objective) = get_sizes(obj.cost)


"""
$(SIGNATURES)
Check max/min bounds for state and control.

Converts scalar bounds to vectors of appropriate size and checks that lengths
are equal and bounds do not result in an empty set (i.e. max > min).

# Arguments
* n: number of elements in the vector (n for states and m for controls)
"""
function _validate_bounds(max,min,n::Int)

    if min isa Real
        min = ones(n)*min
    end
    if max isa Real
        max = ones(n)*max
    end
    if length(max) != length(min)
        throw(DimensionMismatch("u_max and u_min must have equal length"))
    end
    if ~all(max .> min)
        throw(ArgumentError("u_max must be greater than u_min"))
    end
    if length(max) != n
        throw(DimensionMismatch("limit of length $(length(max)) doesn't match expected length of $n"))
    end
    return max, min
end

"""
$(SIGNATURES)
Create unconstrained objective for a problem of the form:
    min (xₙ - xf)ᵀ Qf (xₙ - xf) + ∫ ( (x-xf)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf
    s.t. x(0) = x0
         x(tf) = xf
"""
function LQRObjective(Q,R,Qf,tf,x0,xf)
    cost = LQRCost(Q,R,Qf,xf)
    UnconstrainedObjective(cost,tf,x0,xf)
end


"""
$(SIGNATURES)
Convenience method for getting the stage cost from any objective
"""
function stage_cost(obj::Objective,x::Vector{Float64},u::Vector{Float64})
    stage_cost(obj.cost,x,u)
end

"""
$(SIGNATURES)
    Determine if the constraints are inplace. Returns boolean and number of constraints
"""
function is_inplace_constraints(c::Function,n::Int64,m::Int64)
    x = rand(n)
    u = rand(m)
    q = 100
    iter = 1

    vals = NaN*(ones(q))
    try
        c(vals,x,u)
    catch e
        if e isa MethodError
            return false
        end
    end

    return true
end

function count_inplace_output(c::Function, n::Int, m::Int)
    x = rand(n)
    u = rand(m)
    q0 = 100
    iter = 1

    q = q0
    vals = NaN*(ones(q))
    while iter < 5
        try
            c(vals,x,u)
            break
        catch e
            q *= 10
            iter += 1
            vals = NaN*(ones(q))
        end
    end
    p = count(isfinite.(vals))

    q = q0
    vals = NaN*(ones(q))
    while iter < 5
        try
            c(vals,x)
            break
        catch e
            if e isa MethodError
                p_N = 0
                break
            else
                q *= 10
                iter += 1
                vals = NaN*(ones(q))
            end
        end
    end
    p_N = count(isfinite.(vals))

    return p, p_N
end

function count_inplace_output(c::Function, n::Int)
    x = rand(n)
    q = 100
    iter = 1
    vals = NaN*(ones(q))

    while iter < 5
        try
            c(vals,x)
            break
        catch e
            q *= 10
            iter += 1
            vals = NaN*(ones(q))
        end
    end
    return count(isfinite.(vals))
end
