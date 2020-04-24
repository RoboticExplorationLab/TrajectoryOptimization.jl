"""
    MeritFunction

Measure of progress for a constrained optimization solver.

# Usage
A merit function can be evaluated by treating it as a function:
    merit(solver)
    merit(solver, α)

where `α` is the step size and `merit isa MeritFunction`. If `α` is not passed in or is not
a finite number (e.g. `NaN`) the merit function will retreive the current solution
candidate `x + α⋅δx` using `get_primals(solver)`, otherwise it will update the new step with
`get_primals(solver, α)`.

The derivative can be evaluting using a similar syntax
    derivative(solver)
    derivative(solver, α)

Alteratively, the merit function can generate the functions `ϕ(α)` and `ϕ′(α)` which wrap
the calls to `merit` and `derivative` for convenience, using the methods
    gen_ϕ(merit, solver)
    gen_ϕ′(merit, solver)

# Interface
All merit function must define the following functions
    merit_value(merit::MeritFunction, solver::AbstractSolver, Z)
    merit_derivative(merit::MeritFunction, solver::AbstractSolver, Z)
where `Z` is the representation for the primal variables using by the solver (retrieved using)
`get_primals(solver)`.

"""
abstract type MeritFunction end

(merit::MeritFunction)(solver::AbstractSolver) =
    merit_value(merit, solver, get_primals(solver))
(merit::MeritFunction)(solver::AbstractSolver, α::Real) =
    merit_value(merit, solver, isfinite(α) ? get_primals(solver, α) : get_primals(solver))

derivative(merit::MeritFunction, solver::AbstractSolver) =
    merit_derivative(merit, solver, get_primals(solver))
derivative(merit::MeritFunction, solver::AbstractSolver, α::Real) =
    merit_derivative(merit, solver, isfinite(α) ? get_primals(solver, α) : get_primals(solver))

function gen_ϕ(merit::MeritFunction, solver::AbstractSolver)
    ϕ(α) = isfinite(α) ? merit(solver, α) : merit(solver)
    ϕ() = merit(solver)
end

function gen_ϕ′(merit::MeritFunction, solver::AbstractSolver)
    ϕ′(α) = isfinite(α) ? derivative(merit, solver, α) : derivative(merit, solver)
    ϕ′() = derivative(merit, solver)
end

mutable struct L1Merit{T} <: MeritFunction
    "penalty term"
    μ::T
    "constant for updating μ. Smaller values result in smaller values of μ. 0 < ρ < 1"
    ρ::T
    "margin factor for μ. Larger should result less frequent updates of μ. μ > 1"
    μ_margin::T
    function L1Merit(; μ=1.0, ρ=0.5, μ_margin=1.5)
        @assert 0 < ρ < 1
        p = promote(μ, ρ)
        new{eltype(p)}(μ, ρ, μ_margin)
    end
end

"""
    update_penalty!(merit::MeritFunction, solver, Z)

Update the penalty function of the merit function, if needed.
"""
function update_penalty!(l1::L1Merit, solver::AbstractSolver, Z=get_solution(solver),
        dZ=get_step(solver); recalculate=true)
    g = cost_dgrad(solver, Z, dZ, recalculate=recalculate)  # ∇f'p
    h = cost_dhess(solver, Z, dZ, recalculate=recalculate)  # 0.5*p'∇²L*p
    c = TrajOptCore.norm_violation(solver, recalculate=recalculate, p=1)     # norm(c,1)
    μ = l1.μ
    ρ = l1.ρ
    thresh = (g + max(h, 0))/((1 - ρ)*c)
    if thresh == Inf
        @warn "Infinite merit penalty"
    end
    if μ < thresh
        println("Updated μ")
        l1.μ = thresh * l1.μ_margin
    end
    return nothing
end

function merit_value(l1::L1Merit, solver, Z)
    obj = get_objective(solver)
    _J = get_J(obj)
    cost!(obj, Traj(Z))
    J = sum(_J)

    # Calculate constraint violation
    conSet = get_constraints(solver)
    evaluate!(conSet, Traj(Z))
    c = TrajOptCore.norm_violation(solver.conSet)

    return J + l1.μ*c
end

function merit_derivative(l1::L1Merit, solver, Z)
    dZ = get_step(solver)

    # Calculate cost gradient
    ∇f = cost_dgrad(solver, Z, dZ)

    # Calculate directional derivative of the L1 norm of the constraint violation
    Dc = norm_dgrad(solver, Z, dZ, recalculate=true, p=1)

    return ∇f + l1.μ*Dc
end


abstract type LineSearchCriteria end

@inline function (condition::LineSearchCriteria)(merit::MeritFunction, solver::AbstractSolver,
        α, use_cache=false; kwargs...)
    condition(gen_ϕ(merit, solver), gen_ϕ′(merit, solver), α, use_cache; kwargs...)
end
@inline function (condition::LineSearchCriteria)(ϕ, ϕ′, α, use_cache=false; recalculate::Bool=true)
    sd = sufficient_decrease(condition, ϕ, ϕ′, α, use_cache, recalculate=recalculate)::Bool
    if sd
        cu = curvature(condition, ϕ, ϕ′, α, true, recalculate=recalculate)
        return cu
    else
        return sd
    end
end

mutable struct WolfeConditions{T} <: LineSearchCriteria
    "sufficient decrease condition: closer to 0 requires less decrease (less strict)"
    c1::T
    "curvature condition: closer to 0 requires the derivative to be closer to 0 (more strict)"
    c2::T
    α::T   # last step length
    f0::T  # ϕ(0): initial merit function value
    d0::T  # ϕ′(0): initial derivative
    function WolfeConditions(c1=1e-4, c2=0.9)
        # Default values from Nocedal
        #   c2 = 0.9 for Newton methods
        #   c2 = 0.1 for nonlinear CG methods
        @assert 0.0 < c1 < c2 < 1.0
        c1,c2 = promote(c1,c2)
        new{typeof(c1)}(c1, c2, NaN, NaN, NaN)
    end
end

function sufficient_decrease(condition::WolfeConditions, ϕ, ϕ′, α::Real, use_cache=false;
        recalculate::Bool=true)
    if !use_cache
        condition.f0 = ϕ(0.0)
        condition.d0 = ϕ′(0.0)
    end
    c1,f0,d0 = condition.c1, condition.f0, condition.d0
    condition.α = α  # cache the last step length
    f = recalculate ? ϕ(α) : ϕ()
    f ≤ f0 + c1*α*d0
end

function curvature(condition::WolfeConditions, ϕ, ϕ′, α::Real, use_cache=false;
        recalculate::Bool=true)
    if !use_cache
        d0 = ϕ′(0.0)
    end
    c2, d0 = condition.c2, condition.d0
    condition.α = α  # cache the last step length
    d = recalculate ? ϕ′(α) : ϕ′()
    d ≥ c2*d0
end

"""
    GoldsteinConditions

Not well-suited for quasi-Newton methods
"""
mutable struct GoldsteinConditions{T} <: LineSearchCriteria
    "sufficient decrease condition: closer to 0 less struct"
    c::T
    α::T   # last step length
    f0::T  # ϕ(0): initial merit function value
    d0::T  # ϕ′(0): initial derivative
    function GoldsteinConditions(c=1e-4)
        @assert 0 < c < 0.5
        new{typeof(c)}(c, NaN, NaN, NaN)
    end
end

function sufficient_decrease(condition::GoldsteinConditions, ϕ, ϕ′, α, use_cache=false)
    c,f0,d0 = condition.c, condition.f0, condition.d0
    if !use_cache
        f0 = ϕ(0.0)
        d0 = ϕ′(0.0)
    end
    condition.α = α  # cache the last step length
    ϕ(α) ≤ f0 + c*α*d0
end

function curvature(condition::GoldsteinConditions, ϕ, ϕ′, α, use_cache=false)
    c,f0,d0 = condition.c, condition.f0, condition.d0
    if !use_cache
        f0 = ϕ(0.0)
        d0 = ϕ′(0.0)
    end
    condition.α = α  # cache the last step length
    ϕ(α) ≥ f0 + (1-c)*α*d0
end
