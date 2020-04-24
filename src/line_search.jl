"""
    LineSearch

Line search algorithm used to find an approximate minimizer of ϕ(α) = f(x + α⋅δx).

# Usage
    line_search(ls::LineSearch, crit::LineSearchCriteria, ϕ, ϕ′, α=1.0, use_cache=false)
"""
abstract type LineSearch end

mutable struct SimpleBacktracking{T} <: LineSearch
    "decrease factor. 0 < ρ < 1. default = 0.5"
    ρ::T
    "maximum number of backtrack steps. default=10"
    max_iters::T
    "minimum step size. default=1e-6"
    α_min::T
    function SimpleBacktracking(;ρ = 0.5, max_iters=10, α_min=1e-6)
        @assert 0 < ρ < 1
        @assert max_iters > 0
        @assert α_min ≥ 0
        p = promote(ρ, max_iters, α_min)
        new{eltype(p)}(ρ, max_iters, α_min)
    end
end

"""
    line_search(ls::LineSearch, crit::LineSearchCriteria, ϕ, ϕ′, α=1.0, use_cache=false)

Find the approximate minimizer of the line search function `ϕ(α) = f(x + α*δx)` using it's
    derivative `ϕ′(α)`, the line search algorithm `ls`, and the termination criteria `crit`.
    Returns either the approximate minimizer `α` if the search is successfull, or 0 if
    the line search fails.

# Optional arguments
- `α`: initial step size
- `use_cache`: pass to `crit` for using values of `ϕ` and `ϕ′` cached in `crit` (typically
values at `α = 0`). Will use cached values after the first check.
"""
function line_search(ls::SimpleBacktracking, crit::LineSearchCriteria,
        merit::MeritFunction, solver::AbstractSolver, α=1.0, use_cache::Bool=false)
    ϕ = gen_ϕ(merit, solver)
    ϕ′ = gen_ϕ′(merit, solver)
    for i = 1:ls.max_iters
        if crit(ϕ, ϕ′, α, use_cache)
            return α  # line search success
        end
        use_cache = true  # used cached values to make criteria search more efficient
        α *= ρ
        if α < α_min
            break
        end
    end
    return zero(α)  # line search failure
end

struct SecondOrderCorrector{T} <: LineSearch
    "decrease factor. 0 < ρ < 1. default = 0.5"
    ρ::T
    "maximum number of backtrack steps. default=10"
    max_iters::T
    "minimum step size. default=1e-6"
    α_min::T
    function SecondOrderCorrector(;ρ = 0.5, max_iters=10, α_min=1e-6)
        @assert 0 < ρ < 1
        @assert max_iters > 0
        @assert α_min ≥ 0
        p = promote(ρ, max_iters, α_min)
        new{eltype(p)}(ρ, max_iters, α_min)
    end
end

function line_search(ls::SecondOrderCorrector, crit::LineSearchCriteria,
        merit::MeritFunction, solver::AbstractSolver, α=1.0, use_cache::Bool=false)
    ρ = ls.ρ; α_min = ls.α_min
    ϕ = gen_ϕ(merit, solver)
    ϕ′ = gen_ϕ′(merit, solver)
    for i = 1:ls.max_iters
        if crit(ϕ, ϕ′, α, use_cache)
            return α  # line search success
        else
            # Update
            get_primals(solver, α)
            # add δẑ to current solution
            second_order_correction!(solver)
            if crit(ϕ, ϕ′, α, true; recalculate=false)  # don't recalculate the current step
                println("second order correction")
                return α
            else
                α *= ρ
            end
            if α < α_min
                break
            end
        end
        use_cache = true  # used cached values to make criteria search more efficient
    end
    return zero(α)
end
