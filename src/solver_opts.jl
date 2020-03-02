export
    SolverOptions

abstract type AbstractSolverOptions{T<:Real} end
abstract type DirectSolverOptions{T} <: AbstractSolverOptions{T} end

function (::Type{SolverOpts})(opts::Dict{Symbol,<:Any}) where SolverOpts <: AbstractSolverOptions{T} where T
    # add_subsolver_opts!(opts)
    opts_ = filter(x->hasfield(SolverOpts, x[1]), opts)
    SolverOpts(;opts_...)
end

@with_kw mutable struct SolverOptions{T} <: AbstractSolverOptions{T}
    constraint_tolerance::T = 1e-6
    cost_tolerance::T = 1e-4
    cost_tolerance_intermediate::T = 1e-4
    active_set_tolerance::T = 1e-3
    penalty_initial::T = NaN
    penalty_scaling::T = NaN
    iterations::Int = 300
    iterations_inner::Int = 100
    verbose::Bool = false
end

@with_kw mutable struct UnconstrainedSolverOptions{T} <: AbstractSolverOptions{T}
    cost_tolerance::T = 1e-4
    iterations::Int = 300
    verbose::Bool = false
end

function (::Type{<:UnconstrainedSolverOptions})(opts::SolverOptions)
    UnconstrainedSolverOptions(
        cost_tolerance=opts.cost_tolerance,
        iterations=opts.iterations,
        verbose=opts.verbose
    )
end
