import Base.copy
using Parameters

abstract type AbstractSolver{T} end
abstract type AbstractSolverOptions{T<:Real} end

include("solvers/ilqr.jl")
include("solvers/augmentedlagrangian.jl")

include("solvers/direct/direct_solvers.jl")
include("solvers/direct/sequential_newton.jl")
include("solvers/direct/dircol.jl")
include("solvers/direct/dircol_ipopt.jl")
include("solvers/direct/moi.jl")
include("solvers/direct/sequential_newton_solve.jl")
include("solvers/direct/projected_newton.jl")

include("solvers/altro.jl")

include("solvers/direct/primals_mintime.jl")
include("solvers/direct/direct_solvers_mintime.jl")
include("solvers/direct/dircol_mintime.jl")
include("solvers/direct/moi_mintime.jl")

# Generic methods for calling solve

function solve!(prob::Problem, opts::AbstractSolverOptions)
    solver = AbstractSolver(prob, opts)
    solve!(prob, solver)
end

function solve(prob::Problem, opts::AbstractSolverOptions)
    prob0 = copy(prob)
    solver = solve!(prob0, opts)
    return prob0, solver
end

function solve(prob::Problem, solver::AbstractSolver)
    prob0 = copy(prob)
    solver = solve!(prob0, solver)
    return prob0, solver
end


jacobian!(prob::Problem{T,Continuous}, solver::AbstractSolver) where T = jacobian!(solver.∇F, prob.model, prob.X, prob.U)
jacobian!(prob::Problem{T,Discrete},   solver::AbstractSolver) where T = jacobian!(solver.∇F, prob.model, prob.X, prob.U, prob.dt)


function check_convergence_criteria(opts_uncon::AbstractSolverOptions{T},cost_tolerance::T,gradient_norm_tolerance::T) where T
    if opts_uncon.cost_tolerance != cost_tolerance
        @warn "Augmented Lagrangian cost tolerance overriding unconstrained solver option\n >>cost tolerance=$cost_tolerance"
    end

    if opts_uncon.gradient_norm_tolerance != gradient_norm_tolerance
        @warn "Augmented Lagrangian gradient norm tolerance overriding unconstrained solver option\n >>gradient norm tolerance=$gradient_norm_tolerance"
    end
    return nothing
end
