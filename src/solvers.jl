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
# function init_constraint_trajectories(constraints::ConstraintSet,n::Int,m::Int,N::Int;
#         μ_init::T=1.,λ_init::T=0.) where T
#     p = num_stage_constraints(constraints)
#     p_N = num_terminal_constraints(constraints)
#
#     # Initialize the partitions
#     c_stage = stage(constraints)
#     c_term = terminal(constraints)
#     c_part = create_partition(c_stage)
#     c_part2 = create_partition2(c_stage,n,m)
#
#     # Create Trajectories
#     C          = [PartedVector(zeros(T,p),c_part)       for k = 1:N-1]
#     ∇C         = [PartedMatrix(zeros(T,p,n+m),c_part2)  for k = 1:N-1]
#     λ          = [PartedVector(ones(T,p),c_part) for k = 1:N-1]
#     μ          = [PartedVector(ones(T,p),c_part) for k = 1:N-1]
#     active_set = [PartedVector(ones(Bool,p),c_part)     for k = 1:N-1]
#
#     C = [C..., PartedVector(T,c_term)]
#     ∇C = [∇C..., PartedMatrix(T,c_term,n,0)]
#     λ = [λ..., PartedVector(T,c_term)]
#     μ = [μ..., PartedArray(ones(T,num_constraints(c_term)), create_partition(c_term))]
#     active_set = [active_set..., PartedVector(Bool,c_term)]
#
#     # Initialize dual and penality values
#     for k = 1:N
#         λ[k] .*= λ_init
#         μ[k] .*= μ_init
#     end
#
#     return C,∇C,λ,μ,active_set
# end
