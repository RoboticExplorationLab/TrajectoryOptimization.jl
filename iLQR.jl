module iLQR
    using RigidBodyDynamics
    using ForwardDiff

    export
        Model,
        Solver,
        Objective,
        SolverOptions

    export
        solve,
        solve_al,
        rollout!,
        forwardpass!,
        backwardpass,
        cost,
        bias

    # include("model.jl")
    include("model.jl")
    include("solver.jl")
    include("ilqr_algorithm.jl")
    include("augmented_lagrange.jl")
    # include("infeasible_start.jl")
end
