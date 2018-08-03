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

    include("model.jl")
    include("integration.jl")
    include("solver.jl")
    include("ilqr_algorithm.jl")
    include("augmented_lagrange.jl")
    include("forensics.jl")
end
