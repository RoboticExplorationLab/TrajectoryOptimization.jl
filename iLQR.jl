module iLQR
    using RigidBodyDynamics
    using ForwardDiff

    export
        Model,
        Solver,
        Objective

    export
        rollout!,
        forwardpass!,
        backwardpass,
        cost

    include("model.jl")
    include("solver.jl")
    include("ilqr_algorithm.jl")
    include("solve_sqrt.jl")
end