module iLQR
    using RigidBodyDynamics
    using ForwardDiff

    export
        Model,
        Solver,
        Objective

    include("model.jl")
    include("solver.jl")
    include("ilqr_algorithm.jl")
end