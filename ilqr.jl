module iLQR
    using RigidBodyDynamics
    using ForwardDiff

    include("model.jl")
    include("solver.jl")
    include("ilqr_algorithm.jl")
end
