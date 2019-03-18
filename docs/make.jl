using Documenter
using TrajectoryOptimization
using RigidBodyDynamics

makedocs(
    sitename = "TrajectoryOptimization",
    format = :html,
    modules = [TrajectoryOptimization],
    devbranch = "restructure",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/RoboticExplorationLab/TrajectoryOptimization.jl.git",
    devbranch = "restructure",
)
