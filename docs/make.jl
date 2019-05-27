using Documenter
using TrajectoryOptimization
using RigidBodyDynamics

makedocs(
    sitename = "TrajectoryOptimization",
    format = Documenter.HTML(),
    modules = [TrajectoryOptimization],
    devbranch = "master",
    pages = [
        "index.md",
        "models.md",
        "costfunctions.md",
        "constraints.md",
        "problem.md",
        "solvers.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/RoboticExplorationLab/TrajectoryOptimization.jl.git",
    devbranch = "master",
)
