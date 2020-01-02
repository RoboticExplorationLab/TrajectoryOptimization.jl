using Documenter
using TrajectoryOptimization

makedocs(
    sitename = "TrajectoryOptimization",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md",
        "Getting Started" => [
            "models.md",
            "costfunctions.md",
            "constraints.md",
            "creating_problems.md",
            "solving.md"
        ],
        "Interfaces" => [
            "costfunction_interface.md",
            "constraint_interface.md",
            "solver_interface.md"
        ],
        "Documentation" => [
            "model_types.md",
            "discretization.md",
            "cost_api.md",
            "constraint_api.md",
            "problem.md",
            "solvers.md",
            "rotations.md"
        ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/RoboticExplorationLab/TrajectoryOptimization.jl.git",
    deploy_config=Documenter.Travis(),
    devbranch = "v1.3",
)
