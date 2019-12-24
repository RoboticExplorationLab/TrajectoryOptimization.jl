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
            "problem.md",
        ],
        "Documentation" => [
            "discretization.md"
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
