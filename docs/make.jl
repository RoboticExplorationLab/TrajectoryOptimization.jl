using Documenter
using TrajectoryOptimization

makedocs(
    sitename = "TrajectoryOptimization",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md",
        "Getting Started" => [
            "model.md",
            "costfunctions.md",
            "constraints.md",
            "creating_problems.md"
        ],
        "Interfaces" => [
            "costfunction_interface.md",
            "constraint_interface.md"
        ],
        "API" => [
            "cost_api.md",
            "constraint_api.md",
            "problem.md",
            "rotations.md",
            "dynamics.md",
            "nlp.md"
        ],
        "Examples" => "examples.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/RoboticExplorationLab/TrajectoryOptimization.jl.git",
)
