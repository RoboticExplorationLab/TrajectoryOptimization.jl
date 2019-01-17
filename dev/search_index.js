var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.jl",
    "category": "page",
    "text": ""
},

{
    "location": "#TrajectoryOptimization.jl-1",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.jl",
    "category": "section",
    "text": "Documentation for TrajectoryOptimization.jl"
},

{
    "location": "#Overview-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Overview",
    "category": "section",
    "text": "The purpose of this package is to provide a testbed for state-of-the-art trajectory optimization algorithms. In general, this package focuses on trajectory optimization problems of the form (put LaTeX here)This package currently implements both indirect and direct methods for trajectory optimization:Iterative LQR (iLQR): indirect method based on differential dynamic programming\nDirect Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solverThe primary focus of this package is developing the iLQR algorithm, although we hope this will extend to many algorithms in the future."
},

{
    "location": "#Getting-Started-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Getting Started",
    "category": "section",
    "text": "In order to set up a trajectory optimization problem, the user needs to create a Model and Objective"
},

{
    "location": "#Creating-a-Model-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Creating a Model",
    "category": "section",
    "text": "There are two ways of creating a model:An in-place analytic function of the form f(ẋ,x,u)\nA URDF"
},

{
    "location": "#TrajectoryOptimization.Solver",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.Solver",
    "category": "type",
    "text": "struct Solver{M<:Model, O<:TrajectoryOptimization.Objective}\n\nType for solver\n\n\n\n\n\n"
},

{
    "location": "#TrajectoryOptimization.Model",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(mech::Mechanism, torques::Array{Bool, 1})\n\nConstructor for an underactuated mechanism, where torques is a binary array that specifies whether a joint is actuated.\n\n\n\n\n\nModel(urdf)\n\n\nConstruct a fully actuated model from a string to a urdf file\n\n\n\n\n\nModel(urdf, torques)\n\n\nConstruct a partially actuated model from a string to a urdf file\n\n\n\n\n\n"
},

{
    "location": "#TrajectoryOptimization.SolverResults",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.SolverResults",
    "category": "type",
    "text": "abstract type SolverResults\n\nAbstract type for the output of solving a trajectory optimization problem\n\n\n\n\n\n"
},

{
    "location": "#Analytic-Models-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Analytic Models",
    "category": "section",
    "text": "To create an analytic model we just need to Solver\nModel\nSolverResults"
},

]}
