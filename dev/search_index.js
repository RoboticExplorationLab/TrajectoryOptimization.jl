var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "TrajectoryOptimization.jl Documentation",
    "title": "TrajectoryOptimization.jl Documentation",
    "category": "page",
    "text": ""
},

{
    "location": "#TrajectoryOptimization.jl-Documentation-1",
    "page": "TrajectoryOptimization.jl Documentation",
    "title": "TrajectoryOptimization.jl Documentation",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationDocumentation for TrajectoryOptimization.jl"
},

{
    "location": "#Overview-1",
    "page": "TrajectoryOptimization.jl Documentation",
    "title": "Overview",
    "category": "section",
    "text": "This package is a testbed for state-of-the-art trajectory optimization algorithms. Trajectory optimization problems are of the form,beginequation\nbeginaligned\n  min_x_0Nu_0N-1 quad  ell_f(x_N) + sum_k=0^N-1 ell_k(x_k u_k dt) \n  textrmst            quad  x_k+1 = f(x_k u_k) \n                                  g_k(x_ku_k) leq 0 \n                                  h_k(x_ku_k) = 0\nendaligned\nendequationThis package currently implements the following methods for solving trajectory optimization problems:Iterative LQR (iLQR): indirect method based on Differential Dynamic Programming\nAL-iLQR: iLQR within an Augmented Lagrangian framework\nDirect Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solver\nALTRO (Augmented Lagrangian Trajectory Optimizer): A novel algorithm developed by the Robotic Exploration Lab at Stanford University, which uses iLQR within an augmented Lagrangian framework combined with a \"Projected Newton\" direct method for solution polishing and enforcement of feasibility.Key features include:Support for general, per-timestep constraints\nForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints\nURDF parsing via [RigidBodyDynamics]"
},

{
    "location": "#Getting-Started-1",
    "page": "TrajectoryOptimization.jl Documentation",
    "title": "Getting Started",
    "category": "section",
    "text": "To set up and solve a trajectory optimization problem with TrajectoryOptimization.jl, the user will go through the following steps:Create a Model\nCreate an Objective\n(Optionally) Add constraints\nInstantiate a Problem\nSelect a solver\nSolve the problem\nAnalyze the solution"
},

{
    "location": "models/#",
    "page": "1. Setting up a Dynamics Model",
    "title": "1. Setting up a Dynamics Model",
    "category": "page",
    "text": ""
},

{
    "location": "models/#model_section-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "1. Setting up a Dynamics Model",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"models.md\"]"
},

{
    "location": "models/#Overview-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "Overview",
    "category": "section",
    "text": "The Model type holds information about the dynamics of the system. All dynamics are assumed to be state-space models of the system of the form y = f(x,u) where y is the state derivative (Continous) or the next state (Discrete), x an n-dimentional state vector, and u in an m-dimensional control input vector. The function f can be any nonlinear function.TrajectoryOptimization.jl textitdiscrete trajectory optimization problem by discretizing the state and control trajectories, which requires discretizing the dynamics, i.e., turning the continuous time differential equation into a discrete time difference equation of the form x[k+1] = f(x[k],u[k]), where k is the time step. There many methods of performing this discretization, and TrajectoryOptimization.jl offers several of the most common methods.Sometimes is it convenient to write down the difference equation directly, rather than running a differential equation through a discretizing integration method. TrajectoryOptimization.jl offers method deal directly with either continuous differential equations, or discrete difference equations.The Model type is parameterized by the DynamicsType, which is either Continuous, or Discrete. The models holds the equation f and its Jacobian, ∇f, along with the dimensions of the state and control vectors.Models can be created by writing down the dynamics analytically or be generated from a URDF file via RigidBodyDynamics.jl."
},

{
    "location": "models/#Continuous-Models-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "Continuous Models",
    "category": "section",
    "text": ""
},

{
    "location": "models/#From-analytical-function-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "From analytical function",
    "category": "section",
    "text": "Let\'s start by writing down a dynamics function for a simple pendulum with state [θ; ω] and a torque control inputfunction pendulum_dynamics!(ẋ,x,u)\n    m = 1.\n    l = 0.5\n    b = 0.1\n    lc = 0.5\n    J = 0.25\n    g = 9.81\n    ẋ[1] = x[2]\n    ẋ[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/J\nendNote that the function is in-place, in that it writes the result to the first argument. It is also good practice to concretely specify the location to write to rather than using something like ẋ[1:end] or ẋ[:].Notice that we had to specify a handful of constants when writing down the dynamics. We could have initialized them outside the scope of the function (which may result in global variables, so be careful!) or we can pass them in as a NamedTuple of parameters:function pendulum_dynamics_params!(xdot,x,u,p)\n    xdot[1] = x[2]\n    xdot[2] = (u[1] - p.m * p.g * p.lc * sin(x[1]) - p.b*x[2])/p.J\nendWe can now create our model using our analytical dynamics function with or without the parameters tuplen,m = 2,1\nmodel = Model(pendulum_dynamics!, n, m)\n\nparams = (m=1, l=0.5, b=0.1, lc=0.5, J=0.25, g=9.81)\nmodel = Model(pendulum_dynamics_params!, n, m, params)"
},

{
    "location": "models/#With-analytical-Jacobians-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "With analytical Jacobians",
    "category": "section",
    "text": "Since we have a very simple model, writing down an analytical expression of the Jacobian is straightforward:function pendulum_jacobian!(Z,x,u)\n    m = 1.\n    l = 0.5\n    b = 0.1\n    lc = 0.5\n    J = 0.25\n    g = 9.81\n\n    Z[1,1] = 0                    # ∂θdot/∂θ\n    Z[1,2] = 1                    # ∂θdot/∂ω\n    Z[1,3] = 0                    # ∂θ/∂u\n    Z[2,1] = -m*g*lc*cos(x[1])/J  # ∂ωdot/∂θ\n    Z[2,2] = -b/J                 # ∂ωdot/∂ω\n    Z[2,3] = 1/J                  # ∂ωdot/∂u\nend\n\nfunction pendulum_jacobian_params!(Z,x,u,p)\n    Z[1,1] = 0                                    # ∂θdot/∂θ\n    Z[1,2] = 1                                    # ∂θdot/∂ω\n    Z[1,3] = 0                                    # ∂θ/∂u\n    Z[2,1] = -p.m * p.g * p.lc * cos(x[1]) / p.J  # ∂ωdot/∂θ\n    Z[2,2] = -p.b / p.J                           # ∂ωdot/∂ω\n    Z[2,3] = 1/p.J                                # ∂ωdot/∂u\nendWe can then pass these functions into the model instead of using ForwardDiff to calculate themmodel = Model(pendulum_dynamics!, pendulum_jacobian!, n, m)\nmodel = Model(pendulum_dynamics_params!, pendulum_jacobian_params!, n, m, params)"
},

{
    "location": "models/#URDF-Files-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "URDF Files",
    "category": "section",
    "text": "Instead of writing down the dynamics explicity, we can import the dynamics from geometry specified in a URDF model using RigidBodyDynamics.jl. Let\'s say we have a URDF file for a double pendulum and don\'t want to bother writing down the dynamics, then we can create a model using any of the following methodsusing RigidBodyDynamics\n# From a string\nurdf = \"doublependulum.urdf\"\nmodel = Model(urdf)\n\n# From a RigidBodyDynamics `Mechanism` type\nmech = parse_urdf(urdf)  # return a Mechanism type\nmodel = Model(mech)Now let\'s say we want to control an underactuated acrobot, which can only control the second joint. We can pass in a vector of Booleans to specify which of the joints are \"active.\"joints = [false,true]\n\n# From a string\nurdf = \"doublependulum.urdf\"\nmodel = Model(urdf,joints)\n\n# From a RigidBodyDynamics `Mechanism` type\nmech = parse_urdf(urdf)  # return a Mechanism type\nmodel = Model(mech,joints)"
},

{
    "location": "models/#TrajectoryOptimization.AnalyticalModel",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.AnalyticalModel",
    "category": "type",
    "text": "struct AnalyticalModel{M, D} <: Model{M,D}\n\nDynamics model Holds all information required to uniquely describe a dynamic system, including a general nonlinear dynamics function of the form ẋ = f(x,u), where x ∈ ℜⁿ are the states and u ∈ ℜᵐ are the controls. Dynamics function, f, should be of the form     f(ẋ,x,u,p) for Continuous models, where ẋ is the state derivative     f(ẋ,x,u,p,dt) for Discrete models, where ẋ is the state at the next time step     and x is the state vector, u is the control input vector, and p is an optional NamedTuple of static parameters (mass, gravity, etc.) Dynamics jacobians, ∇f, should be of the form     ∇f(Z,x,u,p) for Continuous models, and     ∇f(Z,x,u,,p,dt) for discrete models     where p is the same NamedTuple of parameters used in the dynamics\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.RBDModel",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.RBDModel",
    "category": "type",
    "text": "struct RBDModel{M, D} <: Model{M,D}\n\nRigidBodyDynamics model. Wrapper for a RigidBodyDynamics Mechanism\n\n\n\n\n\n"
},

{
    "location": "models/#A-note-on-Model-types-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "A note on Model types",
    "category": "section",
    "text": "While the constructors look very similar, URDF models actually return a slightly different type than the the analytical ones of the first section. Analytical models are represented byAnalyticalModelwhereas those created from a URDF are represented byRBDModelwhich explicitly stores the Mechanism internally."
},

{
    "location": "models/#Discrete-Models-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "Discrete Models",
    "category": "section",
    "text": "The previous methods all generate models with continuous dynamics (note that all of the models returned above will be of the type Model{Continuous}. In order to perform trajectory optimization we need to have discrete dynamics. Typically, we will form the continuous dynamics as we did above and then use a particular integration scheme to discretize it. Alternatively, we may know the analytical expression for the discrete dynamics."
},

{
    "location": "models/#From-a-continuous-model-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "From a continuous model",
    "category": "section",
    "text": "Assuming we have a model of type Model{Continuous}, we can discretize as follows:model_discrete = Model{Discrete}(model,discretizer)where discretizer is a function that returns a discretized version of the continuous dynamics. TrajectoryOptimization.jl offers the following integration schemesmidpoint\nrk3 (Third Order Runge-Kutta)\nrk4 (Fourth Order Runge-Kutta)To create a discrete model of the pendulum with fourth order Runge-Kutta integration we would do the following# Create the continuous model (any of the previously mentioned methods would work here)\nparams = (m=1, l=0.5, b=0.1, lc=0.5, J=0.25, g=9.81)\nmodel = Model(pendulum_dynamics_params!, n, m, params)\n\n# Discretize the continuous model\nmodel_discrete = Model{Discrete}(model,rk4)"
},

{
    "location": "models/#From-an-analytical-expression-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "From an analytical expression",
    "category": "section",
    "text": "Very little changes when specifying an analytical discrete model. The only change is that both the dynamics and Jacobian functions must take in the time step dt as an argument. Here is an example for the pendulum using Euler integration for simplicityfunction pendulum_discrete!(xdot,x,u,dt)\n    pendulum_dynamics!(xdot,x,u)\n    xdot .= x + xdot*dt\nendThe Jacobian is similarly specified as a function of the form ∇f(Z,x,u,dt). We don\'t give an example for brevity.The model is then created in a similar fashion to the above methodsmodel_discrete = AnalyticalModel{Discrete}(pendulum_discrete!,n,m)\nmodel_discrete = AnalyticalModel{Discrete}(pendulum_discrete_params!,n,m,params)  # if we defined this\n\n# If we defined the Jacobian function we also create it as\nmodel_discrete = AnalyticalModel{Discrete}(pendulum_discrete!, pendulum_discrete_jacobian!, n, m)\nmodel_discrete = AnalyticalModel{Discrete}(pendulum_discrete_params!, pendulum_discrete_jacobian!, n, m, params)"
},

{
    "location": "models/#Methods-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "Methods",
    "category": "section",
    "text": "Models are pretty basic types and don\'t offer much functionality other than specifying the dynamics. We can get the number of stage and controls as followsn = model.n\nm = model.m"
},

{
    "location": "models/#Testing-the-dynamics-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "Testing the dynamics",
    "category": "section",
    "text": "It\'s often useful to test out the dynamics or its Jacobians. We must pre-allocate the arraysxdot = zeros(n)\nZ = zeros(n,m)Or create a partitioned vector and Jacobian for easy access to separate state and control jacobiansxdot = PartedVector(model)\nZ = PartedMatrix(model)Once the arrays are allocated, we can call using evaluate! and jacobian! (increments the evaluation count, recommended)x,u = rand(x), rand(u)  # or some known test inputs\nevaluate!(xdot,model,x,u)\njacobian!(Z,model,x,u)If we created a partitioned Jacobian using PartedMatrix(model), we can access the different piecesfdx = Z.x   # ∂f/∂x\nfdu = Z.u   # ∂f/∂u\nfdt = Z.dt  # ∂f/∂dt"
},

{
    "location": "models/#API-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "API",
    "category": "section",
    "text": ""
},

{
    "location": "models/#TrajectoryOptimization.Model",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(f::Function, n::Int64, m::Int64) -> TrajectoryOptimization.AnalyticalModel{TrajectoryOptimization.Nominal,Continuous}\nModel(f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{TrajectoryOptimization.Nominal,Continuous}\n\n\nCreate a dynamics model, using ForwardDiff to generate the dynamics jacobian, with parameters Dynamics function passes in parameters:     f(ẋ,x,u,p)     where p in NamedTuple of parameters\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(f::Function, n::Int64, m::Int64, p::NamedTuple) -> TrajectoryOptimization.AnalyticalModel{TrajectoryOptimization.Nominal,Continuous}\nModel(f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{TrajectoryOptimization.Nominal,Continuous}\n\n\nCreate a dynamics model, using ForwardDiff to generate the dynamics jacobian, without parameters Dynamics function of the form:     f(ẋ,x,u)\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(f::Function, ∇f::Function, n::Int64, m::Int64, p::NamedTuple) -> TrajectoryOptimization.AnalyticalModel{TrajectoryOptimization.Nominal,Continuous}\nModel(f::Function, ∇f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{TrajectoryOptimization.Nominal,Continuous}\n\n\nCreate a dynamics model with an analytical Jacobian, with parameters Dynamics functions pass in parameters:     f(ẋ,x,u,p)     ∇f(Z,x,u,p)     where p in NamedTuple of parameters\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(f::Function, ∇f::Function, n::Int64, m::Int64) -> TrajectoryOptimization.AnalyticalModel{TrajectoryOptimization.Nominal,Continuous}\nModel(f::Function, ∇f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{TrajectoryOptimization.Nominal,Continuous}\n\n\nCreate a dynamics model with an analytical Jacobian, without parameters Dynamics functions pass of the form:     f(ẋ,x,u)     ∇f(Z,x,u)\n\n\n\n\n\n"
},

{
    "location": "models/#From-Analytical-Function-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "From Analytical Function",
    "category": "section",
    "text": "The following constructors can be used to create a model from an analytic function, with or without parameters or analyical JacobiansModel(f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())\nModel(f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())\nModel(f::Function, ∇f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())\nModel(f::Function, ∇f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())"
},

{
    "location": "models/#TrajectoryOptimization.Model-Tuple{Mechanism,Array}",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "method",
    "text": "Model(mech, torques)\n\n\nModel(mech::Mechanism, torques::Array{Bool, 1}) Constructor for an underactuated mechanism, where torques is a binary array that specifies whether a joint is actuated.\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model-Tuple{Mechanism}",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "method",
    "text": "Model(mech)\n\n\nConstruct model from a Mechanism type from RigidBodyDynamics\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model-Tuple{String}",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "method",
    "text": "Model(urdf)\n\n\nConstruct a fully actuated model from a string to a urdf file\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model-Tuple{String,Array{Float64,1}}",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "method",
    "text": "Model(urdf, torques)\n\n\nConstruct a partially actuated model from a string to a urdf file, where torques is a binary array that specifies whether a joint is actuated.\n\n\n\n\n\n"
},

{
    "location": "models/#From-URDF-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "From URDF",
    "category": "section",
    "text": "The following constructors can be used to create a Model from a URDF fileModel(mech::Mechanism, torques::Array)\nModel(mech::Mechanism)\nModel(urdf::String)\nModel(urdf::String,torques::Array{Float64,1})"
},

{
    "location": "models/#TrajectoryOptimization.evaluate!-Union{Tuple{M}, Tuple{AbstractArray{T,1} where T,Model{M,Continuous},AbstractArray{T,1} where T,AbstractArray{T,1} where T}} where M<:TrajectoryOptimization.ModelType",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.evaluate!",
    "category": "method",
    "text": "evaluate!(ẋ, model::Model{M,Continuous}, x, u)\n\nEvaluate the continuous dynamics at state x and control u Keeps track of the number of evaluations\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.evaluate!-Union{Tuple{T}, Tuple{M}, Tuple{AbstractArray{T,1} where T,Model{M,Discrete},AbstractArray{T,1} where T,AbstractArray{T,1} where T,T}} where T where M<:TrajectoryOptimization.ModelType",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.evaluate!",
    "category": "method",
    "text": "evaluate!(ẋ, model::Model{M,Discrete}, x, u, dt)\n\nEvaluate the discrete dynamics at state x and control u and time step dt Keeps track of the number of evaluations\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.jacobian!-Union{Tuple{M}, Tuple{AbstractArray{T,2} where T,Model{M,Continuous},AbstractArray{T,1} where T,AbstractArray{T,1} where T}} where M<:TrajectoryOptimization.ModelType",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.jacobian!",
    "category": "method",
    "text": "jacobian!(Z, model::Model{M,Continuous}, x, u)\n\nEvaluate the dynamics Jacobian simultaneously at state x and control x Keeps track of the number of evaluations\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.jacobian!-Union{Tuple{T}, Tuple{M}, Tuple{AbstractArray{T,N} where N,Model{M,Discrete},AbstractArray{T,1} where T,AbstractArray{T,1} where T,T}} where T<:AbstractFloat where M<:TrajectoryOptimization.ModelType",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.jacobian!",
    "category": "method",
    "text": "jacobian!(Z, model::Model{M,Discrete}, x, u, dt)\n\nEvaluate the dynamics Jacobian simultaneously at state x and control x Keeps track of the number of evaluations\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.jacobian!-Union{Tuple{T}, Tuple{M}, Tuple{Array{PartedArray{T,2,Array{T,2},P} where P,1},Model{M,Discrete},Array{Array{T,1},1},Array{Array{T,1},1},Array{T,1}}} where T where M<:TrajectoryOptimization.ModelType",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.jacobian!",
    "category": "method",
    "text": "jacobian!(Z::PartedVecTrajectory, model::Model{M,Discrete}, X, U, dt)\n\nEvaluate discrete dynamics Jacobian along entire trajectory\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.evals-Tuple{Model}",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.evals",
    "category": "method",
    "text": "evals(model)\n\n\nReturn the number of dynamics evaluations \n\n\n\n\n\n"
},

{
    "location": "models/#Base.reset-Tuple{Model}",
    "page": "1. Setting up a Dynamics Model",
    "title": "Base.reset",
    "category": "method",
    "text": "reset(model)\n\n\nReset the evaluation counts for the model \n\n\n\n\n\n"
},

{
    "location": "models/#Evaluation-methods-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "Evaluation methods",
    "category": "section",
    "text": "evaluate!(ẋ::AbstractVector,model::Model{M,Continuous},x::AbstractVector,u::AbstractVector) where M <: ModelType\nevaluate!(ẋ::AbstractVector,model::Model{M,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T}\njacobian!(Z::AbstractMatrix,model::Model{M,Continuous},x::AbstractVector,u::AbstractVector) where M <: ModelType\njacobian!(Z::AbstractArray{T},model::Model{M,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T <: AbstractFloat}\njacobian!(Z::PartedMatTrajectory{T},model::Model{M,Discrete},X::VectorTrajectory{T},U::VectorTrajectory{T},dt::Vector{T}) where {M<:ModelType,T}evals(model::Model)\nreset(model::Model)"
},

{
    "location": "costfunctions/#",
    "page": "2. Setting up an Objective",
    "title": "2. Setting up an Objective",
    "category": "page",
    "text": ""
},

{
    "location": "costfunctions/#objective_section-1",
    "page": "2. Setting up an Objective",
    "title": "2. Setting up an Objective",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"costfunctions.md\"]"
},

{
    "location": "costfunctions/#Overview-1",
    "page": "2. Setting up an Objective",
    "title": "Overview",
    "category": "section",
    "text": "All trajectory optimization problems require a cost function at each stage of the trajectory. Cost functions must be scalar-valued. We assume general cost functions of the form,ell_f(x_N) + sum_k=1^N-1 ell_k(x_ku_k)It is very important to note that ell_k(x_ku_k) is ONLY a function of x_k and u_k, i.e. no coupling across time-steps is permitted. This is a requirement for Differential Dynamic Programming methods such as iLQR, but could be relaxed for methods that parameterize both states and controls, such as DIRCOL. In general, any coupling between adjacent time-steps can be resolved by augmenting the state and defining the appropriate dynamics (this is the method we use to solve minimum time problems).In general, trajectory optimization will take a second order Taylor series approximation of the cost function, resulting in a quadratic cost function of the formx_N^T Q_f x_N + q_f^T x_N + sum_k=1^N-1 x_k^T Q_k x_k + q_k^T x_k + u_k^T R_k u_k + r_k^T u_k + u_k^T H_k x_kThis type of quadratic cost is typical for trajectory optimization problems, especially when Q is positive semi-definite and R is positive definite, which is strictly convex. These problem behave well and reduce the computational requirements of taking second-order Taylor series expansions of the cost at each iteration.In TrajectoryOptimization.jl we differentiate between the entire objective and the cost functions at each time step. We use Objective to describe the function that is being minimized, which typically consists of a sum of cost functions, with potentially some additional terms (as is the case with augmented Lagrangian objectives). Describing the Objective as a sum of individual functions allows the solvers to more efficiently compute the gradient and Hessian of the entire cost, which is block-diagonal given the Markovianity of the problem."
},

{
    "location": "costfunctions/#Cost-functions-1",
    "page": "2. Setting up an Objective",
    "title": "Cost functions",
    "category": "section",
    "text": "There are several different cost function types that all inherit from CostFunction. The following sections detail the various methods for instantiating these cost function types."
},

{
    "location": "costfunctions/#TrajectoryOptimization.QuadraticCost",
    "page": "2. Setting up an Objective",
    "title": "TrajectoryOptimization.QuadraticCost",
    "category": "type",
    "text": "mutable struct QuadraticCost{T} <: TrajectoryOptimization.CostFunction\n\nCost function of the form     1/2xₙᵀ Qf xₙ + qfᵀxₙ +  ∫ ( 1/2xᵀQx + 1/2uᵀRu + xᵀHu + q⁠ᵀx  rᵀu ) dt from 0 to tf R must be positive definite, Q and Qf must be positive semidefinite\n\nConstructor use any of the following constructors:\n\nQuadraticCost(Q, R, H, q, r, c)\nQuadraticCost(Q, R; H, q, r, c)\nQuadraticCost(Q, q, c)\n\nAny optional or omitted values will be set to zero(s).\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#TrajectoryOptimization.LQRCost",
    "page": "2. Setting up an Objective",
    "title": "TrajectoryOptimization.LQRCost",
    "category": "function",
    "text": "LQRCost(Q, R, xf)\n\n\nCost function of the form (x-x_f)^T Q (x_x_f) + u^T R u R must be positive definite, Q must be positive semidefinite\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#TrajectoryOptimization.LQRCostTerminal",
    "page": "2. Setting up an Objective",
    "title": "TrajectoryOptimization.LQRCostTerminal",
    "category": "function",
    "text": "LQRCostTerminal(Qf, xf)\n\n\nCost function of the form (x-x_f)^T Q (x_x_f) Q must be positive semidefinite\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#TrajectoryOptimization.LQRObjective",
    "page": "2. Setting up an Objective",
    "title": "TrajectoryOptimization.LQRObjective",
    "category": "function",
    "text": "LQRObjective(Q, R, Qf, xf, N)\n\nCreate an objective of the form (x_N - x_f)^T Q_f (x_N - x_f) + sum_k=0^N-1 (x_k-x_f)^T Q (x_k-x_f) + u_k^T R u_k\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#Quadratic-Costs-1",
    "page": "2. Setting up an Objective",
    "title": "Quadratic Costs",
    "category": "section",
    "text": "Quadratic costs are the most standard cost function and excellent place to start. Let\'s assume we are creating an LQR tracking cost of the form(x_N - x_f)^T Q_f (x_N - x_f) + sum_k=1^N-1 (x_k - x_f)^T Q (x_k - x_f) + u_k^T R u_kfor the simple pendulum with the goal of doing a swing-up. To do this we have very convenient constructors LQRCost and LQRCostTerminal:using LinearAlgebra\nn,m = 2,1\nQ = Diagonal(0.1I,n)\nR = Diagonal(0.1I,m)\nQf = Diagonal(1000I,n)\nxf = [π,0]\ncostfun = LQRCost(Q,R,Qf)\ncostfun_term = LQRCostTerminal(Qf,xf)It is HIGHLY recommended to specify any special structure, such as Diagonal, especially since these matrices are almost always diagonal.This constructor actually does a simple conversion to turn our cost function into a generic quadratic cost function. We could do this ourselves:H = zeros(m,n)\nq = -Q*xf\nr = zeros(m)\nc = xf\'Q*xf/2\nqf = -Qf*xf\ncf = xf\'Qf*xf/2\ncostfun      = QuadraticCost(Q, R, H, q, r, c)\ncostfun_term = QuadraticCost(Qf, R*0, H, qf, r*0, cf)The QuadraticCost constructor also supports keyword arguments and one that allows for only Q,q and c.:costfun      = QuadraticCost(Q, R, q=q, c=c)\ncostfun_term = QuadraticCost(Q, q, c)Once we have defined the cost function, we can create an objective for our problem by simply copying over all time steps (except for the terminal).# Create an objective from a single cost function\nN = 51\nobj = Objective(costfun, costfun_term, N)There\'s also a convenient constructor that builds an LQRObjectiveobj = LQRObjective(Q, R, Qf, xf, N)QuadraticCost\nLQRCost\nLQRCostTerminal\nLQRObjective"
},

{
    "location": "costfunctions/#TrajectoryOptimization.GenericCost",
    "page": "2. Setting up an Objective",
    "title": "TrajectoryOptimization.GenericCost",
    "category": "type",
    "text": "struct GenericCost <: TrajectoryOptimization.CostFunction\n\nCost function of the form     ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#Generic-Costs-(Experimental)-1",
    "page": "2. Setting up an Objective",
    "title": "Generic Costs (Experimental)",
    "category": "section",
    "text": "For general, non-linear cost functions use GenericCost. Generic cost functions must define their second-order Taylor series expansion, either automatically using ForwardDiff or analytically.Let\'s say we wanted to use a nonlinear objective for the pendulumcos(theta_N) + omega_N^2 sum_k=1^N-1 cos(theta_k) + u_k^T R u_k + Q ω^2which is small when θ = π, encouraging swing-up.We define the cost function by defining ℓ(x,u) and ℓ(x)# Define the stage and terminal cost functions\nfunction mycost(x,u)\n    R = Diagonal(0.1I,1)\n    Q = 0.1\n    return cos(x[1] + u\'R*u + Q*x[2]^2)\nend\nfunction mycost(xN)\n    return cos(xN[1]) + xN[2]^2\nend\n\n# Create the nonlinear cost function\nnlcost = GenericCost(mycost,mycost,n,m)This will use ForwardDiff to generate the gradient and Hessian needed for the 2nd order expansion.Performance-wise, it will be faster to specify the Jacobian analytically (which could also use ForwardDiff for part of it). We just need to define the following functionshess: multiple-dispatch function of the form,   Q,R,H = hess(x,u) with sizes (n,n), (m,m), (m,n)   Qf = hess(xN) with size (n,n)\ngrad: multiple-dispatch function of the form,   q,r = grad(x,u) with sizes (n,), (m,)   qf = grad(x,u) with size (n,)Here\'s an example for the nonlinear cost function we used before# Define the gradient and Hessian functions\nR = Diagonal(0.1I,m)\nQ = 0.1\nfunction hess(x,u)\n    n,m = length(x),length(u)\n    Qexp = Diagonal([-cos(x[1]), 2Q])\n    Rexp = 2R\n    H = zeros(m,n)\n    return Qexp,Rexp,Hexp\nend\nfunction hess(x)\n    return Diagonal([-cos(x[1]), 2])\nend\nfunction grad(x,u)\n    q = [-sin(x[1]), 2Q*x[2]]\n    r = 2R*u\n    return q,r\nend\nfunction grad(x)\n    return [-sin(x[1]), 2*x[2]]\nend\n\n# Create the cost function\nnlcost = GenericCost(mycost, mycost, grad, hess, n, m)Since our cost function is defined at both stage and terminal steps, we can simply copy it over all time steps to create an objective:# Create objective\nN = 51\nnlobj = Objective(nlcost, N)GenericCost"
},

{
    "location": "costfunctions/#Cost-Function-Interface-1",
    "page": "2. Setting up an Objective",
    "title": "Cost Function Interface",
    "category": "section",
    "text": "All cost functions are required to define the following methodsstage_cost(cost, x, u)\nstage_cost(cost, xN)\ncost_expansion!(Q::Expansion, cost, x, u)\ncost_expansion(Q::Expansion, cost, xN)and inherit from CostFunction.The Expansion type is defined in the next section. This common interface allows the Objective to efficiently dispatch over cost functions to compute the overall cost and Taylor series expansion (i.e. gradient and Hessian)."
},

{
    "location": "costfunctions/#Expansion-Type-1",
    "page": "2. Setting up an Objective",
    "title": "Expansion Type",
    "category": "section",
    "text": "The expansion type stores the pieces of the second order Taylor expansion of the cost.If we store the expansion as Q, then Q.x is the partial with respect to the control, Q.xu is the partial with respect to x and u, etc."
},

{
    "location": "costfunctions/#Objectives-1",
    "page": "2. Setting up an Objective",
    "title": "Objectives",
    "category": "section",
    "text": ""
},

{
    "location": "costfunctions/#Constructors-1",
    "page": "2. Setting up an Objective",
    "title": "Constructors",
    "category": "section",
    "text": "Objectives can be created by copying a single cost function over all time stepsObjective(cost::CostFunction, N::Int)or uniquely specifying the terminal cost functionObjective(cost::CostFunction, cost_terminal::CostFunction, N::Int)or by explicitly specifying a list of cost functionsObjective(costfuns::Vector{<:CostFunction})"
},

{
    "location": "costfunctions/#Methods-1",
    "page": "2. Setting up an Objective",
    "title": "Methods",
    "category": "section",
    "text": "Constraints extends the methods on CostFunction to the whole trajectorycost(obj, X, U)\ncost_expansion!(Q::Vector{Expansion}, obj, X, U)where X and U are the state and control trajectories."
},

{
    "location": "costfunctions/#TrajectoryOptimization.cost",
    "page": "2. Setting up an Objective",
    "title": "TrajectoryOptimization.cost",
    "category": "function",
    "text": "cost(obj::Objective, X::Vector, U::Vector, dt::Vector)\n\nCalculate cost over entire state and control trajectories\n\n\n\n\n\nEvaluate the current cost for the problem\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#TrajectoryOptimization.stage_cost",
    "page": "2. Setting up an Objective",
    "title": "TrajectoryOptimization.stage_cost",
    "category": "function",
    "text": "stage_cost(cost, x, u)\n\n\nEvaluate the cost at state x and control u\n\n\n\n\n\nstage_cost(cost, xN)\n\n\nEvaluate the cost at the terminal state xN\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#TrajectoryOptimization.cost_expansion!",
    "page": "2. Setting up an Objective",
    "title": "TrajectoryOptimization.cost_expansion!",
    "category": "function",
    "text": "cost_expansion!(Q, cost, x, u)\n\n\nEvaluate the second order expansion at state x and control u\n\n\n\n\n\ncost_expansion!(Q, cost, xN)\n\n\nEvaluate the second order expansion at the terminal state xN\n\n\n\n\n\ncost_expansion!(Q, obj, X, U, dt)\n\n\nCompute the second order Taylor expansion of the cost for the entire trajectory\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#API-1",
    "page": "2. Setting up an Objective",
    "title": "API",
    "category": "section",
    "text": "cost\nstage_cost\ncost_expansion!"
},

{
    "location": "constraints/#",
    "page": "3. Creating Constraints",
    "title": "3. Creating Constraints",
    "category": "page",
    "text": ""
},

{
    "location": "constraints/#constraint_section-1",
    "page": "3. Creating Constraints",
    "title": "3. Creating Constraints",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"constraints.md\"]"
},

{
    "location": "constraints/#TrajectoryOptimization.ConstraintType",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.ConstraintType",
    "category": "type",
    "text": "Sense of a constraint (inequality / equality / null)\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.Equality",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.Equality",
    "category": "type",
    "text": "Inequality constraints\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.Inequality",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.Inequality",
    "category": "type",
    "text": "Equality constraints\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.Constraint",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.Constraint",
    "category": "type",
    "text": "struct Constraint{S} <: TrajectoryOptimization.AbstractConstraint{S}\n\nGeneral nonlinear vector-valued constraint with p entries at a single timestep for n states and m controls.\n\nConstructors\n\nConstraint{S}(confun, ∇confun, n, m, p, label; inds, term, inputs)\nConstraint{S}(confun, ∇confun, n,    p, label; inds, term, inputs)  # Terminal constraint\nConstraint{S}(confun,          n, m, p, label; inds, term, inputs)\nConstraint{S}(confun,          n,    p, label; inds, term, inputs)  # Terminal constraint\n\nArguments\n\nconfun: Inplace constraint function, either confun!(v, x, u) or confun!(v, xN).\n∇confun: Inplace constraint jacobian function, either ∇confun!(V, x, u) or ∇confun!(V, xN)\nn,m,p: Number of state, controls, and constraints\nlabel: Symbol that unique identifies the constraint\ninds: Specifies the indices of x and u that are passed into confun. Default highly recommended.\nterm: Where the constraint is applicable. One of (:all, :stage, :terminal). Detected automatically based on defined methods of confun but can be manually given.\ninputs: Which inputs are actually used. One of (:xu, :x, :u). Should be specified.\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.BoundConstraint",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.BoundConstraint",
    "category": "type",
    "text": "struct BoundConstraint{T} <: TrajectoryOptimization.AbstractConstraint{Inequality}\n\nLinear bound constraint on states and controls\n\nConstructors\n\nBoundConstraint(n, m; x_min, x_max, u_min, u_max)\n\nAny of the bounds can be ±∞. The bound can also be specifed as a single scalar, which applies the bound to all state/controls.\n\n\n\n\n\n"
},

{
    "location": "constraints/#Constraint-Type-1",
    "page": "3. Creating Constraints",
    "title": "Constraint Type",
    "category": "section",
    "text": "ConstraintType\nEquality\nInequalityAll constraints inherit from AbstractConstraint and are parameterized by ConstraintType, which specifies the type of constraint, Inequality or Equality. This allows the software to easily dispatch over the type of constraint. Each constraint type represents a vector-valued constraint. The intention is that each constraint type represent one line in constraints of problem definition (where they may be vector or scalar-valued). Each constraint has the following interface:evaluate!(v, con, x, u): Stage constraint  \nevaluate!(v, con, x): Terminal constraint  \njacobian!(V, con, x, u): Jacobian wrt states and controls at a stage time step  \njacobian!(V, con, x): Jacobian wrt terminal state at terminal time step  \nis_terminal(con): Boolean true if the constraint is defined at the terminal time step  \nis_stage(con): Boolean true if the constraint is defined at the stage time steps  \nlength(con): Number of constraints (length of the output vector)  There are currently two types of constraints implementedConstraint\nBoundConstraint"
},

{
    "location": "constraints/#General-Constraints-1",
    "page": "3. Creating Constraints",
    "title": "General Constraints",
    "category": "section",
    "text": ""
},

{
    "location": "constraints/#Fields-1",
    "page": "3. Creating Constraints",
    "title": "Fields",
    "category": "section",
    "text": "Each Constraint contains the following fields:c: the in-place constraint function. Methods dispatch over constraint functions of the form c(v,x,u) and c(v,x).\n∇c: the in-place constraint jacobian function defined as ∇c(Z,x,u) where Z is the p × (n+m) concatenated Jacobian. p: number of elements in the constraint vector\nlabel: a Symbol for identifying the constraint\ntype: a Symbol identifying where the constraint applies. One of [:stage, :terminal, :all]"
},

{
    "location": "constraints/#Basic-Constructor-1",
    "page": "3. Creating Constraints",
    "title": "Basic Constructor",
    "category": "section",
    "text": "Let\'s say we have a problem with 3 states and 2 controls, the following constraints x_1^2 + x_2^2 - 1 leq 0, x_2 + u_1  = 0, and x_3 + u_2 = 0. These two constraints could be created as follows:# Problem size\nn,m = 3,2\n\n# Inequality Constraint\nc(v,x,u) = v[1] = x[1]^2 + x[2]^2 - 1\np1 = 1\ncon = Constraint{Inequality}(c, n, m, p1, :mycon1)\n\n# Equality Constraint\nc_eq(v,x,u) = v[1:2] = x[2:3] + u\np2 = 2\ncon_eq = Constraint{Equality}(c_eq, n, m, p2, :mycon2)"
},

{
    "location": "constraints/#Analytical-Jacobians-1",
    "page": "3. Creating Constraints",
    "title": "Analytical Jacobians",
    "category": "section",
    "text": "Previously, we let the constructor build the Jacobians using ForwardDiff. We can alternatively specify them explicitly:# Jacobian for inequality constraint\n∇c(V,x,u) = begin V[1,1] = 2x[1]; V[1,2] = 2x[2]; end\ncon = Constraint{Inequality}(c, ∇c, n, m, p1, :mycon1)\n\n# Jacobian for equality constraint\n∇c_eq(V,x,u) = begin V[1,2] = 1; V[1,4] = 1;\n                     V[2,3] = 1; V[2,5] = 1; end\ncon_eq = Constraint{Equality}(c_eq, ∇c_eq, n, m, p2, :mycon2)"
},

{
    "location": "constraints/#Terminal-Constraints-1",
    "page": "3. Creating Constraints",
    "title": "Terminal Constraints",
    "category": "section",
    "text": "We can also build a terminal constraint that only depends on the terminal state, x_N. Let\'s say we have a terminal constraint x_1 + x_2 + x_3 = 1. We can create it using similar constructors:# Build terminal constraint\nc_term(v,x) = sum(x) - 1\np_term = 1\ncon_term = Constraint{Equality}(c_term, n, p_term, :con_term)\n\n# We can also optionally give it an analytical Jacobian\n∇c_term(V,x) = ones(1,n)\ncon_term = Constraint{Equality}(c_term, ∇c_term, n, p_term, :con_term)"
},

{
    "location": "constraints/#Both-Stage-and-Terminal-Constraints-1",
    "page": "3. Creating Constraints",
    "title": "Both Stage and Terminal Constraints",
    "category": "section",
    "text": "Every constraint can be applied to both stage and terminal time steps. The constructor automatically determines which is applied based on the methods defined for the provided function. You can check this by inspecting the type field, which will be one of [:stage, :terminal, :all].Notice our first constraint is only dependent on the state. If we want to enforce it at the terminal time step we can simply use multiple dispatch:con.label == :stage  # true\nc(v,x) = v[1] = x[1]^2 + x[2]^2 - 1\ncon = Constraint{Inequality}(c, n, m, p1, :mycon1)\ncon.label == :all  # trueThe type can easily be checked with the is_terminal and is_stage commands."
},

{
    "location": "constraints/#Methods-1",
    "page": "3. Creating Constraints",
    "title": "Methods",
    "category": "section",
    "text": "Given that constraints can apply at any time step, we assume that there is the same number of constraints for both stage and terminal time steps. The length method can also accept either :stage or :terminal as a second argument to specify which length you want (since one may be zero). e.g.length(con, :stage) == length(con, :terminal)  == 1  # true\nlength(con_eq, :stage) == 2     # true\nlength(con_eq, :terminal) == 0  # true"
},

{
    "location": "constraints/#TrajectoryOptimization.goal_constraint",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.goal_constraint",
    "category": "function",
    "text": "goal_constraint(xf)\n\nCreates a terminal equality constraint specifying the goal. All states must be specified.\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.planar_obstacle_constraint",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.planar_obstacle_constraint",
    "category": "function",
    "text": "planar_obstacle_constraint(n, m, x_obs, r_obs)\nplanar_obstacle_constraint(n, m, x_obs, r_obs, label)\n\n\nA constraint where x,y positions of the state must remain a distance r from a circle centered at x_obs Assumes x,y are the first two dimensions of the state vector\n\n\n\n\n\n"
},

{
    "location": "constraints/#Special-Constraints-1",
    "page": "3. Creating Constraints",
    "title": "Special Constraints",
    "category": "section",
    "text": "A few constructors for common constraints have been provided:goal_constraint\nplanar_obstacle_constraint"
},

{
    "location": "constraints/#Bound-Constraints-1",
    "page": "3. Creating Constraints",
    "title": "Bound Constraints",
    "category": "section",
    "text": "Bound constraints define simple bounds on the states and controls, allowing the solver to efficiently dispatch on methods to handle these simple constraints (especially when using direct methods). The constructor is very simpleBoundConstraint(n, m; x_min, x_max, u_min, x_max)The bounds can be given by vectors of the appropriate length, using ±Inf for unbounded values, or a scalar can be passed in when all the states have the same bound. If left blank, the value is assumed to be unbounded.Working from the previous examples, let\'s say we have -1 leq x_1 leq 1 x_3 leq 10 -15 leq u leq 12:# Create a bound constraint\nbnd = BoundConstraint(n, m, x_min=[-1,-Inf,-Inf], x_max=[1,Inf,10],\n                            u_min=-15, u_max=12)Note that bound constraints are automatically valid at both stage and terminal time steps, i.e. evaluate!(v, bnd, x, u) and evaluate!(v, bnd, x) are both defined."
},

{
    "location": "constraints/#RigidBodyDynamics.num_constraints-Tuple{Array{#s34,1} where #s34<:TrajectoryOptimization.GeneralConstraint,Any}",
    "page": "3. Creating Constraints",
    "title": "RigidBodyDynamics.num_constraints",
    "category": "method",
    "text": "num_constraints(C)\nnum_constraints(C, type)\n\n\nCount the total number of constraints in constraint set C. Stage or terminal is specified by type\n\n\n\n\n\n"
},

{
    "location": "constraints/#Base.pop!-Tuple{Array{#s34,1} where #s34<:TrajectoryOptimization.GeneralConstraint,Symbol}",
    "page": "3. Creating Constraints",
    "title": "Base.pop!",
    "category": "method",
    "text": "pop!(C, label)\n\n\nRemove a bound from a ConstraintSet given its label\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.evaluate!-Tuple{PartedArrays.PartedArray{T,1,P,P1} where P1 where P where T,Array{#s34,1} where #s34<:TrajectoryOptimization.GeneralConstraint,Any,Any}",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.evaluate!",
    "category": "method",
    "text": "evaluate!(c, C, x, u)\n\n\nEvaluate the constraint function for all the constraint functions in set C\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.evaluate!-Tuple{PartedArrays.PartedArray{T,1,P,P1} where P1 where P where T,Array{#s34,1} where #s34<:TrajectoryOptimization.GeneralConstraint,Any}",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.evaluate!",
    "category": "method",
    "text": "evaluate!(c, C, x)\n\n\nEvaluate the constraint function for all the terminal constraint functions in set C\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.jacobian!-Tuple{PartedArrays.PartedArray{T,2,P,P1} where P1 where P where T,Array{#s34,1} where #s34<:TrajectoryOptimization.GeneralConstraint,Any,Any}",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.jacobian!",
    "category": "method",
    "text": "jacobian!(c, C, x, u)\n\n\nCompute the constraint Jacobian of ConstraintSet C\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.jacobian!-Tuple{PartedArrays.PartedArray{T,2,P,P1} where P1 where P where T,Array{#s34,1} where #s34<:TrajectoryOptimization.GeneralConstraint,Any}",
    "page": "3. Creating Constraints",
    "title": "TrajectoryOptimization.jacobian!",
    "category": "method",
    "text": "jacobian!(c, C, x)\n\n\nCompute the constraint Jacobian of ConstraintSet C at the terminal time step\n\n\n\n\n\n"
},

{
    "location": "constraints/#Constraint-Sets-1",
    "page": "3. Creating Constraints",
    "title": "Constraint Sets",
    "category": "section",
    "text": "A ConstraintSet is simply a vector of constraints, and represents a set of constraints at a particular time step. There are some convenient methods for creating and working with constraint sets.Let\'s say we combine the previous constraints into a single constraint set. We can do this easily using the + method:# Create a constraint set for stage time steps\nconstraints = con + con_eq + bnd\n\n# Create a constraint set for terminal constraints\nconstraints_term = con + con_term + bndThere are several functions provided to work with ConstraintSetsTrajectoryOptimization.num_constraints(C::ConstraintSet, type)\nBase.pop!(C::ConstraintSet, label::Symbol)\nevaluate!(c::PartedVector, C::ConstraintSet, x, u)\nevaluate!(c::PartedVector, C::ConstraintSet, x)\njacobian!(c::PartedMatrix, C::ConstraintSet, x, u)\njacobian!(c::PartedMatrix, C::ConstraintSet, x)The PartedVector and PartedMatrix needed for the evaluate! and jacobian! methods can be generated usingPartedVector(C::ConstraintSet, type=:stage)\nPartedMatrix(C::ConstraintSet, type=:stage)"
},

{
    "location": "constraints/#Problem-Constraints-1",
    "page": "3. Creating Constraints",
    "title": "Problem Constraints",
    "category": "section",
    "text": "A Problem is made up of individual ConstraintSets at each of the N time steps, allowing for different constraints along the trajectory. The collection of ConstraintSets is captured in the Constraints type. There are several methods for constructing Constraints:# Create an empty set\nConstraints(N)\n\n# Copy a single ConstraintSet over every time step\nConstraints(constraints, N)\n\n# Use a different set at the terminal time step\nConstraints(constraints, constraints_term, N)\n\n# Create from a Vector of Constraint Sets\nConstraints([constraints, constraints, constraints, constraints_term])You can easily add or remove constraints from time steps using + and pop! on the appropriate time step:pcon = Constraints(N)\npcon[1] += con\npcon[2] += con + con_eq\npop!(pcon[2], :mycon2)\npcon[N] += con_term"
},

{
    "location": "constraints/#RigidBodyDynamics.num_constraints-Tuple{Constraints}",
    "page": "3. Creating Constraints",
    "title": "RigidBodyDynamics.num_constraints",
    "category": "method",
    "text": "num_constraints(pcon)\n\n\nCount the number of constraints at each time step.\n\n\n\n\n\n"
},

{
    "location": "constraints/#Methods-2",
    "page": "3. Creating Constraints",
    "title": "Methods",
    "category": "section",
    "text": "TrajectoryOptimization.num_constraints(::Constraints)"
},

{
    "location": "problem/#",
    "page": "4. Setting up a Problem",
    "title": "4. Setting up a Problem",
    "category": "page",
    "text": ""
},

{
    "location": "problem/#problem_section-1",
    "page": "4. Setting up a Problem",
    "title": "4. Setting up a Problem",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"problem.md\"]"
},

{
    "location": "problem/#TrajectoryOptimization.Problem",
    "page": "4. Setting up a Problem",
    "title": "TrajectoryOptimization.Problem",
    "category": "type",
    "text": "struct Problem{T<:AbstractFloat, D<:TrajectoryOptimization.DynamicsType}\n\nTrajectory Optimization Problem. Contains the full definition of a trajectory optimization problem, including:\n\ndynamics model (Model). Can be either continuous or discrete.\nobjective (Objective)\nconstraints (Constraints)\ninitial and final states\nPrimal variables (state and control trajectories)\nDiscretization information: knot points (N), time step (dt), and total time (tf)\n\nConstructors:\n\nProblem(model, obj X0, U0; integration, constraints, x0, xf, dt, tf, N)\nProblem(model, obj, U0; integration, constraints, x0, xf, dt, tf, N)\nProblem(model, obj; integration, constraints, x0, xf, dt, tf, N)\n\nArguments\n\nmodel: Dynamics model. Can be either Discrete or Continuous\nobj: Objective\nX0: Initial state trajectory. If omitted it will be initialized with NaNs, to be later overwritten by the solver.\nU0: Initial control trajectory. If omitted it will be initialized with zeros.\nx0: Initial state. Defaults to zeros.\nxf: Final state. Defaults to zeros.\ndt: Time step\ntf: Final time. Set to zero to specify a time penalized problem.\nN: Number of knot points. Defaults to 51, unless specified by dt and tf.\nintegration: One of the defined integration types to discretize the continuous dynamics model. Defaults to :none, which will pass in the continuous dynamics (eg. for DIRCOL)\n\nBoth X0 and U0 can be either a Matrix or a Vector{Vector}, but must be the same. At least 2 of dt, tf, and N need to be specified (or just 1 of dt and tf).\n\n\n\n\n\n"
},

{
    "location": "problem/#Creating-a-Problem-1",
    "page": "4. Setting up a Problem",
    "title": "Creating a Problem",
    "category": "section",
    "text": "Problem"
},

{
    "location": "problem/#TrajectoryOptimization.update_problem",
    "page": "4. Setting up a Problem",
    "title": "TrajectoryOptimization.update_problem",
    "category": "function",
    "text": "update_problem(prob; kwargs...)\n\nCreate a new problem from another, specifing all fields as keyword arguments The newProb argument can be set to true if a the primal variables are to be copied, otherwise they will be passed to the modified problem.\n\n\n\n\n\n"
},

{
    "location": "problem/#TrajectoryOptimization.initial_controls!",
    "page": "4. Setting up a Problem",
    "title": "TrajectoryOptimization.initial_controls!",
    "category": "function",
    "text": "initial_controls!(prob, U0)\n\n\nSet the initial control trajectory for a problem. U0 can be either a Matrix or a Vector{Vector}\n\n\n\n\n\n"
},

{
    "location": "problem/#TrajectoryOptimization.initial_states!",
    "page": "4. Setting up a Problem",
    "title": "TrajectoryOptimization.initial_states!",
    "category": "function",
    "text": "Set the initial state trajectory for a problem. X0 can be either a Matrix or a Vector{Vector}\n\n\n\n\n\n"
},

{
    "location": "problem/#TrajectoryOptimization.set_x0!",
    "page": "4. Setting up a Problem",
    "title": "TrajectoryOptimization.set_x0!",
    "category": "function",
    "text": "set_x0!(prob, x0)\n\n\nSet the initial state\n\n\n\n\n\n"
},

{
    "location": "problem/#Base.size-Tuple{Problem}",
    "page": "4. Setting up a Problem",
    "title": "Base.size",
    "category": "method",
    "text": "n,m,N = size(p::Problem)\n\nReturn the number of states (n), number of controls (m), and the number of knot points (N)\n\n\n\n\n\n"
},

{
    "location": "problem/#Base.copy-Tuple{Problem}",
    "page": "4. Setting up a Problem",
    "title": "Base.copy",
    "category": "method",
    "text": "copy(p::Problem) -> Problem{_1,_2} where _2 where _1\n\n\nCopy a problem\n\n\n\n\n\n"
},

{
    "location": "problem/#TrajectoryOptimization.is_constrained",
    "page": "4. Setting up a Problem",
    "title": "TrajectoryOptimization.is_constrained",
    "category": "function",
    "text": "Checks if a problem has any constraints\n\n\n\n\n\n"
},

{
    "location": "problem/#TrajectoryOptimization.max_violation-Union{Tuple{Problem{T,D} where D<:DynamicsType}, Tuple{T}} where T",
    "page": "4. Setting up a Problem",
    "title": "TrajectoryOptimization.max_violation",
    "category": "method",
    "text": "Evaluate the maximum contraint violation\n\n\n\n\n\nEvaluate maximum constraint violation\n\n\n\n\n\n"
},

{
    "location": "problem/#TrajectoryOptimization.final_time",
    "page": "4. Setting up a Problem",
    "title": "TrajectoryOptimization.final_time",
    "category": "function",
    "text": "final_time(prob)\n\n\nGet the total time for tje trajectory (applicable for time-penalized problems)\n\n\n\n\n\n"
},

{
    "location": "problem/#Methods-1",
    "page": "4. Setting up a Problem",
    "title": "Methods",
    "category": "section",
    "text": "update_problem\ninitial_controls!\ninitial_states!\nset_x0!\nBase.size(::Problem)\nBase.copy(::Problem)\nis_constrained\nmax_violation(::Problem{T}) where T\nfinal_time"
},

{
    "location": "solvers/#",
    "page": "Solvers",
    "title": "Solvers",
    "category": "page",
    "text": ""
},

{
    "location": "solvers/#Solvers-1",
    "page": "Solvers",
    "title": "Solvers",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"solvers.md\"]"
},

{
    "location": "solvers/#TrajectoryOptimization.iLQRSolver",
    "page": "Solvers",
    "title": "TrajectoryOptimization.iLQRSolver",
    "category": "type",
    "text": "struct iLQRSolver{T} <: TrajectoryOptimization.AbstractSolver{T}\n\niLQR is an unconstrained indirect method for trajectory optimization that parameterizes only the controls and enforces strict dynamics feasibility at every iteration by simulating forward the dynamics with an LQR feedback controller. The main algorithm consists of two parts:\n\na backward pass that uses Differential Dynamic Programming to compute recursively a quadratic approximation of the cost-to-go, along with linear feedback and feed-forward gain matrices, K and d, respectively, for an LQR tracking controller, and\na forward pass that uses the gains K and d to simulate forward the full nonlinear dynamics with feedback.\n\n\n\n\n\n"
},

{
    "location": "solvers/#TrajectoryOptimization.iLQRSolverOptions",
    "page": "Solvers",
    "title": "TrajectoryOptimization.iLQRSolverOptions",
    "category": "type",
    "text": "mutable struct iLQRSolverOptions{T} <: TrajectoryOptimization.AbstractSolverOptions{T}\n\nSolver options for the iterative LQR (iLQR) solver.\n\nverbose\nPrint summary at each iteration. Default: false\nlive_plotting\nLive plotting. Default: :off\ncost_tolerance\ndJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve. Default: 0.0001\ngradient_type\ngradient type: :todorov, :feedforward. Default: :todorov\ngradient_norm_tolerance\ngradient_norm < ϵ, gradient norm convergence criteria. Default: 1.0e-5\niterations\niLQR iterations. Default: 300\ndJ_counter_limit\nrestricts the total number of times a forward pass fails, resulting in regularization, before exiting. Default: 10\nsquare_root\nuse square root method backward pass for numerical conditioning. Default: false\nline_search_lower_bound\nforward pass approximate line search lower bound, 0 < linesearchlowerbound < linesearchupperbound. Default: 1.0e-8\nline_search_upper_bound\nforward pass approximate line search upper bound, 0 < linesearchlowerbound < linesearchupperbound < ∞. Default: 10.0\niterations_linesearch\nmaximum number of backtracking steps during forward pass line search. Default: 20\nbp_reg_initial\ninitial regularization. Default: 0.0\nbp_reg_increase_factor\nregularization scaling factor. Default: 1.6\nbp_reg_max\nmaximum regularization value. Default: 1.0e8\nbp_reg_min\nminimum regularization value. Default: 1.0e-8\nbp_reg_type\ntype of regularization- control: () + ρI, state: (S + ρI); see Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization. Default: :control\nbp_reg_fp\nadditive regularization when forward pass reaches max iterations. Default: 10.0\nbp_sqrt_inv_type\ntype of matrix inversion for bp sqrt step. Default: :pseudo\nbp_reg_sqrt_initial\ninitial regularization for square root method. Default: 1.0e-6\nbp_reg_sqrt_increase_factor\nregularization scaling factor for square root method. Default: 10.0\nmax_cost_value\nmaximum cost value, if exceded solve will error. Default: 1.0e8\nmax_state_value\nmaximum state value, evaluated during rollout, if exceded solve will error. Default: 1.0e8\nmax_control_value\nmaximum control value, evaluated during rollout, if exceded solve will error. Default: 1.0e8\n\n\n\n\n\n"
},

{
    "location": "solvers/#Iterative-LQR-(iLQR)-1",
    "page": "Solvers",
    "title": "Iterative LQR (iLQR)",
    "category": "section",
    "text": "iLQRSolver\niLQRSolverOptions"
},

{
    "location": "solvers/#TrajectoryOptimization.AugmentedLagrangianSolver",
    "page": "Solvers",
    "title": "TrajectoryOptimization.AugmentedLagrangianSolver",
    "category": "type",
    "text": "struct AugmentedLagrangianSolver <: TrajectoryOptimization.AbstractSolver{T}\n\nAugmented Lagrangian (AL) is a standard tool for constrained optimization. For a trajectory optimization problem of the form:\n\nbeginequation*\nbeginaligned\n  min_x_0Nu_0N-1 quad  ell_f(x_N) + sum_k=0^N-1 ell_k(x_k u_k dt) \n  textrmst            quad  x_k+1 = f(x_k u_k) \n                                  g_k(x_ku_k) leq 0 \n                                  h_k(x_ku_k) = 0\nendaligned\nendequation*\n\nAL methods form the following augmented Lagrangian function:\n\nbeginalign*\n    ell_f(x_N) + λ_N^T c_N(x_N) + c_N(x_N)^T I_mu_N c_N(x_N) \n            + sum_k=0^N-1 ell_k(x_ku_kdt) + λ_k^T c_k(x_ku_k) + c_k(x_ku_k)^T I_mu_k c_k(x_ku_k)\nendalign*\n\nThis function is then minimized with respect to the primal variables using any unconstrained minimization solver (e.g. iLQR).     After a local minima is found, the AL method updates the Lagrange multipliers λ and the penalty terms μ and repeats the unconstrained minimization.     AL methods have superlinear convergence as long as the penalty term μ is updated each iteration.\n\n\n\n\n\n"
},

{
    "location": "solvers/#TrajectoryOptimization.AugmentedLagrangianSolverOptions",
    "page": "Solvers",
    "title": "TrajectoryOptimization.AugmentedLagrangianSolverOptions",
    "category": "type",
    "text": "mutable struct AugmentedLagrangianSolverOptions{T} <: TrajectoryOptimization.AbstractSolverOptions{T}\n\nSolver options for the augmented Lagrangian solver.\n\nverbose\nPrint summary at each iteration. Default: false\nopts_uncon\nunconstrained solver options. Default: iLQRSolverOptions{T}()\ncost_tolerance\ndJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve. Default: 0.0001\ncost_tolerance_intermediate\ndJ < ϵ_int, intermediate cost convergence criteria to enter outerloop of constrained solve. Default: 0.001\ngradient_norm_tolerance\ngradient_norm < ϵ, gradient norm convergence criteria. Default: 1.0e-5\ngradient_norm_tolerance_intermediate\ngradientnormint < ϵ, gradient norm intermediate convergence criteria. Default: 1.0e-5\nconstraint_tolerance\nmax(constraint) < ϵ, constraint convergence criteria. Default: 0.001\nconstraint_tolerance_intermediate\nmax(constraint) < ϵ_int, intermediate constraint convergence criteria. Default: 0.001\niterations\nmaximum outerloop updates. Default: 30\ndual_min\nminimum Lagrange multiplier. Default: -1.0e8\ndual_max\nmaximum Lagrange multiplier. Default: 1.0e8\npenalty_max\nmaximum penalty term. Default: 1.0e8\npenalty_initial\ninitial penalty term. Default: 1.0\npenalty_scaling\npenalty update multiplier; penalty_scaling > 0. Default: 10.0\npenalty_scaling_no\npenalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ). Default: 1.0\nconstraint_decrease_ratio\nratio of current constraint to previous constraint violation; 0 < constraintdecreaseratio < 1. Default: 0.25\nouter_loop_update_type\ntype of outer loop update (default, feedback). Default: :default\nactive_constraint_tolerance\nnumerical tolerance for constraint violation. Default: 0.0\nkickout_max_penalty\nterminal solve when maximum penalty is reached. Default: false\n\n\n\n\n\n"
},

{
    "location": "solvers/#Augmented-Lagrangian-1",
    "page": "Solvers",
    "title": "Augmented Lagrangian",
    "category": "section",
    "text": "AugmentedLagrangianSolver\nAugmentedLagrangianSolverOptions"
},

{
    "location": "solvers/#TrajectoryOptimization.ALTROSolver",
    "page": "Solvers",
    "title": "TrajectoryOptimization.ALTROSolver",
    "category": "type",
    "text": "struct ALTROSolver{T} <: TrajectoryOptimization.AbstractSolver{T}\n\nAugmented Lagrangian Trajectory Optimizer (ALTRO) is a solver developed by the Robotic Exploration Lab at Stanford University.     The solver is special-cased to solve Markov Decision Processes by leveraging the internal problem structure.\n\nALTRO consists of two \"phases\":\n\nAL-iLQR: iLQR is used with an Augmented Lagrangian framework to solve the problem quickly to rough constraint satisfaction\nProjected Newton: A collocation-flavored active-set solver projects the solution from AL-iLQR onto the feasible subspace to achieve machine-precision constraint satisfaction.\n\n\n\n\n\n"
},

{
    "location": "solvers/#TrajectoryOptimization.ALTROSolverOptions",
    "page": "Solvers",
    "title": "TrajectoryOptimization.ALTROSolverOptions",
    "category": "type",
    "text": "mutable struct ALTROSolverOptions{T} <: TrajectoryOptimization.AbstractSolverOptions{T}\n\nSolver options for the ALTRO solver.\n\nverbose\nDefault: false\nopts_al\nAugmented Lagrangian solver options. Default: AugmentedLagrangianSolverOptions{T}()\nconstraint_tolerance_infeasible\ninfeasible control constraint tolerance. Default: 1.0e-5\nR_inf\nregularization term for infeasible controls. Default: 1.0\ndynamically_feasible_projection\nproject infeasible results to feasible space using TVLQR. Default: true\nresolve_feasible_problem\nresolve feasible problem after infeasible solve. Default: true\npenalty_initial_infeasible\ninitial penalty term for infeasible controls. Default: 1.0\npenalty_scaling_infeasible\npenalty update rate for infeasible controls. Default: 10.0\nR_minimum_time\nregularization term for dt. Default: 1.0\ndt_max\nmaximum allowable dt. Default: 1.0\ndt_min\nminimum allowable dt. Default: 0.001\npenalty_initial_minimum_time_inequality\ninitial penalty term for minimum time bounds constraints. Default: 1.0\npenalty_initial_minimum_time_equality\ninitial penalty term for minimum time equality constraints. Default: 1.0\npenalty_scaling_minimum_time_inequality\npenalty update rate for minimum time bounds constraints. Default: 1.0\npenalty_scaling_minimum_time_equality\npenalty update rate for minimum time equality constraints. Default: 1.0\nprojected_newton\nfinish with a projecte newton solve. Default: false\nopts_pn\noptions for projected newton solver. Default: ProjectedNewtonSolverOptions{T}()\nprojected_newton_tolerance\nconstraint satisfaction tolerance that triggers the projected newton solver.     If set to a non-positive number it will kick out when the maximum penalty is reached. Default: 0.001\n\n\n\n\n\n"
},

{
    "location": "solvers/#ALTRO-1",
    "page": "Solvers",
    "title": "ALTRO",
    "category": "section",
    "text": "ALTROSolver\nALTROSolverOptions"
},

{
    "location": "solvers/#TrajectoryOptimization.DIRCOLSolver",
    "page": "Solvers",
    "title": "TrajectoryOptimization.DIRCOLSolver",
    "category": "type",
    "text": "struct DIRCOLSolver{T, Q} <: TrajectoryOptimization.DirectSolver{T}\n\nDirect Collocation Solver. Uses a commerical NLP solver to solve the Trajectory Optimization problem. Uses the MathOptInterface to interface with the NLP.\n\n\n\n\n\n"
},

{
    "location": "solvers/#TrajectoryOptimization.DIRCOLSolverOptions",
    "page": "Solvers",
    "title": "TrajectoryOptimization.DIRCOLSolverOptions",
    "category": "type",
    "text": "mutable struct DIRCOLSolverOptions{T} <: TrajectoryOptimization.DirectSolverOptions{T}\n\nSolver options for the Direct Collocation solver. Most options are passed to the NLP through the opts dictionary\n\n\n\n\n\n"
},

{
    "location": "solvers/#Direct-Collocation-(DIRCOL)-1",
    "page": "Solvers",
    "title": "Direct Collocation (DIRCOL)",
    "category": "section",
    "text": "DIRCOLSolver\nDIRCOLSolverOptions"
},

{
    "location": "solvers/#TrajectoryOptimization.ProjectedNewtonSolver",
    "page": "Solvers",
    "title": "TrajectoryOptimization.ProjectedNewtonSolver",
    "category": "type",
    "text": "struct ProjectedNewtonSolver{T} <: TrajectoryOptimization.DirectSolver{T}\n\nProjected Newton Solver Direct method developed by the REx Lab at Stanford University\n\nAchieves machine-level constraint satisfaction by projecting onto the feasible subspace.     It can also take a full Newton step by solving the KKT system.\n\nThis solver is to be used exlusively for solutions that are close to the optimal solution.     It is intended to be used as a \"solution polishing\" method for augmented Lagrangian methods.\n\n\n\n\n\n"
},

{
    "location": "solvers/#TrajectoryOptimization.ProjectedNewtonSolverOptions",
    "page": "Solvers",
    "title": "TrajectoryOptimization.ProjectedNewtonSolverOptions",
    "category": "type",
    "text": "mutable struct ProjectedNewtonSolverOptions{T} <: TrajectoryOptimization.DirectSolverOptions{T}\n\nSolver options for the Projected Newton solver\n\n\n\n\n\n"
},

{
    "location": "solvers/#Projected-Newton-1",
    "page": "Solvers",
    "title": "Projected Newton",
    "category": "section",
    "text": "ProjectedNewtonSolver\nProjectedNewtonSolverOptions"
},

]}
