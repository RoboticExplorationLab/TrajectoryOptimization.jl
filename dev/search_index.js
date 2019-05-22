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
    "text": "CurrentModule = TrajectoryOptimizationDocumentation for TrajectoryOptimization.jl"
},

{
    "location": "#Overview-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Overview",
    "category": "section",
    "text": "This package is a testbed for state-of-the-art trajectory optimization algorithms. Trajectory optimization problems are of the form, (put LaTeX here)This package currently implements both indirect and direct methods for trajectory optimization:Iterative LQR (iLQR): indirect method based on Differential Dynamic Programming\nDirect Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solverKey features include the use of ForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints; the use of RigidBodyDynamics to work directly from URDF files; and the ability to specify general constraints."
},

{
    "location": "#Getting-Started-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Getting Started",
    "category": "section",
    "text": "To set up and solve a trajectory optimization problem with TrajectoryOptimization.jl, the user will go through the following steps:Create a Model\nCreate an Objective\nInstantiate a Problem\n(Optionally) Add constraints\nSelect a solver\nSolve the problem\nAnalyze the solution"
},

{
    "location": "#Creating-a-Model-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Creating a Model",
    "category": "section",
    "text": "There are two ways of creating a model:from an in-place analytic function of the form f(y,x,u) that operates on y\nfrom a URDF file"
},

{
    "location": "#Analytic-Models-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Analytic Models",
    "category": "section",
    "text": "To create an analytic model, first create an in-place function for the continuous or discrete dynamics. The function must be of the form f(y,x,u) where y ∈ Rⁿ is the state derivative vector for continuous dynamics or the next state for discrete dynamics. x ∈ Rⁿ is the state vector, and u ∈ Rᵐ is the control input vector. The function should not return any values, but should write y \"inplace,\" e.g. y[1] = x[2]*u[2] NOT y = f(x,u). This makes a significant difference in performance.The Model type is then created using the following signature: model = Model{D}(f,n,m) where n is the dimension of the state input and m is the dimension of the control input, and D is a DynamicsType, either Continuous of Discrete.Model{D}(f::Function, n::Int, m::Int)"
},

{
    "location": "#URDF-Model-1",
    "page": "TrajectoryOptimization.jl",
    "title": "URDF Model",
    "category": "section",
    "text": "This package relies on RigidBodyDynamics.jl to parse URDFs and generate dynamics functions for them. There are several useful constructors:Model(mech::Mechanism)\nModel(mech::Mechanism, torques::Array)\nModel(urdf::String)\nModel(urdf::String, torques::Array)"
},

{
    "location": "#Creating-an-Objective-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Creating an Objective",
    "category": "section",
    "text": "The Objective defines a metric for what you want the dynamics to do. The Objective type contains a CostFunctions for each stage of the trajectory."
},

{
    "location": "#TrajectoryOptimization.QuadraticCost",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.QuadraticCost",
    "category": "type",
    "text": "mutable struct QuadraticCost{T} <: TrajectoryOptimization.CostFunction\n\nCost function of the form     1/2xₙᵀ Qf xₙ + qfᵀxₙ +  ∫ ( 1/2xᵀQx + 1/2uᵀRu + xᵀHu + q⁠ᵀx  rᵀu ) dt from 0 to tf R must be positive definite, Q and Qf must be positive semidefinite\n\n\n\n\n\n"
},

{
    "location": "#TrajectoryOptimization.LQRCost",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.LQRCost",
    "category": "function",
    "text": "LQRCost(Q, R, Qf, xf)\n\n\nCost function of the form     1/2(xₙ-xf)ᵀ Qf (xₙ - xf) + 1/2 ∫ ( (x-x_f)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf R must be positive definite, Q and Qf must be positive semidefinite\n\n\n\n\n\n"
},

{
    "location": "#TrajectoryOptimization.GenericCost",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.GenericCost",
    "category": "type",
    "text": "struct GenericCost <: TrajectoryOptimization.CostFunction\n\nCost function of the form     ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf\n\n\n\n\n\n"
},

{
    "location": "#Creating-a-Cost-Function-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Creating a Cost Function",
    "category": "section",
    "text": "A CostFunction is required for each stage of the trajectory to define an Objective. While the majority of trajectory optimization problems have quadratic objectives, TrajectoryOptimization.jl allows the user to specify any generic cost function of the form ell_N(x_N) + sum_k=0^N ell_k(x_ku_k). Currently GenericObjective is only supported by iLQR, and not by DIRCOL. Since iLQR relies on 2nd Order Taylor Series Expansions of the cost, the user may specify analytical functions for this expansion in order to increase performance; if the user does not specify an analytical expansion it will be generated using ForwardDiff.QuadraticCost\nLQRCost\nGenericCost"
},

{
    "location": "#TrajectoryOptimization.Problem",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.Problem",
    "category": "type",
    "text": "struct Problem{T<:AbstractFloat, D<:TrajectoryOptimization.DynamicsType}\n\nTrajectory Optimization Problem\n\n\n\n\n\n"
},

{
    "location": "#Solving-the-Problem-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Solving the Problem",
    "category": "section",
    "text": "With a defined model and objective, the next step is to create a Problem type. The Problem contains both the Model and Objective and contains all information needed for the solve.ProblemOnce the Problem is instantiated, the user must create an initial guess for the control trajectory, and optionally a state trajectory. For simple problems a initialization of random values, ones, or zeros works well. For more complicated systems it is usually recommended to feed trim conditions, i.e. controls that maintain the initial state values. Note that for trajectory optimization the control trajectory should be length N-1 since there are no controls at the final time step. However, DIRCOL uses controls at the final time step, and iLQR will simply discard any controls at the time step. Therefore, an initial control trajectory of size (m,N) is valid (but be aware that iLQR will return the correctly-sized control trajectory). Once the initial state and control trajectories are specified, they are passed with the solver to one of the solve methods."
},

{
    "location": "#Solve-Methods-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Solve Methods",
    "category": "section",
    "text": "With a Problem instantiated, the user can then select a solver: iLQR, AugmentedLagrangian, ALTRO, DIRCOL."
},

{
    "location": "#Unconstrained-Methods-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Unconstrained Methods",
    "category": "section",
    "text": "iLQR is an unconstrained solver. For unconstrained Problems simply call the solve method:solve(prob,iLQRSolverOptions())"
},

{
    "location": "#Constrained-Methods-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Constrained Methods",
    "category": "section",
    "text": ""
},

{
    "location": "#Augmented-Lagrangian-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Augmented Lagrangian",
    "category": "section",
    "text": "The default constrained solver uses iLQR with an augmented Lagrangian framework to handle general nonlinear constraints. AugmentedLagrangianSolverOptions can be changed to effect solver performance. Other than now having more parameters to tune for better performance (see another section for tips), the user solves a constrained problem using the exact same method for solving an unconstrained problem."
},

{
    "location": "#ALTRO-1",
    "page": "TrajectoryOptimization.jl",
    "title": "ALTRO",
    "category": "section",
    "text": ""
},

{
    "location": "#Constrained-Problem-with-Infeasible-Start-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Constrained Problem with Infeasible Start",
    "category": "section",
    "text": "One of the primary disadvantages of iLQR (and most indirect methods) is that the user must specify an initial input trajectory. Specifying a good initial guess can often be difficult in practice, whereas specifying a guess for the state trajectory is typically more straightforward. To overcome this limitation, the ALTRO solver adds slack controls to the discrete dynamics x_k+1 = f_d(x_ku_k) + diag(tidleu_1hdotstidleu_n) such that the system becomes artificially fully-actuated These slack controls are then constrained to be zero using the augmented Lagrangian method. This results in an algorithm similar to that of DIRCOL: initial solutions are dynamically infeasible but become dynamically infeasible at convergence. To solve the problem using \"infeasible start\", simply pass in an initial guess for the state and control:copyto!(prob.X,X0)\nsolve(prob,ALTROSolverOptions())"
},

{
    "location": "#Minimum-Time-Problem-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Minimum Time Problem",
    "category": "section",
    "text": "A minimum time problem can be solved using the ALTRO solver by setting tf=:min"
},

{
    "location": "#Direct-Collocation-(DIRCOL)-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Direct Collocation (DIRCOL)",
    "category": "section",
    "text": "Problems can be solved using DIRCOL by simply callingsolve(prob,DIRCOLSolverOptions())"
},

{
    "location": "models/#",
    "page": "1. Setting up a Dynamics Model",
    "title": "1. Setting up a Dynamics Model",
    "category": "page",
    "text": ""
},

{
    "location": "models/#.-Setting-up-a-Dynamics-Model-1",
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
    "text": "struct AnalyticalModel{D} <: Model{D}\n\nDynamics model\n\nHolds all information required to uniquely describe a dynamic system, including a general nonlinear dynamics function of the form ẋ = f(x,u), where x ∈ ℜⁿ are the states and u ∈ ℜᵐ are the controls.\n\nDynamics function, f, should be of the form     f(ẋ,x,u,p) for Continuous models, where ẋ is the state derivative     f(ẋ,x,u,p,dt) for Discrete models, where ẋ is the state at the next time step     and x is the state vector, u is the control input vector, and p is an optional NamedTuple of static parameters (mass, gravity, etc.)\n\nDynamics jacobians, ∇f, should be of the form     ∇f(Z,x,u,p) for Continuous models, and     ∇f(Z,x,u,,p,dt) for discrete models     where p is the same NamedTuple of parameters used in the dynamics\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.RBDModel",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.RBDModel",
    "category": "type",
    "text": "struct RBDModel{D} <: Model{D}\n\nRigidBodyDynamics model. Wrapper for a RigidBodyDynamics Mechanism\n\n\n\n\n\n"
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
    "text": "It\'s often useful to test out the dynamics or it\'s Jacobians. We must pre-allocate the arraysxdot = zeros(n)\nZ = zeros(n,m)Or create a partitioned vector and Jacobian for easy access to separate state and control jacobiansxdot = BlockVector(model)\nZ = BlockMatrix(model)Once the arrays are allocated, we can call using evaluate! and jacobian! (increments the evaluation count, recommended)x,u = rand(x), rand(u)  # or some known test inputs\nevaluate!(xdot,model,x,u)\njacobian!(Z,model,x,u)or call them directly (not recommended)model.f(xdot,x,u)\nmodel.∇f(Z,x,u,dt)If we created a partitioned Jacobian using BlockMatrix(model), we can access the different piecesfdx = Z.x   # ∂f/∂x\nfdu = Z.u   # ∂f/∂u\nfdt = Z.dt  # ∂f/∂dt"
},

{
    "location": "models/#TrajectoryOptimization.Model",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(f::Function, n::Int64, m::Int64) -> TrajectoryOptimization.AnalyticalModel{Continuous}\nModel(f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{Continuous}\n\n\nCreate a dynamics model, using ForwardDiff to generate the dynamics jacobian, with parameters Dynamics function passes in parameters:     f(ẋ,x,u,p)     where p in NamedTuple of parameters\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(f::Function, n::Int64, m::Int64, p::NamedTuple) -> TrajectoryOptimization.AnalyticalModel{Continuous}\nModel(f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{Continuous}\n\n\nCreate a dynamics model, using ForwardDiff to generate the dynamics jacobian, without parameters Dynamics function of the form:     f(ẋ,x,u)\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(f::Function, ∇f::Function, n::Int64, m::Int64, p::NamedTuple) -> TrajectoryOptimization.AnalyticalModel{Continuous}\nModel(f::Function, ∇f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{Continuous}\n\n\nCreate a dynamics model with an analytical Jacobian, with parameters Dynamics functions pass in parameters:     f(ẋ,x,u,p)     ∇f(Z,x,u,p)     where p in NamedTuple of parameters\n\n\n\n\n\n"
},

{
    "location": "models/#TrajectoryOptimization.Model",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.Model",
    "category": "type",
    "text": "Model(f::Function, ∇f::Function, n::Int64, m::Int64) -> TrajectoryOptimization.AnalyticalModel{Continuous}\nModel(f::Function, ∇f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{Continuous}\n\n\nCreate a dynamics model with an analytical Jacobian, without parameters Dynamics functions pass of the form:     f(ẋ,x,u)     ∇f(Z,x,u)\n\n\n\n\n\n"
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
    "location": "models/#API-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "API",
    "category": "section",
    "text": "The following constructors can be used to create a model from an analytic function, with or without parameters or analyical JacobiansModel(f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())\nModel(f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())\nModel(f::Function, ∇f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())\nModel(f::Function, ∇f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())The following constructors can be used to create a Model from a URDF fileModel(mech::Mechanism, torques::Array)\nModel(mech::Mechanism)\nModel(urdf::String)\nModel(urdf::String,torques::Array{Float64,1})evaluate!(ẋ::AbstractVector,model::Model,x,u)\nevaluate!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model,x,u)\njacobian!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model,x,u)\njacobian!(Z::AbstractMatrix,model::Model,x,u)\nevals(model::Model)\nreset(model::Model)"
},

{
    "location": "costfunctions/#",
    "page": "2. Setting up a Cost Function",
    "title": "2. Setting up a Cost Function",
    "category": "page",
    "text": ""
},

{
    "location": "costfunctions/#.-Setting-up-a-Cost-Function-1",
    "page": "2. Setting up a Cost Function",
    "title": "2. Setting up a Cost Function",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"costfunctions.md\"]"
},

{
    "location": "costfunctions/#Overview-1",
    "page": "2. Setting up a Cost Function",
    "title": "Overview",
    "category": "section",
    "text": "All trajectory optimization problems require a cost function at each stage of the trajectory. Cost functions must be scalar-valued. We assume general cost functions of the form,ell_f(x_N) + sum_k=1^N-1 ell_k(x_ku_k)It is very important to note that ell_k(x_ku_k) is ONLY a function of x_k and u_k, i.e. no coupling across time-steps is permitted. This is a requirement for Differential Dynamic Programming methods such as iLQR, but could be relaxed for methods that parameterize both states and controls, such as DIRCOL (although this is currently not supported). In general, any coupling between adjacent time-steps can be resolved by augmenting the state and defining the appropriate dynamics (this is the method we use to solve minimum time problems).In general, trajectory optimization will take a second order Taylor series approximation of the cost function, resulting in a quadratic cost function of the formx_N^T Q_f x_N + q_f^T x_N + sum_k=1^N-1 x_k^T Q_k x_k + q_k^T x_k + u_k^T R_k u_k + r_k^T u_k + u_k^T H_k x_kThis type of quadratic cost is typical for trajectory optimization problems, especially when Q is positive semi-definite and R is positive definite, which is strictly convex. These problem behave well and reduce the computational requirements of taking second-order Taylor series expansions of the cost at each iteration."
},

{
    "location": "costfunctions/#Creating-a-Cost-function-1",
    "page": "2. Setting up a Cost Function",
    "title": "Creating a Cost function",
    "category": "section",
    "text": "There are several different cost function types that all inherit from CostFunction. The following sections detail the various methods for instantiating these cost function types."
},

{
    "location": "costfunctions/#Quadratic-Costs-1",
    "page": "2. Setting up a Cost Function",
    "title": "Quadratic Costs",
    "category": "section",
    "text": "Quadratic costs are the most standard cost function and excellent place to start. Let\'s assume we are creating an LQR tracking cost of the form(x_N - x_f)^T Q_f (x_N - x_f) + sum_k=1^N-1 (x_k - x_f)^T Q (x_k - x_f) + u_k^T R u_kfor the simple pendulum with the goal of doing a swing-up. To do this we have a very convenient constructorusing LinearAlgebra\nn,m = 2,1\nQ = Diagonal(0.1I,n)\nR = Diagonal(0.1I,m)\nQf = Diagonal(1000I,n)\nxf = [π,0]\ncostfun = LQRCost(Q,R,Qf,xf)It is HIGHLY recommended to specify any special structure, such as Diagonal, especially since these matrices are almost always diagonal.This constructor actually does a simple conversion to turn our cost function into a generic quadratic cost function. We could do this ourselves:H = zeros(m,n)\nq = -Q*xf\nr = zeros(m)\nc = xf\'Q*xf/2\nqf = -Qf*xf\ncf = xf\'Qf*xf/2\ncostfun = QuadraticCost(Q, R, H, q, r, c, Qf, qf, cf)QuadraticCost"
},

{
    "location": "costfunctions/#Generic-Costs-1",
    "page": "2. Setting up a Cost Function",
    "title": "Generic Costs",
    "category": "section",
    "text": "For general, non-linear cost functions use GenericCost. Generic cost functions must define their second-order Taylor series expansion, either automatically using ForwardDiff or analytically.Let\'s say we wanted to use the nonlinear objective for the pendulumcos(theta_N) + omega_N^2 sum_k=1^N-1 cos(theta_k) + u_k^T R + u_k + Q ω^2which is small when θ = π, encouraging swing-up.We define the cost function by defining ℓ(x,u) and ell_f(x)# Define the stage and terminal cost functions\nfunction mycost(x,u)\n    R = Diagonal(0.1I,1)\n    Q = 0.1\n    return cos(x[1] + u\'R*u + Q*x[2]^2)\nend\nfunction mycost(xN)\n    return cos(xN[1]) + xN[2]^2\nend\n\n# Create the nonlinear cost function\nnlcost = GenericCost(mycost,mycost,n,m)This will use ForwardDiff to generate the gradient and Hessian needed for the 2nd order expansion.Performance-wise, it will be faster to specify the Jacobian analytically (which could also use ForwardDiff for part of it). We just need to define the following functionshess: multiple-dispatch function of the form,   Q,R,H = hess(x,u) with sizes (n,n), (m,m), (m,n)   Qf = hess(xN) with size (n,n)\ngrad: multiple-dispatch function of the form,   q,r = grad(x,u) with sizes (n,), (m,)   qf = grad(x,u) with size (n,)Here\'s an example for the nonlinear cost function we used before# Define the gradient and Hessian functions\nR = Diagonal(0.1I,m)\nQ = 0.1\nfunction hess(x,u)\n    n,m = length(x),length(u)\n    Qexp = Diagonal([-cos(x[1]), 2Q])\n    Rexp = 2R\n    H = zeros(m,n)\n    return Qexp,Rexp,Hexp\nend\nfunction hess(x)\n    return Diagonal([-cos(x[1]), 2])\nend\nfunction grad(x,u)\n    q = [-sin(x[1]), 2Q*x[2]]\n    r = 2R*u\n    return q,r\nend\nfunction grad(x)\n    return [-sin(x[1]), 2*x[2]]\nend\n\n# Create the cost function\nnlcost = GenericCost(mycost, mycost, grad, hess, n, m)"
},

{
    "location": "constraints/#",
    "page": "4. Add Constraints",
    "title": "4. Add Constraints",
    "category": "page",
    "text": ""
},

{
    "location": "constraints/#.-Add-Constraints-1",
    "page": "4. Add Constraints",
    "title": "4. Add Constraints",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"constraints.md\"]"
},

{
    "location": "constraints/#TrajectoryOptimization.Constraint",
    "page": "4. Add Constraints",
    "title": "TrajectoryOptimization.Constraint",
    "category": "type",
    "text": "struct Constraint{S} <: TrajectoryOptimization.AbstractConstraint{S}\n\nGeneral nonlinear constraint\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.ConstraintType",
    "page": "4. Add Constraints",
    "title": "TrajectoryOptimization.ConstraintType",
    "category": "type",
    "text": "Sense of a constraint (inequality / equality / null)\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.Equality",
    "page": "4. Add Constraints",
    "title": "TrajectoryOptimization.Equality",
    "category": "type",
    "text": "Inequality constraints\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.Inequality",
    "page": "4. Add Constraints",
    "title": "TrajectoryOptimization.Inequality",
    "category": "type",
    "text": "Equality constraints\n\n\n\n\n\n"
},

{
    "location": "constraints/#Constraint-Type-1",
    "page": "4. Add Constraints",
    "title": "Constraint Type",
    "category": "section",
    "text": "Constraint\nConstraintType\nEquality\nInequalityAll constraint types inherit from AbstractConstraint and are parameterized by ConstraintType, which specifies the type of constraint, Inequality or Equality. This allows the software to easily dispatch over the type of constraint. Each constraint type represents a vector-valued constraint. The intention is that each constraint type represent one line in constraints of problem definition (where they may be vector or scalar-valued). Each constraint has the following interface:evaluate!(v, con, x, u): Stage constraint evaluate!(v, con, x): Terminal constraint jacobian!(V, con, x, u): Jacobian wrt states and controls at a stage time step jacobian!(V, con, x): Jacobian wrt terminal state at terminal time step is_terminal(con): Boolean true if the constraint is defined at the terminal time step is_stage(con): Boolean true if the constraint is defined at the stage time steps length(con): Number of constraints (length of the output vector)There are currently two types of constraints implementedConstraint\nBoundConstraint"
},

{
    "location": "constraints/#General-Constraints-1",
    "page": "4. Add Constraints",
    "title": "General Constraints",
    "category": "section",
    "text": "Each Constraint contains the following fields:c: the in-place constraint function. Methods dispatch over constraint functions of the form c(v,x,u) and c(v,x).\n∇c: the in-place constraint jacobian function defined as ∇c(Z,x,u) where Z is the p × (n+m) concatenated Jacobian. p: number of elements in the constraint vector\nlabel: a Symbol for identifying the constraint\ntype: a Symbol identifying where the constraint applies. One of [:stage, :terminal, :all]Let\'s say we have a problem with 3 states and 2 controls, the following constraints x_1^2 + x_2^2 - 1 leq 0, x_2 + u_1  = 0, and x_3 + u_2 = 0. These two constraints could be created as follows:n,m = 3,2\n\n# Inequality Constraint\nc(v,x,u) = v[1] = x[1]^2 + x[2]^2 - 1\np1 = 1\ncon = Constraint{Inequality}(c, n, m, p1, :mycon1)\n\n# Equality Constraint\nc_eq(v,x,u) = v[1:2] = x[2:3] + u\np2 = 2\ncon_eq = Constraint{Equality}(c_eq, n, m, p2, :mycon2)Here we let the constructor build the Jacobians using ForwardDiff. We can alternatively specify them explicitly:∇c(V,x,u) = begin V[1,1] = 2x[1]; V[1,2] = 2x[2]; end\ncon = Constraint{Inequality}(c, ∇c, n, m, p1, :mycon1)\n\n∇c_eq(V,x,u) = begin V[1,2] = 1; V[1,4] = 1;\n                     V[2,3] = 1; V[2,5] = 1; end\ncon_eq = Constraint{Equality}(c_eq, ∇c_eq, n, m, p2, :mycon2)We can also build a terminal constraint that only depends on the terminal state, x_N. Let\'s say we have a terminal constraint x_1 + x_2 + x_3 = 1. We can create it using similar constructors:c_term(v,x) = sum(x) - 1\np_term = 1\ncon_term = Constraint{Equality}(c_term, n, p_term, :con_term)\n\n# We can also optionally give it an analytical Jacobian\n∇c_term(V,x) = ones(1,n)\ncon_term = Constraint{Equality}(c_term, ∇c_term, n, p_term, :con_term)Every constraint can be applied to both stage and terminal time steps. The constructor automatically determines which is applied based on the methods defined for the provided function. You can check this by inspecting the type field, which will be one of [:stage, :terminal, :all].Notice our first constraint is only dependent on the state. If we want to enforce it at the terminal time step we can simply use multiple dispatch:con.label == :stage  # true\nc(v,x) = v[1] = x[1]^2 + x[2]^2 - 1\ncon = Constraint{Inequality}(c, n, m, p1, :mycon1)\ncon.label == :all  # trueThe type can easily be checked with the is_terminal and is_stage commands.In summary, here are the constructors:Constraint{S}(c::Function, ∇c::Function, n::Int, m::Int, p::Int, label::Symbol) where S<:ConstraintType\nConstraint{S}(c::Function, n::Int, m::Int, p::Int, label::Symbol) where S<:ConstraintType\nConstraint{S}(c::Function, ∇c::Function, n::Int, p::Int, label::Symbol) where S<:ConstraintType\nConstraint{S}(c::Function, n::Int, p::Int, label::Symbol) where S<:ConstraintTypeGiven that constraints can apply at any time step, we assume that there is the same number of constraints for both stage and terminal time steps. The length method can also accept either :stage or :terminal as a second argument to specify which length you want (since one may be zero). e.g.length(con, :stage) == length(con, :terminal)  == 1  # true\nlength(con_eq, :stage) == 2     # true\nlength(con_eq, :terminal) == 0  # true"
},

{
    "location": "constraints/#TrajectoryOptimization.goal_constraint",
    "page": "4. Add Constraints",
    "title": "TrajectoryOptimization.goal_constraint",
    "category": "function",
    "text": "Creates a terminal equality constraint specifying the goal. All states must be specified.\n\n\n\n\n\n"
},

{
    "location": "constraints/#TrajectoryOptimization.planar_obstacle_constraint",
    "page": "4. Add Constraints",
    "title": "TrajectoryOptimization.planar_obstacle_constraint",
    "category": "function",
    "text": "planar_obstacle_constraint(n, m, x_obs, r_obs)\nplanar_obstacle_constraint(n, m, x_obs, r_obs, label)\n\n\nA constraint where x,y positions of the state must remain a distance r from a circle centered at x_obs Assumes x,y are the first two dimensions of the state vector\n\n\n\n\n\n"
},

{
    "location": "constraints/#Special-Constraints-1",
    "page": "4. Add Constraints",
    "title": "Special Constraints",
    "category": "section",
    "text": "A few constructors for common constraints have been provided:goal_constraint\nplanar_obstacle_constraint"
},

{
    "location": "constraints/#Bound-Constraints-1",
    "page": "4. Add Constraints",
    "title": "Bound Constraints",
    "category": "section",
    "text": "Bound constraints define simple bounds on the states and controls, allowing the solver to efficiently dispatch on methods to handle these simple constraints (especially when using direct methods). The constructor is very simpleBoundConstraint(n, m; x_min, x_max, u_min, x_max)The bounds can be given by vectors of the appropriate length, using ±Inf for unbounded values, or a scalar can be passed in when all the states have the same bound. If left blank, the value is assumed to be unbounded.Working from the previous examples, let\'s say we have -1 geq x_1 leq 1 x_3 leq 10 -15 geq u leq 12:bnd = BoundConstraint(n, m, x_min=[-1,-Inf,-Inf], x_max=[1,Inf,10],\n                            u_min=-15, u_max=12)Note that bound constraints are automatically valid at both stage and terminal time steps, i.e. evaluate!(v, bnd, x, u) and evaluate!(v, bnd, x) are both defined."
},

{
    "location": "constraints/#Constraint-Sets-1",
    "page": "4. Add Constraints",
    "title": "Constraint Sets",
    "category": "section",
    "text": "A ConstraintSet is simply a vector of constraints, and represents a set of constraints at a particular time step. There are some convenient methods for creating and working with constraint sets.Let\'s say we combine the previous constraints into a single constraint set. We can do this easily using the + method:constraints = con + con_eq + bnd\nconstraints_term = con + con_term + bndThere are several functions provided to work with ConstraintSetsBase.pop!(C::ConstraintSet, label::Symbol)\nevaluate!(c::PartedVector, C::ConstraintSet, x, u)\nevaluate!(c::PartedVector, C::ConstraintSet, x)\njacobian!(c::PartedMatrix, C::ConstraintSet, x, u)\njacobian!(c::PartedMatrix, C::ConstraintSet, x)The PartedVector and PartedMatrix needed for the evaluate! and jacobian! methods can be generated usingPartedVector(C::ConstraintSet, type=:stage)\nPartedMatrix(C::ConstraintSet, type=:stage)"
},

{
    "location": "constraints/#Problem-Constraints-1",
    "page": "4. Add Constraints",
    "title": "Problem Constraints",
    "category": "section",
    "text": "A Problem is made up of individual ConstraintSets at each of the N time steps, allowing for different constraints along the trajectory. The collection of ConstraintSets is captured in the ProblemConstraints type. There are several methods for constructing ProblemConstraints:# Create an empty set\nProblemConstraints(N)\n\n# Copy a single ConstraintSet over every time step\nProblemConstraints(constraints, N)\n\n# Use a different set at the terminal time step\nProblemConstraints(constraints, constraints_term, N)\n\n# Create from a Vector of Constraint Sets\nProblemConstraints([constraints, constraints, constraints, constraints_term])You can easily add or remove constraints from time steps using + and pop! on the appropriate time step:pcon = ProblemConstraints(N)\npcon[1] += con\npcon[2] += con + con_eq\npop!(pcon[2], :mycon2)\npcon[N] += con_term"
},

{
    "location": "problem/#",
    "page": "Setting up a Problem",
    "title": "Setting up a Problem",
    "category": "page",
    "text": ""
},

{
    "location": "problem/#Setting-up-a-Problem-1",
    "page": "Setting up a Problem",
    "title": "Setting up a Problem",
    "category": "section",
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"problem.md\"]The Problem type represents the trajectory optimization problem to be solved, which consists of the following information:Dynamics model: the system that is being controlled, specified by differential or difference equations.\nObjective: a collection of CostFunction\'s to be minimized of the form ell_N(x_N) + sum_k=0^N ell(x_ku_k)\nProblemConstraints: Optional stage wise constraints of the form c_k(x_ku_k) or c_N(x_N).\nInitial state: all trajectory optimization algorithms require the initial state.\nN: the number of knot points (or number of discretization points)\ndt: the time step used for discretizing the continuous dynamics\ntf: the total duration of the trajectoryThe Problem type also stores the nominal state and control input trajectories (i.e. the primal variables)."
},

{
    "location": "problem/#Creating-a-Problem-1",
    "page": "Setting up a Problem",
    "title": "Creating a Problem",
    "category": "section",
    "text": "A problem is typically created using the following constructorsProblem(model::Model,cost::CostFunction,x0::Vector{T},U::VectorTrajectory{T},dt::T) where T\nProblem(model::Model,cost::CostFunction,x0::Vector{T},U::Matrix{T},dt::T) where Twhere U is either an n × N-1 matrix or a vector of n-dimensional vectors (VectorTrajectory). A bare-minimum constructor is also availableProblem(model::Model,cost::CostFunction,N::Int,dt::T) where Twhich initializes the initial state and the control input trajectory to zeros.The inner constructor can also be used with caution.Problem(model::Model, cost::CostFunction, constraints::ConstraintSet,\n      x0::Vector{T}, X::VectorTrajectory, U::VectorTrajectory, N::Int, dt::T) where T"
},

{
    "location": "problem/#Adding-constraints-1",
    "page": "Setting up a Problem",
    "title": "Adding constraints",
    "category": "section",
    "text": "A constraint can be added once the problem is created usingadd_constraints"
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
    "location": "solvers/#TrajectoryOptimization.iLQRSolverOptions",
    "page": "Solvers",
    "title": "TrajectoryOptimization.iLQRSolverOptions",
    "category": "type",
    "text": "mutable struct iLQRSolverOptions{T} <: TrajectoryOptimization.AbstractSolverOptions{T}\n\nSolver options for the iterative LQR (iLQR) solver. iLQR is an indirect, unconstrained solver.\n\nverbose\nPrint summary at each iteration Default: false\nlive_plotting\nLive plotting Default: :off\ncost_tolerance\ndJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve Default: 0.0001\ngradient_type\ngradient type: :todorov, :feedforward Default: :todorov\ngradient_norm_tolerance\ngradient_norm < ϵ, gradient norm convergence criteria Default: 1.0e-5\niterations\niLQR iterations Default: 300\ndJ_counter_limit\nrestricts the total number of times a forward pass fails, resulting in regularization, before exiting Default: 10\nsquare_root\nuse square root method backward pass for numerical conditioning Default: false\nline_search_lower_bound\nforward pass approximate line search lower bound, 0 < linesearchlowerbound < linesearchupperbound Default: 1.0e-8\nline_search_upper_bound\nforward pass approximate line search upper bound, 0 < linesearchlowerbound < linesearchupperbound < ∞ Default: 10.0\niterations_linesearch\nmaximum number of backtracking steps during forward pass line search Default: 20\nbp_reg_initial\ninitial regularization Default: 0.0\nbp_reg_increase_factor\nregularization scaling factor Default: 1.6\nbp_reg_max\nmaximum regularization value Default: 1.0e8\nbp_reg_min\nminimum regularization value Default: 1.0e-8\nbp_reg_type\ntype of regularization- control: () + ρI, state: (S + ρI); see Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization Default: :control\nbp_reg_fp\nadditive regularization when forward pass reaches max iterations Default: 10.0\nbp_sqrt_inv_type\ntype of matrix inversion for bp sqrt step Default: :pseudo\nbp_reg_sqrt_initial\ninitial regularization for square root method Default: 1.0e-6\nbp_reg_sqrt_increase_factor\nregularization scaling factor for square root method Default: 10.0\nmax_cost_value\nmaximum cost value, if exceded solve will error Default: 1.0e8\nmax_state_value\nmaximum state value, evaluated during rollout, if exceded solve will error Default: 1.0e8\nmax_control_value\nmaximum control value, evaluated during rollout, if exceded solve will error Default: 1.0e8\n\n\n\n\n\n"
},

{
    "location": "solvers/#Iterative-LQR-(iLQR)-1",
    "page": "Solvers",
    "title": "Iterative LQR (iLQR)",
    "category": "section",
    "text": "iLQR is an unconstrained indirect method for trajectory optimization that parameterizes only the controls and enforces strict dynamics feasibility at every iteration by simulating forward the dynamics with an LQR feedback controller. The main algorithm consists of two parts: 1) a backward pass that uses Differential Dynamic Programming to compute recursively a quadratic approximation of the cost-to-go, along with linear feedback and feed-forward gain matrices, K and d, respectively, for an LQR tracking controller, and 2) a forward pass that uses the gains K and d to simulate forward the full nonlinear dynamics with feedback.The iLQR solver has the following solver optionsiLQRSolverOptions"
},

{
    "location": "solvers/#TrajectoryOptimization.AugmentedLagrangianSolverOptions",
    "page": "Solvers",
    "title": "TrajectoryOptimization.AugmentedLagrangianSolverOptions",
    "category": "type",
    "text": "mutable struct AugmentedLagrangianSolverOptions{T} <: TrajectoryOptimization.AbstractSolverOptions{T}\n\nSolver options for the augmented Lagrangian solver.     Augmented Lagrangian is a general method for solving constrained problems by solving a sequence of unconstrained problems.\n\nverbose\nPrint summary at each iteration Default: false\nopts_uncon\nunconstrained solver options Default: iLQRSolverOptions{T}()\ncost_tolerance\ndJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve Default: 0.0001\ncost_tolerance_intermediate\ndJ < ϵ_int, intermediate cost convergence criteria to enter outerloop of constrained solve Default: 0.001\ngradient_norm_tolerance\ngradient_norm < ϵ, gradient norm convergence criteria Default: 1.0e-5\ngradient_norm_tolerance_intermediate\ngradientnormint < ϵ, gradient norm intermediate convergence criteria Default: 1.0e-5\nconstraint_tolerance\nmax(constraint) < ϵ, constraint convergence criteria Default: 0.001\nconstraint_tolerance_intermediate\nmax(constraint) < ϵ_int, intermediate constraint convergence criteria Default: 0.001\niterations\nmaximum outerloop updates Default: 30\ndual_min\nminimum Lagrange multiplier Default: -1.0e8\ndual_max\nmaximum Lagrange multiplier Default: 1.0e8\npenalty_max\nmaximum penalty term Default: 1.0e8\npenalty_initial\ninitial penalty term Default: 1.0\npenalty_scaling\npenalty update multiplier; penalty_scaling > 0 Default: 10.0\npenalty_scaling_no\npenalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ) Default: 1.0\nconstraint_decrease_ratio\nratio of current constraint to previous constraint violation; 0 < constraintdecreaseratio < 1 Default: 0.25\nouter_loop_update_type\ntype of outer loop update (default, feedback) Default: :default\nactive_constraint_tolerance\nnumerical tolerance for constraint violation Default: 0.0\n\n\n\n\n\n"
},

{
    "location": "solvers/#Augmented-Lagrangian-1",
    "page": "Solvers",
    "title": "Augmented Lagrangian",
    "category": "section",
    "text": "Augmented LagrangianAugmentedLagrangianSolverOptions"
},

{
    "location": "solvers/#ALTRO-1",
    "page": "Solvers",
    "title": "ALTRO",
    "category": "section",
    "text": "ALTROALTROSolverOptions"
},

{
    "location": "solvers/#Direct-Collocation-(DIRCOL)-1",
    "page": "Solvers",
    "title": "Direct Collocation (DIRCOL)",
    "category": "section",
    "text": ""
},

]}
