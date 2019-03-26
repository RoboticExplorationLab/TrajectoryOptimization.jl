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
    "text": "The purpose of this package is to provide a testbed for state-of-the-art trajectory optimization algorithms. In general, this package focuses on trajectory optimization problems of the form (put LaTeX here)This package currently implements both indirect and direct methods for trajectory optimization:Iterative LQR (iLQR): indirect method based on differential dynamic programming\nDirect Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solverKey features include the use of ForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints; the use of RigidBodyDynamics to work directly from URDF files; and the ability to specify general constraints.The primary focus of this package is developing the ALTRO algorithm, although we hope this will extend to many algorithms in the future."
},

{
    "location": "#Getting-Started-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Getting Started",
    "category": "section",
    "text": "To set up and solve a trajectory optimization problem with TrajectoryOptimization.jl, the user will go through the following steps:Create a Model\nCreate a CostFunction\nInstantiate a Problem\nAdd constraints\nPick an appropriate solver\nSolve the problem\nAnalyze the solution"
},

{
    "location": "#Creating-a-Model-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Creating a Model",
    "category": "section",
    "text": "There are two ways of creating a model:An in-place analytic function of the form f(ẋ,x,u)\nA URDF"
},

{
    "location": "#TrajectoryOptimization.Model-Tuple{Function,Int64,Int64}",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.Model",
    "category": "method",
    "text": "Model(f::Function, n::Int64, m::Int64) -> TrajectoryOptimization.AnalyticalModel{Continuous}\nModel(f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}) -> TrajectoryOptimization.AnalyticalModel{Continuous}\n\n\nCreate a dynamics model, using ForwardDiff to generate the dynamics jacobian, with parameters Dynamics function passes in parameters:     f(ẋ,x,u,p)     where p in NamedTuple of parameters\n\n\n\n\n\n"
},

{
    "location": "#Analytic-Models-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Analytic Models",
    "category": "section",
    "text": "To create an analytic model create an in-place function for the continuous dynamics. The function must be of the form f(ẋ,x,u) where ẋ ∈ Rⁿ is the state derivative vector, x ∈ Rⁿ is the state vector, and u ∈ Rᵐ is the control input vector. The function should not return any values, but should write ẋ \"inplace,\" e.g. ẋ[1] = x[2]*u[2] NOT ẋ = f(x,u). This makes a significant difference in performance.Specifying discrete-time dynamics directly is currently not supported (but should be straight-forward to implement).The Model type is then created using the following signature: model = Model(f,n,m) where n is the dimension of the state input and m is the dimension of the control input.Model(f::Function, n::Int, m::Int)"
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
    "text": "While the model defines the dynamics of the system, the Objective defines what you want the dynamics to do. The Objective class defines the objective function via a CostFunction type, as well as the initial states and trajectory duration. The Objective class also specifies constraints on the states and controls. Both iLQR and Direct Collocation (DIRCOL) allow generic cost functions of the form g(xu) leq 0 or h(xu) = 0: any generic function of the state and control is permitted, but no couples between time steps is allowed."
},

{
    "location": "#TrajectoryOptimization.QuadraticCost",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.QuadraticCost",
    "category": "type",
    "text": "mutable struct QuadraticCost{TM, TH, TV, T} <: TrajectoryOptimization.CostFunction\n\nCost function of the form     1/2xₙᵀ Qf xₙ + qfᵀxₙ +  ∫ ( 1/2xᵀQx + 1/2uᵀRu + xᵀHu + q⁠ᵀx  rᵀu ) dt from 0 to tf R must be positive definite, Q and Qf must be positive semidefinite\n\n\n\n\n\n"
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
    "text": "The cost (or objective) function is the first piece of the objective. While the majority of trajectory optimization problems have quadratic objectives, TrjaectoryOptimization.jl allows the user to specify any generic cost function of the form ell_N(x_N) + sum_k=0^N ell(x_ku_k). Currently GenericObjective is only supported by iLQR, and not by DIRCOL. Since iLQR relies on 2nd Order Taylor Series Expansions of the cost, the user may specify analytical functions for this expansion in order to increase performance; if the user does not specify an analytical expansion it will be generated using ForwardDiff.QuadraticCost\nLQRCost\nGenericCost"
},

{
    "location": "#TrajectoryOptimization.UnconstrainedObjective",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.UnconstrainedObjective",
    "category": "type",
    "text": "struct UnconstrainedObjective{C} <: TrajectoryOptimization.Objective\n\nDefines an objective for an unconstrained optimization problem. xf does not have to specified. It is provided for convenience when used as part of the cost function (see LQRObjective function) If tf = 0, the objective is assumed to be minimum-time.\n\n\n\n\n\n"
},

{
    "location": "#TrajectoryOptimization.ConstrainedObjective",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.ConstrainedObjective",
    "category": "type",
    "text": "struct ConstrainedObjective{C} <: TrajectoryOptimization.Objective\n\nDefine a quadratic objective for a constrained optimization problem.\n\nConstraint formulation\n\nEquality constraints: f(x,u) = 0\nInequality constraints: f(x,u) ≥ 0\n\n\n\n\n\n"
},

{
    "location": "#TrajectoryOptimization.LQRObjective",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.LQRObjective",
    "category": "function",
    "text": "LQRObjective(Q, R, Qf, tf, x0, xf)\n\n\nCreate unconstrained objective for a problem of the form:     min (xₙ - xf)ᵀ Qf (xₙ - xf) + ∫ ( (x-xf)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf     s.t. x(0) = x0          x(tf) = xf\n\n\n\n\n\n"
},

{
    "location": "#TrajectoryOptimization.update_objective",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.update_objective",
    "category": "function",
    "text": "update_objective(obj; cost, tf, x0, xf, u_min, u_max, x_min, x_max, cI, cE, cI_N, cE_N, use_xf_equality_constraint, cIx, cIu, cEx, cEu, cI_Nx, cE_Nx)\n\n\nUpdates constrained objective values and returns a new objective.\n\nOnly updates the specified fields, all others are copied from the previous Objective.\n\n\n\n\n\n"
},

{
    "location": "#Creating-the-Objective-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Creating the Objective",
    "category": "section",
    "text": "Once the cost function is specified, the user then creates either an Unconstrained or Constrained Objective. When running iLQR, specifying any ConstrainedObjective will perform outer loop updates using an Augmented Lagrangian method.UnconstrainedObjective\nConstrainedObjective\nLQRObjectiveSince objectives are immutable types, the user can \"update\" the objective using the following functionupdate_objective"
},

{
    "location": "#TrajectoryOptimization.Solver",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.Solver",
    "category": "type",
    "text": "struct Solver{M<:Model, O<:TrajectoryOptimization.Objective}\n\nResponsible for storing all solve-dependent variables and solve parameters.\n\n\n\n\n\n"
},

{
    "location": "#Solving-the-Problem-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Solving the Problem",
    "category": "section",
    "text": "With a defined model and objective, the next step is to create a Solver type. The Solver is responsible for storing solve-dependent variables (such as number of knot points, step size, discrete dynamics functions, etc.) and storing parameters used during the solve (via SolverOptions). The solver contains both the Model and Objective and contains all information needed for the solve, except for the initial trajectories.SolverOnce the solver is created, the user must create an initial guess for the control trajectory, and optionally a state trajectory. For simple problems a initialization of random values, ones, or zeros works well. For more complicated systems it is usually recommended to feed trim conditions, i.e. controls that maintain the initial state values. For convenience, the function get_sizes returns n,m,N from the solver. Note that for trajectory optimization the control trajectory should be length N-1 since there are no controls at the final time step. However, DIRCOL uses controls at the final time step, and iLQR will simply discard any controls at the time step. Therefore, an initial control trajectory of size (m,N) is valid (but be aware that iLQR will return the correctly-sized control trajectory). Once the initial state and control trajectories are specified, they are passed with the solver to one of the solve methods."
},

{
    "location": "#Solve-Methods-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Solve Methods",
    "category": "section",
    "text": "With a Solver instantiated, the user can then choose to solve the problem using iLQR (solve function) or DIRCOL (solve_dircol function), where are detailed below"
},

{
    "location": "#iLQR-Methods-1",
    "page": "TrajectoryOptimization.jl",
    "title": "iLQR Methods",
    "category": "section",
    "text": ""
},

{
    "location": "#Unconstrained-Problem-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Unconstrained Problem",
    "category": "section",
    "text": "For unconstrained problems the user doesn\'t have any options. iLQR can usually solve unconstrained problems without any modification. Simply call the solve method, passing in a initial guess for the control trajectory:solve(solver,U0)where U0 is a Matrix of size (m,N-1) (although a trajectory of N points will also be accepted)."
},

{
    "location": "#Constrained-Problem-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Constrained Problem",
    "category": "section",
    "text": "The default constrained iLQR method uses an Augmented Lagrangian approach to handle the constraints. Nearly all of the options in SolverOptions determine parameters used by the Augmented Lagrangian method. Other than now having more parameters to tune for better performance (see another section for tips), the user solves a constrained problem using the exact same method for solving an unconstrained problem."
},

{
    "location": "#Constrained-Problem-with-Infeasible-Start-1",
    "page": "TrajectoryOptimization.jl",
    "title": "Constrained Problem with Infeasible Start",
    "category": "section",
    "text": "One of the primary disadvantages of iLQR (and most indirect methods) is that the user must specify an initial input trajectory. Specifying a good initial guess can often be difficult in practice, whereas specifying a guess for the state trajectory is typically more straightforward. To overcome this limitation, TrajectoryOptimization adds artificial controls to the discrete dynamics $x{k+1} = fd(xk,uk) + \\diag{(\\tidle{u}1,\\hdots,\\tidle{u}n)} such that the system is fully-actuated (technically over-actuated), so that an arbitrary state trajectory can be achieved. These artificial controls are then constrained to be zero using the Augmented Lagrangian method. This results in an algorithm similar to that of DIRCOL: initial solutions are dynamically infeasible but become dynamically infeasible at convergence. To solve the problem using \"infeasible start\", simply pass in an initial guess for the state and control:solve(solver,X0,U0)"
},

{
    "location": "#DIRCOL-Method-1",
    "page": "TrajectoryOptimization.jl",
    "title": "DIRCOL Method",
    "category": "section",
    "text": "Problems can be solved using DIRCOL by simply callingsolve_dircol(solver,X0,U0)"
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
    "text": "The Model type holds information about the dynamics of the system. All dynamics are assumed to be state-space models of the system of the form ẋ = f(x,u) where ẋ is the state derivative, x an n-dimentional state vector, and u in an m-dimensional control input vector. The function f can be any nonlinear function.TrajectoryOptimization.jl poses the trajectory optimization problem by discretizing the state and control trajectories, which requires discretizing the dynamics, turning the continuous time differential equation into a discrete time difference equation of the form x[k+1] = f(x[k],u[k]), where k is the time step. There many methods of performing this discretization, and TrajectoryOptimization.jl offers several of the most common methods.Sometimes is it convenient to write down the difference equation directly, rather than running a differential equation through a discretizing integration method. TrajectoryOptimization.jl offers method deal directly with either continuous differential equations, or discrete difference equations.The Model type is parameterized by the DynamicsType, which is either Continuous, or Discrete. The models holds the equation f and it\'s Jacobian, ∇f, along with the dimensions of the state and control vectors.Models can be created by writing down the dynamics analytically or be generated from a URDF file via RigidBodyDynamics.jl."
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
    "text": "Let\'s start by writing down a dynamics function for a simple pendulum with state [θ; ω] and a torque control inputfunction pendulum_dynamics!(xdot,x,u)\n    m = 1.\n    l = 0.5\n    b = 0.1\n    lc = 0.5\n    J = 0.25\n    g = 9.81\n    xdot[1] = x[2]\n    xdot[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/J\nendNote that the function is in-place, in that it writes the result to the first argument. It is also good practice to concretely specify the location to write to rather than using something like xdot[1:end] or xdot[:].Notice that we had to specify a handful of constants when writing down the dynamics. We could have initialized them outside the scope of the function (which may result in global variables, so be careful!) or we can pass them in as a NamedTuple of parameters:function pendulum_dynamics_params!(xdot,x,u,p)\n    xdot[1] = x[2]\n    xdot[2] = (u[1] - p.m * p.g * p.lc * sin(x[1]) - p.b*x[2])/p.J\nendWe can now create our model using our analytical dynamics function with or without the parameters tuplen,m = 2,1\nmodel = Model(pendulum_dynamics!, n, m)\n\nparams = (m=1, l=0.5, b=0.1, lc=0.5, J=0.25, g=9.81)\nmodel = Model(pendulum_dynamics_params!, n, m, params)"
},

{
    "location": "models/#With-analytical-Jacobians-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "With analytical Jacobians",
    "category": "section",
    "text": "Since we have a very simple model, writing down an analytical expression of the Jacobian is pretty easy:function pendulum_jacobian!(Z,x,u)\n    m = 1.\n    l = 0.5\n    b = 0.1\n    lc = 0.5\n    J = 0.25\n    g = 9.81\n\n    Z[1,1] = 0                    # ∂θdot/∂θ\n    Z[1,2] = 1                    # ∂θdot/∂ω\n    Z[1,3] = 0                    # ∂θ/∂u\n    Z[2,1] = -m*g*lc*cos(x[1])/J  # ∂ωdot/∂θ\n    Z[2,2] = -b/J                 # ∂ωdot/∂ω\n    Z[2,3] = 1/J                  # ∂ωdot/∂u\nend\n\nfunction pendulum_jacobian_params!(Z,x,u,p)\n    Z[1,1] = 0                                    # ∂θdot/∂θ\n    Z[1,2] = 1                                    # ∂θdot/∂ω\n    Z[1,3] = 0                                    # ∂θ/∂u\n    Z[2,1] = -p.m * p.g * p.lc * cos(x[1]) / p.J  # ∂ωdot/∂θ\n    Z[2,2] = -p.b / p.J                           # ∂ωdot/∂ω\n    Z[2,3] = 1/p.J                                # ∂ωdot/∂u\nendWe can then pass these functions into the model instead of using ForwardDiff to calculate themmodel = Model(pendulum_dynamics!, pendulum_jacobian!, n, m)\nmodel = Model(pendulum_dynamics_params!, pendulum_jacobian_params!, n, m, params)"
},

{
    "location": "models/#URDF-Files-1",
    "page": "1. Setting up a Dynamics Model",
    "title": "URDF Files",
    "category": "section",
    "text": "Instead of writing down the dynamics explicity, we can import the dynamics from geometry specified in a URDF model using RigidBodyDynamics.jl. Let\'s say we have a URDF file for a double pendulum and don\'t want to bother writing down the dynamics, then we can create a model using any of the following methodsusing RigidBodyDynamics\n# From a string\nurdf = \"doublependulum.urdf\"\nmodel = Model(urdf)\n\n# From a RigidBodyDynamics `Mechanism` type\nmech = parse_urdf(urdf)  # return a Mechanism type\nmodel = Model(mech)Now let\'s say we want to control an acrobot, which can only control the first joint. We can pass in a vector of Booleans to specify which of the joints are \"active.\"joints = [true,false]\n\n# From a string\nurdf = \"doublependulum.urdf\"\nmodel = Model(urdf,joints)\n\n# From a RigidBodyDynamics `Mechanism` type\nmech = parse_urdf(urdf)  # return a Mechanism type\nmodel = Model(mech,joints)"
},

{
    "location": "models/#TrajectoryOptimization.AnalyticalModel",
    "page": "1. Setting up a Dynamics Model",
    "title": "TrajectoryOptimization.AnalyticalModel",
    "category": "type",
    "text": "struct AnalyticalModel{D} <: Model{D}\n\nDynamics model\n\nHolds all information required to uniquely describe a dynamic system, including a general nonlinear dynamics function of the form ẋ = f(x,u), where x ∈ ℜⁿ are the states and u ∈ ℜᵐ are the controls.\n\nDynamics function Model.f should be in the following forms:     \'f!(ẋ,x,u)\' and modify ẋ in place\n\n\n\n\n\n"
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
    "text": "Assuming we have a model of type Model{Continuous}, we can discretize as follows:model_discrete = Model{Discrete}(model,discretizer)where discretizer is a function that returns a discretized version of the continuous dynamics. TrajectoryOptimization.jl offers the following integration schemesmidpoint\nrk3 (Third Order Runge-Kutta)\nrk4 (Fourth Order Runge-Kutta)So to create a discrete model of the pendulum with fourth order Runge-Kutta integration we would do the following# Create the continuous model (any of the previously mentioned methods would work here)\nparams = (m=1, l=0.5, b=0.1, lc=0.5, J=0.25, g=9.81)\nmodel = Model(pendulum_dynamics_params!, n, m, params)\n\n# Discretize the continuous model\nmodel_discrete = Model{Discrete}(model,rk4)"
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
    "text": "All trajectory optimization problems require a cost function to specify the value(s) to be minimized. Cost functions must be scalar-valued. We assume cost functions of the following form ell_f(x_N) + sum_k=1^N-1 ell_k(x_ku_k) where ell_f and ell_k are general non-linear functions. It is very important to note that ell_k(x_ku_k) is ONLY a function of x_k and u_k, i.e. no coupling across time-steps is permitted. This is a requirement for recursive methods such as iLQR, but could be relaxed for methods that parameterize both states and controls, such as DIRCOL (although currently not supported). In general, any coupling between adjacent time-steps can be resolved by augmenting the state and defining the appropriate dynamics (this is the method we use to solve minimum time problems).In general, trajectory optimization will take a second order Taylor series approximation of the cost function, resulting in a quadratic cost function of the form x_N^T Q_f x_N + q_f^T x_N + sum_k=1^N-1 x_k^T Q_k x_k + q_k^T x_k + u_k^T R_k u_k + r_k^T u_k + u_k^T H_k x_k This type of quadratic cost is typical for trajectory optimization problems, especially when Q is positive semi-definite and R is positive definite, which is strictly convex. These problem behave well and reduce the computational requirements of taking second-order Taylor series expansions of the cost at each iteration.note: Note\nIf the dynamics were linear and the constraints convex, the trajectory optimization reduces to a convex optimization problem which can be solved reliably in a variety of ways. The algorithms in TrajectoryOptimization.jl focus on non-convex optimization, which arise with non-linear dynamics or non-convex constraints (such as obstacle avoidance)."
},

{
    "location": "costfunctions/#Creating-a-Cost-function-1",
    "page": "2. Setting up a Cost Function",
    "title": "Creating a Cost function",
    "category": "section",
    "text": "There are several different cost function types that all inherit from CostFunction. The following sections detail the various methods for instantiating these cost function types.note: Note\nAlthough cost functions with unique ell_k are possible, TrajectoryOptimization.jl currently only supports cost functions with a single stage cost function ell(x_ku_k). AugmentedLagrangianCost is a caveat to this."
},

{
    "location": "costfunctions/#Quadratic-Costs-1",
    "page": "2. Setting up a Cost Function",
    "title": "Quadratic Costs",
    "category": "section",
    "text": "Quadratic costs are the most standard cost function and excellent place to start. Let\'s assume we are creating an LQR tracking cost of the form (x_N - x_f)^T Q_f (x_N - x_f) + c_N + sum_k=1^N-1 (x_k - x_f)^T Q (x_k - x_f) + u_k^T R u_k + c_k for the simple pendulum with the goal of doing a swing-up. To do this we have a very convenient constructorusing LinearAlgebra\nn,m = 2,1\nQ = Diagonal(0.1I,n)\nR = Diagonal(0.1I,m)\nQf = Diagonal(1000I,n)\nxf = [π,0]\ncostfun = LQRCost(Q,R,Qf,xf)It is HIGHLY recommended to specify any special structure, such as Diagonal, especially since these matrices are almost always diagonal.This constructor actually does a simple conversion to turn our cost function into a generic quadratic cost function. We could do this ourselves:H = zeros(m,n)\nq = -Q*xf\nr = zeros(m)\nc = xf\'Q*xf/2\nqf = -Qf*xf\ncf = xf\'Qf*xf/2\ncostfun = QuadraticCost(Q, R, H, q, r, c, Qf, qf, cf)QuadraticCost"
},

{
    "location": "costfunctions/#TrajectoryOptimization.AugmentedLagrangianCost",
    "page": "2. Setting up a Cost Function",
    "title": "TrajectoryOptimization.AugmentedLagrangianCost",
    "category": "type",
    "text": "struct AugmentedLagrangianCost{T} <: TrajectoryOptimization.CostFunction\n\nCost function of the form     ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf\n\n\n\n\n\n"
},

{
    "location": "costfunctions/#Augmented-Lagrangian-Costs-1",
    "page": "2. Setting up a Cost Function",
    "title": "Augmented Lagrangian Costs",
    "category": "section",
    "text": "Augmented Lagrangian cost functions transform a constrained problem to an unconstrained problem by minimizing a cost function of the form J(XU) + λ^T c(XU) + frac12 c(XU)^T I_mu c(XU) where X and U are state and control trajectories, J(XU) is the original cost function, c(XU) is the vector-value constraint function, μ is the penalty parameter, and I_mu is a diagonal matrix that whose entries are μ for active constraints and 0 otherwise.Let\'s say we use our quadratic cost function we created for the pendulum, bound the control inputs between ±5, and use a terminal equality constraint for reaching the goal. We\'ll then form an AugmentedLagrangianCost.# Create constraints\nbnd = bound_constraint(n,m,u_min=-5,u_max=5)     # bound constraint on the controls\nterm = goal_constraint(n,xf)                     # terminal goal constraint\nconstraints = [bnd,term]                         # form a ConstraintSet\n\n# Create Cost function\nN = 21  # Number of knot points (need to initialize the trajectories)\nalcost = AugmentedLagrangianCost(cost, constraints, N)\nalcost = AugmentedLagrangianCost(cost, constraints, N, μ_init=10., λ_init=1.)When creating the cost function, you can initialize the penalty parameter to a specific value, as well as initialize the Lagrange multipliers to a value (which is typically zero). You can also optionally use the default inner constructor, but requires initializing all the needed variables. See source code of the previous constructor for an example.If you want to specify the Lagrange multipliers (say from a previous solve), you can use the following constructoralcost = AugmentedLagrangianCost(cost, constraints, λ, μ_init=1.)where the keyword argument for the initial penalty term is obviously optional. λ must be a PartedVecTrajectory, or a vector of partitioned vectors generated by PartedArrays. Examples for creating these can be found in # 4. Add Constraints.Typically augmented Lagrangian cost functions are formed as part of a solve with the AugmentedLagrangianSolver. The most convenient way to create one is from a Problem and an AugmentedLagrangianSolver.model = Dynamics.pendulum[1]\ndt = 0.1\nprob = Problem(model, costfun, N, dt)  # see Documentation for more way of creating a problem\nsolver = AugmentedLagrangianSolver(prob)\nalcost = AugmentedLagrangianCost(prob,solver)Note that this will not create new memory for arrays stored in alcost, they will simply point to the arrays stored in the solver.AugmentedLagrangianCost"
},

{
    "location": "costfunctions/#Generic-Costs-1",
    "page": "2. Setting up a Cost Function",
    "title": "Generic Costs",
    "category": "section",
    "text": "For general, non-linear cost functions use GenericCost. Generic cost functions must define their second-order Taylor series expansion, either automatically using ForwardDiff or analytically.Let\'s say we wanted to use the nonlinear cost function for the pendulum cos(theta_N) + omega_N^2 sum_k=1^N-1 cos(theta_k) + u_k^T R + u_k + Q ω^2 which is small when θ = π, encouraging swing-up.We define the cost function by defining ℓ(x,u) and ell_f(x)# Define the stage and terminal cost functions\nfunction mycost(x,u)\n    R = Diagonal(0.1I,1)\n    Q = 0.1\n    return cos(x[1] + u\'R*u + Q*x[2]^2)\nend\nfunction mycost(xN)\n    return cos(xN[1]) + xN[2]^2\nend\n\n# Create the nonlinear cost function\nnlcost = GenericCost(mycost,mycost,n,m)This will use ForwardDiff to generate the gradient and Hessian needed for the 2nd order expansion.Performance-wise, it will be faster to specify the Jacobian analytically (which could also use ForwardDiff for part of it). We just need to define the following functionshess: multiple-dispatch function of the form,   Q,R,H = hess(x,u) with sizes (n,n), (m,m), (m,n)   Qf = hess(xN) with size (n,n)\ngrad: multiple-dispatch function of the form,   q,r = grad(x,u) with sizes (n,), (m,)   qf = grad(x,u) with size (n,)Here\'s an example for the nonlinear cost function we used before# Define the gradient and Hessian functions\nR = Diagonal(0.1I,m)\nQ = 0.1\nfunction hess(x,u)\n    n,m = length(x),length(u)\n    Qexp = Diagonal([-cos(x[1]), 2Q])\n    Rexp = 2R\n    H = zeros(m,n)\n    return Qexp,Rexp,Hexp\nend\nfunction hess(x)\n    return Diagonal([-cos(x[1]), 2])\nend\nfunction grad(x,u)\n    q = [-sin(x[1]), 2Q*x[2]]\n    r = 2R*u\n    return q,r\nend\nfunction grad(x)\n    return [-sin(x[1]), 2*x[2]]\nend\n\n# Create the cost function\nnlcost = GenericCost(mycost, mycost, grad, hess, n, m)"
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
    "location": "constraints/#Constraint-Type-1",
    "page": "4. Add Constraints",
    "title": "Constraint Type",
    "category": "section",
    "text": "AbstractConstraint\nConstraint\nTerminalConstraint\nConstraintType\nEquality\nInequalityThere are two constraint types that inherit from AbstractConstraint: Constraint and TerminalConstraint. Both of these constraints are parameterized by a ConstraintType, which can be either Equality or Inequality. This allows the software to easily dispatch over the type of constraint. Each constraint type represents a vector-valued constraint. The intention is that each constraint type represent one line in constraints of problem definition (where they may be vectoxr or scalar-valued). Each constraint contains the following fields:c: the in-place constraint function. Of the form c(v,x,u) for Constraint and c(v,x) for TerminalConstraint.\n∇c: the in-place constraint jacobian function. For Constraint it can either be called as ∇c(A,B,x,u) where A is the state Jacobian and B is the control Jacobian, or as ∇c(Z,x,u) where Z is the p × (n+m) concatenated Jacobian. For TerminalConstraint there is only ∇c(A,x).\np: number of elements in the constraint vector\nlabel: a Symbol for identifying the constraint"
},

{
    "location": "constraints/#Creating-Constraints-1",
    "page": "4. Add Constraints",
    "title": "Creating Constraints",
    "category": "section",
    "text": "A stage-wise constraint can be created with either of the two constructorsConstraint{S}(c::Function,∇c::Function,p::Int,label::Symbol) where S<:ConstraintType\nConstraint{S}(c::Function,n::Int,m::Int,p::Int,label::Symbol) where S<:ConstraintTypeThe first is the default constructor. c must be in-place of the form c(v,x,u) where v holds the constraint function values. ∇c must be multiple dispatched to have the forms ∇c(A,B,x,u) where A is the state Jacobian and B is the control Jacobian, and ∇c(Z,x,u) where Z is the p × (n+m) concatenated Jacobian.The second will use ForwardDiff to generate the constraint Jacobian, so requires the size of the state and control input vectors.A terminal constraint can be similarly defined using one of the following constructorsTerminalConstraint{S}(c::Function,∇c::Function,p::Int,label::Symbol) where S<:ConstraintType\nTerminalConstraint{S}(c::Function,n::Int,p::Int,label::Symbol) where S<:ConstraintType\nConstraint{S}(c::Function,n::Int,p::Int,label::Symbol) where S<:ConstraintTypewhich are identical to the ones above, expect that they require a constraint function and Jacobian of the form c(v,x) and ∇c(A,x)."
},

{
    "location": "constraints/#TrajectoryOptimization.bound_constraint",
    "page": "4. Add Constraints",
    "title": "TrajectoryOptimization.bound_constraint",
    "category": "function",
    "text": "bound_constraint(n, m; x_min, x_max, u_min, u_max, trim)\n\n\nCreate a stage bound constraint Will default to bounds at infinity. \"trim\" will remove any bounds at infinity from the constraint function.\n\n\n\n\n\n"
},

{
    "location": "constraints/#Special-Constraints-1",
    "page": "4. Add Constraints",
    "title": "Special Constraints",
    "category": "section",
    "text": "A few constructors for common constraints have been provided:bound_constraint"
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
    "text": "CurrentModule = TrajectoryOptimizationPages = [\"problem.md\"]The Problem type represents the trajectory optimization problem to be solved, which consists of the following information:Dynamics model: the physical system that is being controlled, specified by a system of difference equations. We assume that continuous dynamics have been discretized, see From Continuous Model.\nCost function: the function to be minimized, and must be of the form ell_N(x_N) + sum_k=0^N ell(x_ku_k)\nConstraints: Aside from the dynamics constraints imposed by the dynamics model, the problem specifies \"stage\" constraints of the form c(x_ku_k) or \"terminal\" constraints of the form c(x_N).\nInitial state: all trajectory optimization algorithms require the initial state.\nN: the number of knot points (or number of discretization points)\ndt: the time step used for integrating the continuous dynamics, if required by the dynamics modelThe Problem type also stores the state and control input trajectories (i.e. the primal variables)."
},

{
    "location": "problem/#TrajectoryOptimization.Problem-Union{Tuple{T}, Tuple{Model,CostFunction,Int64,T}} where T",
    "page": "Setting up a Problem",
    "title": "TrajectoryOptimization.Problem",
    "category": "method",
    "text": "Problem(model, cost, N, dt)\n\n\nProblem(model::T, cost::T, N::T, dt::T)\n\n\nCreate a problem, initializing the initial state and control input trajectories to zeros\n\n\n\n\n\n"
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
    "text": "mutable struct iLQRSolverOptions{T} <: TrajectoryOptimization.AbstractSolverOptions{T}\n\nSolver options for the iterative LQR (iLQR) solver. iLQR is an indirect, unconstrained solver. DocStringExtensions.FIELDS\n\n\n\n\n\n"
},

{
    "location": "solvers/#Iterative-LQR-(iLQR)-1",
    "page": "Solvers",
    "title": "Iterative LQR (iLQR)",
    "category": "section",
    "text": "iLQR is an indirect method for trajectory optimization that parameterizes only the controls and enforces strict dynamics feasibility at every iteration by simulating forward the dynamics with an LQR feedback controller. The main algorithm consists of two parts: 1) a backward pass that uses differential dynamic programming to compute recursively a quadratic approximation of the cost-to-go, along with linear feedback and feed-forward gain matrices, K and d, respectively, for an LQR tracking controller, and 2) a forward pass that uses the gains K and d to simulate forward the dynamics with feedback.The vanilla iLQR algorithm is incapable of handling constraints aside from the dynamics. Any reference to the iLQR algorithm within TrajectoryOptimization.jl will assume the problem is solving an unconstrained problem. Other algorithms, such as ALTRO, use iLQR an an internal, unconstrained solver to solve a trajectory optimization problem with constraints.The iLQR solver has the following solver optionsiLQRSolverOptionsAugmented LagrangianAugmentedLagrangianSolverOptions"
},

]}
