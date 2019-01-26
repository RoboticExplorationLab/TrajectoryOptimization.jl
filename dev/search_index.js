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
    "text": "The purpose of this package is to provide a testbed for state-of-the-art trajectory optimization algorithms. In general, this package focuses on trajectory optimization problems of the form (put LaTeX here)This package currently implements both indirect and direct methods for trajectory optimization:Iterative LQR (iLQR): indirect method based on differential dynamic programming\nDirect Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solverKey features include the use of ForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints; the use of RigidBodyDynamics to work directly from URDF files; and the ability to specify general constraints.The primary focus of this package is developing the iLQR algorithm, although we hope this will extend to many algorithms in the future."
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
    "location": "#TrajectoryOptimization.Model-Tuple{Function,Int64,Int64}",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.Model",
    "category": "method",
    "text": "Model(f, n, m)\n\n\nCreate a Model given an inplace analytical function for the continuous dynamics with n states and m controls\n\n\n\n\n\n"
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
    "text": "mutable struct QuadraticCost{TM, TH, TV, T} <: TrajectoryOptimization.CostFunction\n\nCost function of the form     xₙᵀ Qf xₙ + qfᵀxₙ + ∫ ( xᵀQx + uᵀRu + xᵀHu + q⁠ᵀx  rᵀu ) dt from 0 to tf R must be positive definite, Q and Qf must be positive semidefinite\n\n\n\n\n\n"
},

{
    "location": "#TrajectoryOptimization.LQRCost",
    "page": "TrajectoryOptimization.jl",
    "title": "TrajectoryOptimization.LQRCost",
    "category": "function",
    "text": "LQRCost(Q, R, Qf, xf)\n\n\nCost function of the form     (xₙ-xf)ᵀ Qf (xₙ - xf) ∫ ( (x-x_f)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf R must be positive definite, Q and Qf must be positive semidefinite\n\n\n\n\n\n"
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

]}
