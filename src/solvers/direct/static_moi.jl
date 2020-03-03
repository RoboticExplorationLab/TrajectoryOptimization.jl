export
    DIRCOLSolver

"$(TYPEDEF) Solver options for the Direct Collocation solver.
Most options are passed to the NLP through the `opts` dictionary
$(FIELDS)"
@with_kw mutable struct DIRCOLSolverOptions{T} <: DirectSolverOptions{T}
    "NLP Solver to use. See MathOptInterface for available NLP solvers"
    nlp::MathOptInterface.AbstractOptimizer = Ipopt.Optimizer()

    "Options dictionary for the nlp solver"
    opts::Dict{Symbol,Any} = Dict{Symbol,Any}()

    "Print output to console"
    verbose::Bool = true

    "Feasibility tolerance"
    constraint_tolerance::T = -1.0
end

function DIRCOLSolverOptions(opts::SolverOptions)
	DIRCOLSolverOptions(
		constraint_tolerance = opts.constraint_tolerance,
		verbose=verbose
	)
end

"""
$(TYPEDEF)
Direct Collocation Solver.
Uses a commerical NLP solver to solve the Trajectory Optimization problem.
Uses the MathOptInterface to interface with the NLP.
"""
struct DIRCOLSolver{Q<:QuadratureRule,L,T,N,M,NM} <: DirectSolver{T}
    opts::SolverOptions{T}
    stats::Dict{Symbol,Any}

    NN::Int
    NP::Int

    dyn_con::DynamicsConstraint{Q,L,T,N,NM}
    objective::AbstractObjective
    constraints::ConstraintSet{T}
    constraints_all::ConstraintSet{T}
    Z::Vector{KnotPoint{T,N,M,NM}}
    x0::SVector{N,T}

    optimizer::MOI.AbstractOptimizer

    E::CostExpansion

    xinds::Vector{SVector{N,Int}}
    uinds::Vector{SVector{M,Int}}

    linds::Vector{<:Vector{SV} where SV}
    con_inds::Vector{<:Vector{SV} where SV}

    jacobian_structure::Symbol

    function DIRCOLSolver(opts,stats, NN,NP,dyn_con::DynamicsConstraint{Q,L,T,N},
            obj,conSet,conSet_all,Z::Vector{KnotPoint{T,N,M,NM}},
            x0, optimizer, E,
            xinds,uinds,linds,con_inds,jac_structure) where {Q,L,T,N,M,NM}
        new{Q,L,T,N,M,NM}(opts,stats, NN,NP,dyn_con,obj,conSet,conSet_all,Z,x0,optimizer,E,
            xinds,uinds,linds,con_inds,jac_structure)
    end

end

Base.size(solver::DIRCOLSolver{Q,L,T,n,m,NM}) where {Q,L,T,n,m,NM} = n,m,length(solver.Z)

function DIRCOLSolver(prob::Problem{Q},
        opts=SolverOptions(),
        jacobian_structure=:by_knotpoint,
    	opts_optimizer::Dict{Symbol,Any} = Dict{Symbol,Any}(),
		nlp=Ipopt.Optimizer()) where Q

    n,m,N = size(prob)
    Z = prob.Z

    stats = Dict{Symbol,Any}()

    # Add dynamics constraints
    prob = copy(prob)
    add_dynamics_constraints!(prob)
    dyn_con = get_constraints(prob).constraints[2].con::DynamicsConstraint{Q}
    conSet = get_constraints(prob)

    # Store a copy of the constraint set with all constraints
    conSet_all = copy(conSet)

    # Add bounds at infinity if the problem doesn't have any bound constraints
    if !any(is_bound.(conSet))
        bnd = BoundConstraint(n,m)
        add_constraint!(conSet, ConstraintVals(bnd, 1:N))
    end

    # Remove bounds
    zU,zL,gU,gL = get_bounds(conSet)

    # Initialize arrays
    dyn_vals = DynamicsVals(dyn_con)
    E = CostExpansion(n,m,N)

    NN = (n+m)*N
    NP = sum(num_constraints(prob))

    P = Primals(n,m,N)
    xinds, uinds = P.xinds, P.uinds

    blk_len = map(con->length(con.∇c[1]), conSet.constraints)
    con_len = map(con->length(con.∇c), conSet.constraints)
    con_inds = gen_con_inds(get_constraints(prob), jacobian_structure)

    # These get filled in with call to constraint_jacobian_structure
    linds = [[@SVector zeros(Int,blk) for i = 1:len] for (blk,len) in zip(blk_len, con_len)]

    # Create MOI Optimizer
    nlp_opts = Dict(Symbol(key)=>value for (key,val) in pairs(opts_optimizer))
    optimizer = typeof(nlp)(;nlp_opts..., nlp_options(nlp, opts)...)

    # Create Solver
    d = DIRCOLSolver(opts, stats, NN, NP, dyn_con, prob.obj, conSet, conSet_all,
        Z, prob.x0, optimizer, E, xinds, uinds, linds, con_inds, jacobian_structure)

    # Set up MOI problem
    V0 = zeros(NN)
    copyto!(V0, Z, xinds, uinds)

    has_objective = true
    nlp_bounds = MOI.NLPBoundsPair.(gL, gU)
    block_data = MOI.NLPBlockData(nlp_bounds, d, has_objective)

    V = MOI.add_variables(optimizer, NN)

    MOI.add_constraints(optimizer, V, MOI.LessThan.(zU))
    MOI.add_constraints(optimizer, V, MOI.GreaterThan.(zL))

    MOI.set(optimizer, MOI.VariablePrimalStart(), V, V0)

    # Set up NLP problem
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return d
end

@inline AbstractSolver(prob::Problem, opts::DIRCOLSolverOptions) = DIRCOLSolver(prob, opts)

# AbstractSolver Interface
get_initial_state(solver::DIRCOLSolver) = solver.x0
get_model(solver::DIRCOLSolver) = solver.dyn_con.model
get_constraints(solver::DIRCOLSolver) = solver.constraints
get_trajectory(solver::DIRCOLSolver) = solver.Z
get_objective(solver::DIRCOLSolver) = solver.objective
iterations(solver::DIRCOLSolver) = stats(solver)[:iterations]

num_primals(solver::DIRCOLSolver) = solver.NN
num_duals(solver::DIRCOLSolver) =  solver.NP

# Include bounds when calculating max violation on the solver
function max_violation(solver::DIRCOLSolver)
	Z = get_trajectory(solver)
    conSet = solver.constraints_all
    max_violation(conSet, Z)
end

primal_partition(prob::DIRCOLSolver) = prob.xinds, prob.uinds
jacobian_linear_inds(prob::DIRCOLSolver) = prob.linds

@inline copy_gradient!(grad_f, solver::DIRCOLSolver) = copy_gradient!(grad_f, solver.E, solver.xinds, solver.uinds)
@inline Base.copyto!(d::DIRCOLSolver, V::Vector{<:Real}) = copyto!(d.Z, V, primal_partition(d)...)

function initial_controls!(d::DIRCOLSolver, U0)
    Z = get_trajectory(d)
    set_controls!(Z, U0)
    V = [MOI.VariableIndex(i) for i = 1:d.NN]
    V0 = zeros(d.NN)
    copyto!(V0, Z, d.xinds, d.uinds)
    MOI.set(d.optimizer, MOI.VariablePrimalStart(), V, V0)
end

function initial_states!(d::DIRCOLSolver, X0)
    Z = get_trajectory(d)
    set_states!(Z, X0)
    V = [MOI.VariableIndex(i) for i = 1:d.NN]
    V0 = zeros(d.NN)
    copyto!(V0, Z, d.xinds, d.uinds)
    MOI.set(d.optimizer, MOI.VariablePrimalStart(), V, V0)
end

function get_rc(A::SparseMatrixCSC)
    row,col,inds = findnz(A)
    v = sortperm(inds)
    row[v],col[v]
end

function cost(solver::DIRCOLSolver{Q}) where Q<:QuadratureRule
	Z = get_trajectory(solver)
	obj = get_objective(solver)
	cost(obj, solver.dyn_con, Z)
end

function cost_gradient!(solver::DIRCOLSolver{Q}) where Q
	obj = get_objective(solver)
	Z = get_trajectory(solver)
	dyn_con = solver.dyn_con
	E = solver.E
	cost_gradient!(E, obj, dyn_con, Z)
end


# Define MOI Interface
MOI.features_available(d::DIRCOLSolver) = [:Grad, :Jac]
MOI.initialize(d::DIRCOLSolver, features) = nothing

function MOI.jacobian_structure(d::DIRCOLSolver)
    jac_struct = constraint_jacobian_structure(d, d.jacobian_structure)
    r,c = get_rc(jac_struct)
    jac_struct = collect(zip(r,c))
end

MOI.hessian_lagrangian_structure(d::DIRCOLSolver) = []

function MOI.eval_objective(d::DIRCOLSolver, V)
    copyto!(d, V)
    cost(d)
end

function MOI.eval_objective_gradient(d::DIRCOLSolver, grad_f, V)
    copyto!(d, V)
    cost_gradient!(d)
    copy_gradient!(grad_f, d)
    return nothing
end

function MOI.eval_constraint(d::DIRCOLSolver, g, V)
    copyto!(d, V)
    update_constraints!(d)
    copy_constraints!(g, d)
end

function MOI.eval_constraint_jacobian(d::DIRCOLSolver, jac, V)
    copyto!(d, V)
    constraint_jacobian!(d)
    copy_jacobians!(jac, d)
end

MOI.eval_hessian_lagrangian(::DIRCOLSolver, H, x, σ, μ) = nothing

function solve!(d::DIRCOLSolver)
    # Update options
    nlp_opts = nlp_options(d.optimizer, d.opts)
    for (key,val) in nlp_opts
        d.optimizer.options[String(key)] = val
    end

    # Solve with MOI
    MOI.optimize!(d.optimizer)

    # Get result and copy to trajectory
    V = [MOI.VariableIndex(k) for k = 1:d.NN]
    res = MOI.get(d.optimizer, MOI.VariablePrimal(), V)
    copyto!(d, res)

    # Copy stats
    for (key,val) in parse_ipopt_summary()
        d.stats[key] = val
    end
    return nothing
end

function nlp_options(nlp::MOI.AbstractOptimizer, opts::SolverOptions)
    solver_name = optimizer_name(nlp)
	opts_optimizer = Dict{Symbol,Any}()
    if solver_name == :Ipopt
        !opts.verbose ? opts_optimizer[:print_level] = 0 : opts_optimizer[:print_level] = 5
        if opts.constraint_tolerance > 0.
            opts_optimizer[:constr_viol_tol] = opts.constraint_tolerance
            opts_optimizer[:tol] = opts.constraint_tolerance
        end
    elseif solver_name == :SNOPT7
        if !opts.verbose
            opts_optimizer[:Major_print_level] = 0
            opts_optimizer[:Minor_print_level] = 0
        end
        if opts.constraint_tolerance > 0.
            opts_optimizer[:Major_constraint_tolerance] = opts.constraint_tolerance
            opts_optimizer[:Minor_constraint_tolerance] = opts.constraint_tolerance
            opts_optimizer[:Major_optimality_tolerance] = opts.constraint_tolerance
        end
    else
        error("Nonlinear solver not implemented")
    end

    return opts_optimizer
end

function optimizer_name(optimizer::MathOptInterface.AbstractOptimizer)
    nameof(parentmodule(typeof(optimizer)))
end
