const MOI = MathOptInterface


struct StaticDIRCOLSolver{Q<:QuadratureRule,L,T,N,M,NM} <: DirectSolver{T}
    opts::DIRCOLSolverOptions
    stats::Dict{Symbol,Any}

    NN::Int
    NP::Int

    dyn_con::DynamicsConstraint{Q,L,T,N,NM}
    objective::AbstractObjective
    constraints::ConstraintSets{T}
    constraints_all::ConstraintSets{T}
    Z::Vector{KnotPoint{T,N,M,NM}}
    x0::SVector{N,T}

    optimizer::MOI.AbstractOptimizer

    E::CostExpansion

    xinds::Vector{SVector{N,Int}}
    uinds::Vector{SVector{M,Int}}

    linds::Vector{<:Vector{SV} where SV}
    con_inds::Vector{<:Vector{SV} where SV}

    function StaticDIRCOLSolver(opts,stats, NN,NP,dyn_con::DynamicsConstraint{Q,L,T,N},
            obj,conSet,conSet_all,Z::Vector{KnotPoint{T,N,M,NM}},
            x0, optimizer, E,
            xinds,uinds,linds,con_inds) where {Q,L,T,N,M,NM}
        new{Q,L,T,N,M,NM}(opts,stats, NN,NP,dyn_con,obj,conSet,conSet_all,Z,x0,optimizer,E,
            xinds,uinds,linds,con_inds)
    end

end

Base.size(solver::StaticDIRCOLSolver{Q,L,T,n,m,NM}) where {Q,L,T,n,m,NM} = n,m,length(solver.Z)

function StaticDIRCOLSolver(prob::StaticProblem{Q}, opts::DIRCOLSolverOptions=DIRCOLSolverOptions()) where Q
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
        bnd = StaticBoundConstraint(n,m)
        add_constraint!(conSet, bnd)
    end

    # Remove bounds
    zU,zL,gU,gL = get_bounds(conSet)

    # Initialize arrays
    dyn_vals = DynamicsVals(dyn_con)
    E = CostExpansion(n,m,N)

    NN = (n+m)*N
    NP = sum(num_constraints(prob))

    P = StaticPrimals(n,m,N)
    xinds, uinds = P.xinds, P.uinds

    blk_len = map(con->length(con.∇c[1]), conSet.constraints)
    con_len = map(con->length(con.∇c), conSet.constraints)
    linds = [[@SVector zeros(Int,blk) for i = 1:len] for (blk,len) in zip(blk_len, con_len)]
    con_inds = gen_con_inds(get_constraints(prob))


    # Create MOI Optimizer
    nlp_opts = Dict(Symbol(key)=>value for (key,val) in pairs(opts.nlp.options))
    optimizer = typeof(opts.nlp)(;nlp_opts..., nlp_options(opts)...)

    # Create Solver
    d = StaticDIRCOLSolver(opts, stats, NN, NP, dyn_con, prob.obj, conSet, conSet_all,
        Z, prob.x0, optimizer, E, xinds, uinds, linds, con_inds)

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

get_initial_state(solver::StaticDIRCOLSolver) = solver.x0
get_model(solver::StaticDIRCOLSolver) = solver.dyn_con.model
get_constraints(solver::StaticDIRCOLSolver) = solver.constraints
get_trajectory(solver::StaticDIRCOLSolver) = solver.Z
get_objective(solver::StaticDIRCOLSolver) = solver.objective
num_primals(solver::StaticDIRCOLSolver) = solver.NN
num_duals(solver::StaticDIRCOLSolver) =  solver.NP

"Include bounds when calculating max violation on the solver"
function max_violation(solver::StaticDIRCOLSolver)
    conSet = solver.constraints_all
    max_violation!(conSet)
    return maximum(conSet.c_max)
end

primal_partition(prob::StaticDIRCOLSolver) = prob.xinds, prob.uinds
jacobian_linear_inds(prob::StaticDIRCOLSolver) = prob.linds

@inline copy_gradient!(grad_f, solver::StaticDIRCOLSolver) = copy_gradient!(grad_f, solver.E, solver.xinds, solver.uinds)
@inline Base.copyto!(d::StaticDIRCOLSolver, V::Vector{<:Real}) = copyto!(d.Z, V, primal_partition(d)...)

function initial_controls!(d::StaticDIRCOLSolver, U0)
    Z = get_trajectory(d)
    set_controls!(Z, U0)
    V = [MOI.VariableIndex(i) for i = 1:d.NN]
    V0 = zeros(d.NN)
    copyto!(V0, Z, d.xinds, d.uinds)
    MOI.set(d.optimizer, MOI.VariablePrimalStart(), V, V0)
end

function initial_states!(d::StaticDIRCOLSolver, X0)
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


# Define MOI Interface
MOI.features_available(d::StaticDIRCOLSolver) = [:Grad, :Jac]
MOI.initialize(d::StaticDIRCOLSolver, features) = nothing

function MOI.jacobian_structure(d::StaticDIRCOLSolver)
    jac_struct = constraint_jacobian_structure(d)
    r,c = get_rc(jac_struct)
    jac_struct = collect(zip(r,c))
end

MOI.hessian_lagrangian_structure(d::StaticDIRCOLSolver) = []

function MOI.eval_objective(d::StaticDIRCOLSolver, V)
    copyto!(d, V)
    cost(d)
end

function MOI.eval_objective_gradient(d::StaticDIRCOLSolver, grad_f, V)
    copyto!(d, V)
    cost_gradient!(d)
    copy_gradient!(grad_f, d)
    return nothing
end

function MOI.eval_constraint(d::StaticDIRCOLSolver, g, V)
    copyto!(d, V)
    update_constraints!(d)
    copy_constraints!(g, d)
end

function MOI.eval_constraint_jacobian(d::StaticDIRCOLSolver, jac, V)
    copyto!(d, V)
    constraint_jacobian!(d)
    copy_jacobians!(jac, d)
end

MOI.eval_hessian_lagrangian(::StaticDIRCOLSolver, H, x, σ, μ) = nothing

function solve!(d::StaticDIRCOLSolver)
    # Update options
    nlp_opts = nlp_options(d.opts)
    for (key,val) in nlp_opts
        d.optimizer.options[String(key)] = val
    end

    # Solve with MOI
    MOI.optimize!(d.optimizer)

    # Get result and copy to trajectory
    V = [MOI.VariableIndex(k) for k = 1:ds.NN]
    res = MOI.get(d.optimizer, MOI.VariablePrimal(), V)
    copyto!(d, res)

    # Copy stats
    for (key,val) in parse_ipopt_summary()
        d.stats[key] = val
    end
    return nothing
end
