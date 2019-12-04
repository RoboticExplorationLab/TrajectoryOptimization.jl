const MOI = MathOptInterface

struct StaticDIRCOLSolver{T,N,M,NM,NNM,L1,L2,L3} <: DirectSolver{T}
    NN::Int
    NP::Int

    Xm::Vector{SVector{N,T}}
    fVal::Vector{SVector{N,T}}
    ∇F::Vector{SMatrix{N,NM,T,NNM}}
    E::CostExpansion{T,N,M,L1,L2,L3}

    xinds::Vector{SVector{N,Int}}
    uinds::Vector{SVector{M,Int}}

    linds::Vector{<:Vector{SV} where SV}
    con_inds::Vector{<:Vector{SV} where SV}
end

function StaticDIRCOLSolver(prob::StaticProblem)
    n,m,N = size(prob)
    Xm = [@SVector zeros(n) for k = 1:N-1]
    fVal = [@SVector zeros(n) for k = 1:N]
    ∇F = [@SMatrix zeros(n,n+m) for k = 1:N]
    E = CostExpansion(n,m,N)

    NN = (n+m)*N
    NP = sum(num_constraints(prob))

    P = StaticPrimals(n,m,N)
    xinds, uinds = P.xinds, P.uinds

    blk_len = map(con->length(con.∇c[1]), conSet.constraints)
    con_len = map(con->length(con.∇c), conSet.constraints)
    linds = [[@SVector zeros(Int,blk) for i = 1:len] for (blk,len) in zip(blk_len, con_len)]

    con_inds = gen_con_inds(get_constraints(prob))
    StaticDIRCOLSolver(NN, NP, Xm, fVal, ∇F, E, xinds, uinds, linds, con_inds)
end


num_primals(solver::StaticDIRCOLSolver) = solver.NN
num_duals(solver::StaticDIRCOLSolver) =  solver.NP

primal_partition(prob::StaticDIRCOLSolver) = prob.xinds, prob.uinds
jacobian_linear_inds(prob::StaticDIRCOLSolver) = prob.linds

struct StaticDIRCOLProblem{T,Q<:QuadratureRule} <: MOI.AbstractNLPEvaluator
    prob::StaticProblem
    solver::StaticDIRCOLSolver
    jac_struct::Vector{NTuple{2,Int}}
    zL::Vector{T}
    zU::Vector{T}
    gL::Vector{T}
    gU::Vector{T}
    function StaticDIRCOLProblem(prob::StaticProblem{L,T}) where {L,T}
        conSet = get_constraints(prob)
        @assert has_dynamics(conSet)
        dyn_con = filter(x->x.con isa ExplicitDynamics, conSet.constraints)[1]
        Q = quadrature_rule(dyn_con.con)

        # Remove bounds
        zU,zL,gU,gL = get_bounds(conSet)

        # Create solver
        solver = StaticDIRCOLSolver(prob)
        jac_structure = constraint_jacobian_structure(prob, solver)
        r,c = get_rc(jac_structure)
        jac_struct = collect(zip(r,c))

        new{T,Q}(prob, solver, jac_struct, zL, zU, gL, gU)
    end
end



@inline Base.copyto!(d::StaticDIRCOLProblem, V::Vector{<:Real}) = copyto!(d.prob.Z, V, primal_partition(d.solver)...)
MOI.features_available(d::StaticDIRCOLProblem) = [:Grad, :Jac]
MOI.initialize(d::StaticDIRCOLProblem, features) = nothing

MOI.jacobian_structure(d::StaticDIRCOLProblem) = d.jac_struct
MOI.hessian_lagrangian_structure(d::StaticDIRCOLProblem) = []

function MOI.eval_objective(d::StaticDIRCOLProblem, V)
    copyto!(d, V)
    cost(d.prob, d.solver)
end

function MOI.eval_objective_gradient(d::StaticDIRCOLProblem, grad_f, V)
    copyto!(d, V)
    E = d.solver.E
    xinds, uinds = d.solver.xinds, d.solver.uinds

    cost_gradient!(d.prob, d.solver)
    copy_gradient!(grad_f, E, xinds, uinds)

    return nothing
end

function copy_gradient!(grad_f, E::CostExpansion, xinds, uinds)
    for k = 1:length(uinds)
        grad_f[xinds[k]] = E.x[k]
        grad_f[uinds[k]] = E.u[k]
    end
    if length(xinds) != length(uinds)
        grad_f[xinds[end]] = E.x[end]
    end
end

function MOI.eval_constraint(d::StaticDIRCOLProblem, g, V)
    copyto!(d, V)
    update_constraints!(d.prob, d.solver)
    copy_constraints!(d.prob, d.solver, g)
end

function MOI.eval_constraint_jacobian(d::StaticDIRCOLProblem, jac, V)
    copyto!(d, V)
    constraint_jacobian!(d.prob, d.solver)
    copy_jacobians!(d.prob, d.solver, jac)
end

MOI.eval_hessian_lagrangian(::StaticDIRCOLProblem, H, x, σ, μ) = nothing

function solve_moi!(prob::StaticProblem, d::StaticDIRCOLProblem)
    n,m,N = size(prob)
    NN = num_primals(d.solver)
    NP = num_duals(d.solver)

    V0 = zeros(NN)
    xinds, uinds = primal_partition(d.solver)
    copyto!(V0, prob.Z, xinds, uinds)

    v_L, v_U = d.zL, d.zU
    g_L, g_U = d.gL, d.gU

    has_objective = true
    nlp_bounds = MOI.NLPBoundsPair.(g_L, g_U)
    block_data = MOI.NLPBlockData(nlp_bounds, d, has_objective)

    solver = Ipopt.Optimizer()
    V = MOI.add_variables(solver, NN)

    # Add bound constraints
    MOI.add_constraints(solver, V, MOI.LessThan.(v_U))
    MOI.add_constraints(solver, V, MOI.GreaterThan.(v_L))

    # Initial Condition
    MOI.set(solver, MOI.VariablePrimalStart(), V, V0)

    # Set up NLP problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Solve
    MOI.optimize!(solver)

    # Get solution
    res = MOI.get(solver, MOI.VariablePrimal(), Z)
    copyto!(prob.Z, res, xinds, uinds)

    status = MOI.get(solver, MOI.TerminationStatus())
end

function build_moi_problem(d::StaticDIRCOLProblem)
    NN = length(d.zL)
    V0 = zeros(NN)
    xinds, uinds = primal_partition(d.solver)
    copyto!(V0, d.prob.Z, xinds, uinds)

    has_objective = true
    nlp_bounds = MOI.NLPBoundsPair.(d.gL, d.gU)
    block_data = MOI.NLPBlockData(nlp_bounds, d, has_objective)

    solver = Ipopt.Optimizer(print_level=0)
    V = MOI.add_variables(solver, NN)

    MOI.add_constraints(solver, V, MOI.LessThan.(d.zU))
    MOI.add_constraints(solver, V, MOI.GreaterThan.(d.zL))

    MOI.set(solver, MOI.VariablePrimalStart(), V, V0)

    # Set up NLP problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return solver
end
