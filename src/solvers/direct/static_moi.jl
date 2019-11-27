const MOI = MathOptInterface

struct StaticDIRCOLProblem{T,Q<:QuadratureRule,N,M,L,O,L1,L2,L3}
    prob::StaticProblem{L,T,O,N,M,NM}
    solver::StaticDIRCOLSolver
    jac_struct::Vector{NTuple{2,Int}}
    zL::Vector{T}
    zU::Vector{T}
    gL::Vector{T}
    gU::Vector{T}
end

struct StaticDIRCOLSolver{T,N,M,NM,NNM,L1,L2,L3} <: DirectSolver{T}
    Xm::Vector{SVector{N,T}}
    fVal::Vector{SVector{N,T}}
    ∇F::Vector{SMatrix{N,NM,T,NNM}}
    E::CostExpansion{T,N,M,L1,L2,L3}

    xinds::Vector{SVector{N,Int}}
    uinds::Vector{SVector{M,Int}}

    dyn_inds::Vector{SVector{N,Int}}
    con_inds::Vector{Vector{SV} where SV}
end


@inline Base.copyto!(d::StaticDIRCOLProblem, V::Vector{<:Real}) = copyto!(d.prob.Z, V, d.xinds, x.uinds)

MOI.features_available(d::StaticDIRCOLProblem) = [:Grad, :Jac]
MOI.initialize(d::StaticDIRCOLProblem, features) = nothing

MOI.jacobian_structure(d::StaticDIRCOLProblem) = d.jac_struct
MOI.hessian_lagrangian_structure(d::StaticDIRCOLProblem) = []

function MOI.eval_objective(d::StaticDIRCOLProblem, V)
    copyto!(d, V)
    cost(d.prob)
end

function MOI.eval_objective_gradient(d::StaticDIRCOLProblem, grad_f, V)
    copyto!(d, V)
    E = d.E
    xinds, uinds = d.solver.xinds, d.solver.uinds
    N = d.prob.N

    cost_gradient!(E, d.prob.obj, d.prob.Z)
    for k = 1:N-1
        g[xinds[k]] .= E.x[k]
        g[uinds[k]] .= E.u[k]
    end
    g[xinds[N]] .= E.x[N]
    return nothing
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
