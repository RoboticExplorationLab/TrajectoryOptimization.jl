using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using Statistics
using BenchmarkTools
using ForwardDiff
using Test
using SparseArrays
const TO = TrajectoryOptimization

function make_mutable(conSet::ConstraintSet)
    n,m,P = size(conSet)
    N =  length(conSet.p)
    conSet_ = ConstraintSet(n,m,N)
    for con in conSet
        add_constraint!(conSet_, con.con, con.inds, mutable=true)
    end
    return conSet_
end
function make_mutable(prob::Problem)
    return Problem(prob, constraints = make_mutable(prob.constraints))
end

# Test entire solve
prob,opts = Problems.Quadrotor()
solver0 = iLQRSolver(copy(prob), opts)
solver1 = iLQRSolver2(copy(make_mutable(prob)), opts)
solver0 = ALTROSolver(copy(prob), opts) #, infeasible=true, R_inf=0.1)
solver1 = ALTROSolver(copy(make_mutable(prob)), opts, solver_uncon=iLQRSolver2) #, infeasible=true, R_inf=0.1)
# solve!(solver0)
# solve!(solver1)
solver0.opts.projected_newton = false
solver1.opts.projected_newton = false
b1 = benchmark_solve!(solver0)
b2 = benchmark_solve!(solver1)
judge(median(b2),median(b1))
@show median(b2).allocs
iterations(solver0)
iterations(solver1)
@show cost(solver0) ≈ cost(solver1)

model = get_model(solver1)
n,m = size(model)
x,u = rand(model)
z = KnotPoint(x,u,0.1)
∇f = zeros(n,n+m+1)
@btime TO.discrete_jacobian!($RK3, $∇f, $model, $z)
∇f = SizedMatrix{n,n+m+1}(∇f)
@btime TO.discrete_jacobian!($RK3, $∇f, $model, $z)

altro0 = ALTROSolver(copy(prob), opts)#, infeasible=true, R_inf=0.1)
altro1 = ALTROSolver(copy(make_mutable(prob)), opts, solver_uncon=iLQRSolver2)#, infeasible=true, R_inf=0.1)
al0 = altro0.solver_al
al1 = altro1.solver_al
initialize!(al0)
initialize!(al1)
solver0 = al0.solver_uncon
solver1 = al1.solver_uncon
solve!(solver0)
solve!(solver1)
iterations(solver0)
iterations(solver1)

# Step through
altro0 = ALTROSolver(copy(prob), opts)#, infeasible=true, R_inf=0.1)
altro1 = ALTROSolver(copy(make_mutable(prob)), opts, solver_uncon=iLQRSolver2)#, infeasible=true, R_inf=0.1)
al0 = altro0.solver_al
al1 = altro1.solver_al
solver0 = al0.solver_uncon
solver1 = al1.solver_uncon
rollout!(solver0)
rollout!(solver1)
states(solver0) ≈ states(solver1)
cost(solver0) ≈ cost(solver1)
TO.state_diff_jacobian!(solver0.G, solver0.model, solver0.Z)
TO.state_diff_jacobian!(solver1.G, solver1.model, solver1.Z)
solver0.G ≈ solver1.G
TO.cost_expansion!(solver0.Q, solver0.G, solver0.obj, solver0.model, solver0.Z)
TO.cost_expansion!(solver1.Q, solver1.obj, solver1.Z)
TO.error_expansion!(solver1.Q, solver1.model, solver1.Z, solver1.G)

Qxx = [Q.xx_ for Q in solver1.Q]
Quu = [Q.uu_ for Q in solver1.Q]
Qux = [Q.ux_ for Q in solver1.Q]
Qx = [Q.x_ for Q in solver1.Q]
Qu = [Q.u_ for Q in solver1.Q]
Qxx ≈ solver0.Q.xx
Quu ≈ solver0.Q.uu
Qux ≈ solver0.Q.ux
Qx ≈ solver0.Q.x
Qu ≈ solver0.Q.u
TO.discrete_jacobian!(solver0.∇F, solver0.model, solver0.Z)
TO.dynamics_expansion!(solver1.D, solver1.G, solver1.model, solver1.Z)
solver1.D[1].A_
∇F = [D.∇f for D in solver1.D]
∇F ≈ solver0.∇F

# step through backward pass
ΔV0 = backwardpass!(solver0)
ΔV1 = backwardpass!(solver1)
ΔV2 = TO.static_backwardpass!(solver1)
ΔV0 ≈ ΔV1 ≈ ΔV2
solver0.K ≈ solver1.K
solver0.d ≈ solver1.d

J0 = TO.forwardpass!(solver0, ΔV0, cost(solver0))
J1 = forwardpass!(solver1, ΔV1, cost(solver1))
J0 ≈ J1
states(solver0.Z̄) ≈ states(solver1.Z̄)

TO.copy_trajectories!(solver0)
TO.copy_trajectories!(solver1)


# Compare timing
al0 = AugmentedLagrangianSolver(copy(prob), opts)
al1 = AugmentedLagrangianSolver(copy(make_mutable(prob)), opts, solver_uncon=iLQRSolver2)
solver0 = al0.solver_uncon
solver1 = al1.solver_uncon
@btime rollout!($solver0)
@btime rollout!($solver1)
@btime TO.cost!($solver0.obj, $solver0.Z)
@btime TO.cost!($solver1.obj, $solver1.Z)
@btime TO.state_diff_jacobian!($solver0.G, $solver0.model, $solver0.Z)
@btime TO.state_diff_jacobian!($solver1.G, $solver1.model, $solver1.Z)
@btime TO.cost_expansion!($solver0.Q, $solver0.G, $solver0.obj, $solver0.model, $solver0.Z)
@btime TO.cost_expansion!($solver1.Q, $solver1.obj, $solver1.Z)
@btime TO.error_expansion!($solver1.Q, $solver1.model, $solver1.Z, $solver1.G)
@btime TO.discrete_jacobian!($solver0.∇F, $solver0.model, $solver0.Z)
@btime TO.dynamics_expansion!($solver1.D, $solver1.model, $solver1.Z)
@btime TO.error_expansion!($solver1.D, $solver1.model, $solver1.G)

@btime ΔV0 = backwardpass!($solver0)
@btime ΔV1 = backwardpass!($solver1)
@btime ΔV1 = TO.static_backwardpass!($solver1)
Q = solver1.Q[1]
z = solver1.Z[1]
G = solver1.G[1]
@btime TO.error_expansion!($Q, $solver1.model, $z, SMatrix($G))
@btime TO.error_expansion($Q)

@btime TO.forwardpass!($solver0, $ΔV0, cost($solver0))
@btime TO.forwardpass!($solver1, $ΔV1, cost($solver1))
@btime rollout!($solver0, 1.0)
@btime rollout!($solver1, 1.0)

E1 = TO.GeneralExpansion{Float64}(SizedArray, 4,4,1)
E2 = TO.GeneralExpansion{Float64}(SizedArray, 4,4,1)
E1 = TO.SizedExpansion{Float64}(4,4,1)
E2 = TO.SizedExpansion{Float64}(4,4,1)

@btime TO._error_expansion!($E1,$E2,$G)
@btime TO.error_expansion!($solver.E, $solver.Q[1], $solver.model, $solver.Z[1], $(solver.G[1]))


struct MyType{T}
