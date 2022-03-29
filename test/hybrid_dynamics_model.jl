using TrajectoryOptimization
using Test
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using ForwardDiff
using RobotDynamics
using BenchmarkTools
using FiniteDiff
const TO = TrajectoryOptimization
const RD = RobotDynamics

RD.@autodiff struct Model1 <: RD.ContinuousDynamics end
RD.@autodiff struct JumpMap <: RD.ContinuousDynamics end 
RD.@autodiff struct Model2 <: RD.ContinuousDynamics end

RD.state_dim(::Model1) = 4
RD.control_dim(::Model1) = 2
RD.state_dim(::JumpMap) = 4
RD.control_dim(::JumpMap) = 2
RD.output_dim(::JumpMap) = 2
RD.state_dim(::Model2) = 2
RD.control_dim(::Model2) = 1

function RD.dynamics(::Model1, x, u)
    return SA[x[3], x[4], u[1], u[2]]
end

RD.dynamics!(model::Model1, y, x, u) = y .= RD.dynamics(model, x, u)

function RD.dynamics(::JumpMap, x, u)
    return SA[(x[3] + x[4])/2, (u[1] + u[2])/2]
end

function RD.dynamics(::Model2, x, u)
    return SA[x[2], u[1]]
end

model1 = RD.DiscretizedDynamics{RD.RK4}(Model1())
jumpmap = RD.DiscretizedDynamics{RD.RK4}(JumpMap())
model2 = RD.DiscretizedDynamics{RD.RK4}(Model2())

# Get state and control dimension vectors
models = [
    [copy(model1) for k = 1:5];
    jumpmap
    [copy(model2) for k = 1:4];
]
@test eltype(models) isa UnionAll 
models2 = [copy(model1) for k = 1:10]
nx, nu = RD.dims(models)
@test nx == [4,4,4,4,4, 4, 2,2,2,2,2]
@test nu == [2,2,2,2,2, 2, 1,1,1,1,1]

x,u = rand(models[1])
xn = zeros(4)
RD.discrete_dynamics!(models[1], xn, x, u, 0.0, 0.1)

# Test bad model vector (no jump map)
models_bad = [
    [copy(model1) for k = 1:5];
    [copy(model2) for k = 1:5];
]
@test_throws DimensionMismatch RD.dims(models_bad)

# Try building a problem
costfuns = map(1:11) do k
    LQRCost(Diagonal(fill(1.0, nx[k])), Diagonal(fill(0.1, nu[k])), zeros(nx[k]))
end
obj = Objective(costfuns)

x0 = zeros(4)
tf = 2.0

# Build unconstrained problem
prob = Problem(models, obj, x0, tf)
Z = prob.Z
@test Z isa SampledTrajectory{Any,Any}
@test RD.dims(Z[5]) == (4,2)
@test RD.dims(Z[6]) == (4,2)
@test RD.dims(Z[end]) == (2,1)

# Build constrained problem
bnd1 = BoundConstraint(4, 2, u_max=4, u_min=-4)
bnd2 = BoundConstraint(2, 1, u_max=2, u_min=-2, x_max=[10,Inf]) 
goal = GoalConstraint(randn(2))
cons = ConstraintList(models)
add_constraint!(cons, bnd1, 1:5)
add_constraint!(cons, bnd2, 7:10)
add_constraint!(cons, goal, 11)

prob = Problem(models, obj, x0, tf, constraints=cons)
@test TO.num_constraints(prob) == [4,4,4,4,4, 0, 3,3,3,3,2]

# Try adding incompatible constraint
@test_throws DimensionMismatch add_constraint!(cons, bnd1, 3:8)
@test_throws DimensionMismatch add_constraint!(cons, bnd2, 1:3)

# Test passing bad inputs
@test_throws DimensionMismatch Problem(models_bad, obj, x0, tf)

obj_bad = LQRObjective(Diagonal(ones(4)), Diagonal(ones(2)), Diagonal(ones(4)), zeros(4), 11)
@test_throws DimensionMismatch Problem(models, obj_bad, x0, tf)

cons_bad = ConstraintList(4,2,11)
add_constraint!(cons_bad, bnd1, 1:5)
@test_throws DimensionMismatch Problem(models, obj, x0, tf, constraints=cons_bad)

# Test bad control dimension
nu_bad = copy(nu)
nu_bad[7] = 2
cons_bad2 = ConstraintList(nx, nu_bad)
add_constraint!(cons_bad2, bnd1, 1:5)
@test_throws DimensionMismatch Problem(models, obj, x0, tf, constraints=cons_bad2)

obj_bad2 = Objective([LQRCost(Diagonal(ones(n)), Diagonal(ones(m)), zeros(n)) for (n,m) in zip(nx, nu_bad)])
@test_throws DimensionMismatch Problem(models, obj_bad2, x0, tf, constraints=cons)