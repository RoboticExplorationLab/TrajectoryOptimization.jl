using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using TrajOptPlots
using GeometryTypes
using MeshCat
const TO = TrajectoryOptimization
import TrajectoryOptimization.Controllers: RBState

# Model
Rot = UnitQuaternion{Float64,CayleyMap}
model = Dynamics.FreeBody{Rot,Float64}()
n,m = size(model)

if !isdefined(Main,:vis)
    vis = Visualizer()
    open(vis)
    set_mesh!(vis, model)
end

# Discretization
N = 101
tf = 5.0

# Initial and final conditions
X0 = zero(RBState)
Xf = RBState([1,1,1.], expm(deg2rad(70) * @SVector rand(3)), zeros(3), zeros(3))
x0 = Dynamics.build_state(model, X0)
xf = Dynamics.build_state(model, Xf)

# Objective
Qd = Dynamics.fill_state(model, 0e-0, 0e-1, 1e-2, 1e-2)
Rd = @SVector fill(1e-2, 6)
costfun = QuatLQRCost(Diagonal(Qd), Diagonal(Rd), xf, w=0.)
obj = Objective(costfun, N)

# Constraint
struct PointingConstraint <: TO.AbstractConstraint{Equality,State,1}
    p::SVector{3,Float64}
    b::SVector{3,Float64}
    PointingConstraint(p::SVector{3}, b=@SVector [0,0,1.]) = new(p,b)
end
TO.state_dim(::PointingConstraint) = 13

function TO.evaluate(con::PointingConstraint, x::SVector)
    q = UnitQuaternion(x[4],x[5],x[6],x[7])
    x3 = q*con.b
    r = @SVector [x[1],x[2],x[3]]
    dx = con.p - r
    return @SVector [x3'normalize(dx) - 1]
end

point = @SVector [1,-.5,1]
add_point!(vis, point)
con = PointingConstraint(point)
point2 = @SVector [-1,1,0.]
add_point!(vis, point2, name="point2")
con2 = PointingConstraint(point2, @SVector [1,0,0])

conSet = ConstraintSet(n,m,N)
add_constraint!(conSet, con, N:N)
add_constraint!(conSet, con2, N:N)

# Initialization
U0 = [@SVector zeros(6) for k = 1:N-1]

# Problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)

# Solve
solver = iLQRSolver(prob)
rollout!(solver)
cost(solver)
solver.opts.verbose = true
solve!(solver)


solver = AugmentedLagrangianSolver(prob)
initial_controls!(solver, U0)
rollout!(solver)
solver.opts.verbose = true
TO.update_constraints!(solver)
TO.constraint_jacobian!(solver)
max_violation(solver)
get_constraints(solver)
cost(solver)
evaluate(con, states(solver)[end])

solve!(solver)
err = Xf ⊖ RBState(model, states(solver)[end])
norm(err)

visualize!(vis, solver)

qf = Xf.q
e3 = @SVector [0,0,1.]
qf*e3




rotmat(qf)
