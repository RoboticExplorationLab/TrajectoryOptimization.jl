using TrajectoryOptimization
using BenchmarkTools
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Test
using TrajOptPlots
using MeshCat
const TO = TrajectoryOptimization


model = Dynamics.DubinsCar()
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

models = Dynamics.CopyModel(model, 2)
size(models) == (6,4)
n,m = size(models)
zeros(models)

x,u = rand(models)
x1,u1 = x[1:3], u[1:2]
x2,u2 = x[4:6], u[3:4]
@test !(states(models, x, 1) ≈ states(models, x, 2))
@test !(states(models, x, 1) ≈ states(models, x, 2))

@test dynamics(models, x, u) == [dynamics(model, x1,u1); dynamics(model,x2,u2)]
z = KnotPoint(x,u,0.1)
z1 = KnotPoint(x1,u1,0.1)
z2 = KnotPoint(x2,u2,0.1)
@test discrete_dynamics(RK3, models, z) ==
    [discrete_dynamics(model, z1);discrete_dynamics(model, z2)]

function my_jacobian(z::SVector)
    n,m = size(models)
    ix,iu = 1:n, n .+ (1:m)
    f_aug(z) = dynamics(models, view(z,ix), view(z,iu))
    ForwardDiff.jacobian(f_aug, z)
end
@test jacobian(models, z) ≈ my_jacobian(z.z)

function my_discrete_jacobian(s::SVector{NM1}) where NM1
    t = 0.0
    ix = @SVector [1,2,3,4,5,6]
    iu = @SVector [7,8,9,10]
    idt = NM1
    fd_aug(s) = discrete_dynamics(RK3, models, s[ix], s[iu], t, s[idt])
    ForwardDiff.jacobian(fd_aug, s)
end
s = [z.z; @SVector [z.dt]]
@test my_discrete_jacobian(s) ≈ discrete_jacobian(RK3, models, z)

xf = rand(models)[1]
@test TO.state_diff(models, x, xf) == x - xf
@test TO.state_diff_jacobian(models, x) == I
@test TO.∇²differential(models, x, @SVector rand(6)) == I*0


# Test solve
N = 101
tf = 5.0

x0_1 = @SVector [ 1,0,pi/2]
x0_2 = @SVector [-1,0,pi/2]
x0 = Dynamics.build_state(models, x0_1, x0_2)

xf_1 = @SVector [-1,2,pi/2]
xf_2 = @SVector [ 1,2,pi/2]
xf = Dynamics.build_state(models, xf_1, xf_2)

Q = Diagonal(@SVector fill(1e-2,n))
Rd = @SVector [0.1, 1e-2]
R = Diagonal( [Rd; Rd] )
obj = LQRObjective(Q,R,Q*100,xf,N)

conSet = ConstraintSet(n,m,N)
con = TO.CollisionConstraint(n, (@SVector [1,2]), (@SVector [4,5]), 0.5)
add_constraint!(conSet, con, 1:N-1)

prob = Problem(models, obj, xf, tf, x0=x0, constraints=conSet)
# solver = iLQRSolver(prob)
solver = AugmentedLagrangianSolver(prob)
@test size(solver) == (6,4,N)
solve!(solver)
@test iterations(solver) < 20  # should be 16
@test max_violation(solver) < 1e-10 # should be 0

prob = Problem(models, obj, xf, tf, x0=x0, constraints=conSet)
solver = iLQRSolver2(prob)
solve!(solver)
visualize!(vis, solver)

# Test with rigid bodies
Rot = UnitQuaternion{Float64,CayleyMap}
model = Dynamics.FreeBody{Rot,Float64}()
models = Dynamics.CopyModel(model, 2)

x,u = zeros(models)
x = repeat(zeros(model)[1],2)
u = repeat(zeros(model)[2],2)
x,u = rand(models)
x1,u1 = SVector{13}(x[1:13]), SVector{6}(u[1:6])
x2,u2 = SVector{13}(x[14:26]), SVector{6}(u[7:12])
@test !(states(models, x, 1) ≈ states(models, x, 2))
@test !(states(models, x, 1) ≈ states(models, x, 2))

@test dynamics(models, x, u) == [dynamics(model, x1,u1); dynamics(model,x2,u2)]
z = KnotPoint(x,u,0.1)
z1 = KnotPoint(x1,u1,0.1)
z2 = KnotPoint(x2,u2,0.1)
@test discrete_dynamics(RK3, models, z) ==
    [discrete_dynamics(model, z1);discrete_dynamics(model, z2)]

function my_jacobian(z::SVector)
    n,m = size(models)
    ix,iu = 1:n, n .+ (1:m)
    f_aug(z) = dynamics(models, view(z,ix), view(z,iu))
    ForwardDiff.jacobian(f_aug, z)
end
@test jacobian(models, z) ≈ my_jacobian(z.z)

function my_discrete_jacobian(s::SVector{NM1}) where NM1
    t = 0.0
    ix = z._x
    iu = z._u
    idt = NM1
    fd_aug(s) = discrete_dynamics(RK3, models, s[ix], s[iu], t, s[idt])
    ForwardDiff.jacobian(fd_aug, s)
end
s = [z.z; @SVector [z.dt]]
@test my_discrete_jacobian(s) ≈ discrete_jacobian(RK3, models, z)

xf = rand(models)[1]
x1f = SVector{13}(xf[1:13])
x2f = SVector{13}(xf[14:26])
@test TO.state_diff(models, x, xf) ==
    [TO.state_diff(model, x1, x1f); TO.state_diff(model, x2, x2f)]
G1 = TO.state_diff_jacobian(model, x1)
G2 = TO.state_diff_jacobian(model, x2)
@test cat(G1,G2,dims=[1,2]) ≈ TO.state_diff_jacobian(models, x)
b1 = @SVector rand(12)
b2 = @SVector rand(12)
∇G1 = TO.∇²differential(model, x1, b1)
∇G2 = TO.∇²differential(model, x2, b2)
@test cat(∇G1, ∇G2, dims=[1,2]) ≈ TO.∇²differential(models, x, [b1;b2])


# Test solve
N = 101
tf = 5.0

x01 = Dynamics.build_state(model, [0, 1,0], UnitQuaternion(I), zeros(3), zeros(3))
x02 = Dynamics.build_state(model, [0,-1,0], UnitQuaternion(I), zeros(3), zeros(3))
x0 = [x01; x02]

xf1 = Dynamics.build_state(model, [ 1,0,0], expm(deg2rad( 45)*@SVector [1,0,0.]), zeros(3), zeros(3))
xf2 = Dynamics.build_state(model, [-1,0,0], expm(deg2rad(-45)*@SVector [1,0,0.]), zeros(3), zeros(3))
xf = [xf1; xf2]

Qd = Dynamics.fill_state(model, 1e-1, 1e-1, 1e-2, 1e-2)
Q = Diagonal([Qd; Qd])
Rd = @SVector fill(1e-2, 6)
R = Diagonal([Rd; Rd])
obj = LQRObjective(Q,R,Q*10,xf,N)

prob = Problem(models, obj, xf, tf, x0=x0)
solver = iLQRSolver(prob)
solver.opts.verbose = true
solve!(solver)

if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end
visualize!(vis, models, get_trajectory(solver))
