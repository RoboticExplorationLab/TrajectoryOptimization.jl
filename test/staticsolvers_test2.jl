using LabelledArrays

prob = copy(Problems.quad_obs)
n,m,N = size(prob)
dt = prob.dt

# Generate model
quad_ = Dynamics.Quadrotor()
generate_jacobian(quad_)
rk3_gen(quad_)
generate_discrete_jacobian(quad_)


# Test model
xs,us = (@SVector rand(n)), (@SVector rand(m))
x,u = Array(xs), Array(us)
zs = [xs;us]
z = [x;u]
initial_controls!(prob, [u for k = 1:N-1])
initial_states!(prob, [x for k = 1:N])

@btime dynamics($quad_,$xs,$us)
@btime discrete_dynamics($quad_,$xs,$us,$dt)

@btime jacobian($quad_,$zs)
@btime discrete_jacobian($quad_,$zs,$dt)


# Build Problem
x0 = SVector{n}(prob.x0)
xf = SVector{n}(prob.xf)
Z = [zs for k = 1:N]
SProblem(quad_, prob.obj, x0, xf, Z, N, dt, prob.tf)

# Build Solver
silqr = StaticiLQRSolver(prob)
ilqr = iLQRSolver(prob)


z = KnotPoint(x,u,dt)
@btime state($z)
@btime control($z)
@btime is_terminal($z)
Z = [z for k = 1:N]
Z[end] = KnotPoint(x,m)

# Dynamics evaluations
dyn = [xs for k = 1:N]
@btime discrete_dynamics!($dyn, $quad_, $Z)

# Dynamics jacobians
∇f = silqr.∇F
jacobian!(prob, ilqr)
discrete_jacobian!(∇f, quad_, Z)
silqr.∇F ≈ ilqr.∇F
@btime jacobian!($prob, $ilqr)
@btime discrete_jacobian!($∇f, $quad_, $Z) # ≈3x faster

# Objective
obj = prob.obj
Q = (1.0e-3)*Diagonal(@SVector ones(n))  # use static diagonal to avoid allocations
R = (1.0e-2)*Diagonal(@SVector ones(m))
Qf = 1.0*Diagonal(@SVector ones(n))
obj = LQRObjective(Q,R,Qf,xf,N)

cost(prob) ≈ cost(obj, Z)
@btime cost($prob)
@btime cost($obj, $Z) # 7.5x faster


# Cost expansion
E = CostExpansion(
    [@SVector zeros(n) for k = 1:N],
    [@SVector zeros(m) for k = 1:N],
    [@SMatrix zeros(n,n) for k = 1:N],
    [@SMatrix zeros(m,m) for k = 1:N],
    [@SMatrix zeros(m,n) for k = 1:N]
)
cost_exupansion!(prob, ilqr)
cost_expansion(E, obj, Z)
all([E.xx[k] ≈ ilqr.Q[k].xx for k in eachindex(E.xx)])
all([E.uu[k] ≈ ilqr.Q[k].uu for k = 1:N-1])
all([E.x[k] ≈ ilqr.Q[k].x for k in eachindex(E.x)])
all([E.u[k] ≈ ilqr.Q[k].u for k = 1:N-1])

@btime cost_expansion!($prob, $ilqr)
@btime cost_expansion($E, $obj, $Z)  # 7x faster
