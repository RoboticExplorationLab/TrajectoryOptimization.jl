using Test
import TrajectoryOptimization: planar_obstacle_constraint

# Discretize the model
T = Float64
integration = :rk3
model = Dynamics.quadrotor_model
n = model.n; m = model.m

# cost
Q = (1.0e-2)*Diagonal(I,n)
R = (1.0e-2)*Diagonal(I,m)
Qf = 1000.0*Diagonal(I,n)

# -initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; 0.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;50.;0.] # xyz position
xf[4:7] = q0

# Knot points
N = 11

# Constraints
bnd = BoundConstraint(n,m, u_min=0, u_max=10)
goal = goal_constraint(xf)
obs = planar_obstacle_constraint(n,m, [1,1], 0.5)
constraints = bnd + goal + obs
pcon = Constraints(bnd+obs, obs+goal, N)
@test length(pcon) == N
num_constraints(pcon)
@test num_constraints(pcon) == [ones(N-1)*(2m+1); n+1]

# Create Objective
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)


# Initial state and control
U0 = [ones(m) for k = 1:N-1]
X0 = [copy(x0) for k = 1:N]

# Test model discretization
@test_nowarn rk3(model)
model_d = rk3(model)

# Create Problem from continuous model
@test_throws ArgumentError Problem(model, obj)  # needs N and another time indicator
prob = Problem(model, obj, tf=3.0)
@test prob.N == N
@test prob.dt == 0.3

@test_nowarn prob = Problem(model, obj, tf=3)  # pass in integer tf
@test_throws AssertionError Problem(model, obj, tf=3, N=15)  # N doesn't match length of obj
prob = Problem(model, obj, integration=:rk3, constraints=pcon, tf=3)
@test num_constraints(prob) == num_constraints(pcon)
@test prob.model.info[:integration] == :rk3

prob = Problem(model, obj, integration=:none, tf=3)
@test prob isa Problem{T, Continuous} where T
prob = Problem(model, obj, U0, integration=:none, tf=3)
@test prob isa Problem{T, Continuous} where T
prob = Problem(model, obj, X0, U0, integration=:none, constraints=pcon, tf=3)
@test prob isa Problem{T, Continuous} where T

# Create Problem from discrete model
prob = Problem(model_d, obj, tf=5, N=N)
@test prob.model.info[:integration] == :rk3

N2 = 101
obj2 = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N2)

prob = Problem(model_d, obj2, N=N2, tf=5)
@test prob.dt == 0.05
@test length(prob.X) == N2
@test length(prob.U) == N2-1

@test_nowarn Problem(model_d, obj, X0, U0, N=N, tf=5)
@test_nowarn Problem(model_d, obj, U0, N=N, tf=5)
U0 = rand(m,N)
@test_nowarn Problem(model_d, obj, to_array(X0), U0, tf=3)
@test_throws MethodError Problem(model_d, obj, X0, U0, tf=3)


# # Change knot points
# import TrajectoryOptimization: change_N, initial_state!
# X0 = zeros(n,51)
# X0[1,:] = sin.(range(0,stop=10,length=51))
# initial_state!(prob,X0)
# prob2 = change_N(prob,101)
# @test prob2.N == 101


# TODO: Test minimum time


# Test methods
import TrajectoryOptimization: set_x0!, is_constrained
prob = Problem(model_d, obj, constraints=pcon, x0=x0, tf=3)
@test size(prob) == (n,m,N)
prob2 = copy(prob)
prob2.U[1] = rand(m)
@test prob.U[1] != prob2.U[1]
set_x0!(prob2, rand(n))
@test prob.x0 == x0

@test is_constrained(prob)
prob = Problem(model_d, obj, x0=x0, tf=3)
@test !is_constrained(prob)
@test max_violation(prob) == 0

prob = Problem(model_d, obj, U0, constraints=pcon, x0=x0, tf=3)
rollout!(prob)
@test isfinite(cost(prob))
@test isfinite(max_violation(prob))

# Test integrating the problem
prob = Problem(model, obj, tf=3)
prob_d = rk3(prob)
@test prob_d isa Problem{T,Discrete} where T
@test prob_d.model.info[:integration] == :rk3
prob_d = rk4(prob)
@test prob_d isa Problem{T,Discrete} where T
@test prob_d.model.info[:integration] == :rk4
prob_d = midpoint(prob)
@test prob_d isa Problem{T,Discrete} where T
@test prob_d.model.info[:integration] == :midpoint
