using Test
using TrajectoryOptimization: ConstraintSet, rk4, rk3

# Discretize the model
model = Dynamics.quadrotor[1]
@test_nowarn Model{Discrete}(model, rk4)
@test_nowarn rk3(model)
model_d = Model{Discrete}(model, rk3)

# Get cost function
costfun = Dynamics.quadrotor[2].cost

# Create Problem
@test_throws ArgumentError Problem(model,costfun)
prob = (@test_logs (:warn, "Neither dt or N were specified. Setting N = 51") Problem(model,costfun,tf=5))
@test prob.model.info[:integration] == :rk4
@test prob.dt == 0.1

prob = Problem(model_d, costfun, tf=5, N=51)
@test prob.model.info[:integration] == :rk3

@test_nowarn Problem(model_d, costfun, N=51, dt=0.2)
prob = Problem(model_d, costfun, N=101, tf=5)
@test prob.dt == 0.05
@test length(prob.X) == 101
@test length(prob.U) == 100

n,m = model.n, model.m
N = 51
X0 = rand(n,N)
U0 = rand(m,N-1)
@test_nowarn Problem(model_d, costfun, X0, U0, N=N, tf=5)
@test_nowarn Problem(model_d, costfun, U0, N=N, tf=5)
U0 = rand(m,N)
prob = (@test_logs (:info, "Length of U should be N-1, not N. Trimming last entry") Problem(model_d, costfun, U0, N=N, tf=5))
@test length(prob.U) == N-1


# TODO: Test minimum time
