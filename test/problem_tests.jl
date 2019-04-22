using Test
using TrajectoryOptimization: ConstraintSet, rk4, rk3

# Discretize the model
T = Float64
integration = :rk4
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

costfun = LQRCost(Q, R, Qf, xf)

@test_nowarn Model{Discrete}(model, rk4)
@test_nowarn rk3(model)
model_d = Model{Discrete}(model, rk3)

# Create Problem
N = 10
@test_throws ArgumentError Problem(model,ObjectiveNew(costfun,N))
@test prob.model.info[:integration] == :rk4
@test prob.dt == 0.1

prob = Problem(model_d, ObjectiveNew(costfun,N), tf=5, N=51)
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
@test_nowarn Problem(model_d, ObjectiveNew(costfun,N), X0, U0, N=N, tf=5)
@test_nowarn Problem(model_d, ObjectiveNew(costfun,N), U0, N=N, tf=5)
U0 = rand(m,N)
prob = (@test_logs (:info, "Length of U should be N-1, not N. Trimming last entry") Problem(model_d, costfun, U0, N=N, tf=5))
@test length(prob.U) == N-1


# Change knot points
import TrajectoryOptimization: change_N, initial_state!
X0 = zeros(n,51)
X0[1,:] = sin.(range(0,stop=10,length=51))
initial_state!(prob,X0)
prob2 = change_N(prob,101)
@test prob2.N == 101


# TODO: Test minimum time
