using Test
if !isdefined(Main,:TEST_TIME)
    TEST_TIME = true
else
    TEST_TIME = false
end

# Double Integrator
solver = ALTROSolver(Problems.DoubleIntegrator()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 0.5
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 8 # 8

# Pendulum
solver = ALTROSolver(Problems.Pendulum()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 1
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 19 # 19

# Cartpole
solver = ALTROSolver(Problems.Cartpole()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 5
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 40 # 40

# Acrobot
solver = ALTROSolver(Problems.Acrobot()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 10
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 50 # 50

# Parallel Park
solver = ALTROSolver(Problems.DubinsCar(:parallel_park)...)
b =  benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time /1e6 < 10
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 13 # 13

# Three Obstacles
solver = ALTROSolver(Problems.DubinsCar(:three_obstacles)...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time /1e6 < 10
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 20 # 20

# Escape
solver = ALTROSolver(Problems.DubinsCar(:escape)..., infeasible=true, R_inf=0.1)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 20
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 13 # 13

# Zig-zag
solver = ALTROSolver(Problems.Quadrotor(:zigzag)...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 50
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 15 # 14

# Barrell Roll
solver = ALTROSolver(Problems.YakProblems()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 100
@test max_violation(solver) < 1e-6
@test iterations(solver) <= 18 # 18
