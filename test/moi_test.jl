using TrajectoryOptimization
using Ipopt
using MathOptInterface
const MOI = MathOptInterface
const TO = TrajectoryOptimization

if !isdefined(Main,:TEST_TIME)
    TEST_TIME = true
end

# Parallel Park
Random.seed!(1)
prob = DubinsCarProblem(:parallel_park)
TO.add_dynamics_constraints!(prob)

nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)

optimizer = Ipopt.Optimizer()
TO.build_MOI!(nlp, optimizer)
MOI.optimize!(optimizer)
@test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
@test cost(nlp) < 0.0541
@test max_violation(nlp) < 1e-11
# TEST_TIME && @test optimizer.solve_time < 0.1

@test norm(states(nlp)[end] - prob.xf) < 1e-10
@test norm(states(nlp)[1] - prob.x0) < 1e-10

# # Cartpole
# prob = CartpoleProblem()
# # prob = Problems.Cartpole()[1]
# TO.add_dynamics_constraints!(prob)

# nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
# optimizer = Ipopt.Optimizer()
# TO.build_MOI!(nlp, optimizer)
# MOI.optimize!(optimizer)
# @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
# @test cost(nlp) < 1.50
# @test max_violation(nlp) < 1e-11
# # TEST_TIME && @test optimizer.solve_time < 1

# @test norm(states(nlp)[end] - prob.xf) < 1e-10
# @test norm(states(nlp)[1] - prob.x0) < 1e-10

# # Pendulum
# prob = Problems.Pendulum()[1]
# TO.add_dynamics_constraints!(prob)
#
# nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
# optimizer = Ipopt.Optimizer()
# TO.build_MOI!(nlp, optimizer)
# MOI.optimize!(optimizer)
# @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
# @test cost(nlp) < 0.0185
# @test max_violation(nlp) < 1e-9
# # TEST_TIME && @test optimizer.solve_time < 0.1
#
# @test norm(states(nlp)[end] - prob.xf) < 1e-10
# @test norm(states(nlp)[1] - prob.x0) < 1e-10
#
# # 3 Obstacles
# prob = Problems.DubinsCar(:three_obstacles)[1]
# TO.add_dynamics_constraints!(prob)
#
# nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
# optimizer = Ipopt.Optimizer()
# TO.build_MOI!(nlp, optimizer)
# MOI.optimize!(optimizer)
# @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
# @test cost(nlp) < 12.1
# @test max_violation(nlp) < 1e-8
# # TEST_TIME && @test optimizer.solve_time < 0.5
#
# @test norm(states(nlp)[end] - prob.xf) < 1e-10
# @test norm(states(nlp)[1] - prob.x0) < 1e-10
#
# # Escape
# prob = Problems.DubinsCar(:escape)[1]
# TO.add_dynamics_constraints!(prob)
#
# nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
# optimizer = Ipopt.Optimizer()
# TO.build_MOI!(nlp, optimizer)
# MOI.optimize!(optimizer)
# @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
# @test cost(nlp) < 0.333
# @test max_violation(nlp) < 1e-8
# # TEST_TIME && @test optimizer.solve_time < 5
# @test norm(states(nlp)[end] - prob.xf) < 1e-10
# @test norm(states(nlp)[1] - prob.x0) < 1e-10
