using TrajectoryOptimization
using Ipopt
using MathOptInterface
const MOI = MathOptInterface
const TO = TrajectoryOptimization

prob = Problems.DubinsCar(:parallel_park)[1]
TO.add_dynamics_constraints!(prob)

nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
optimizer = Ipopt.Optimizer()
TO.build_MOI!(nlp, optimizer)
MOI.optimize!(optimizer)
@test cost(nlp) < 0.0541
@test max_violation(nlp) < 1e-11
TEST_TIME && @test optimizer.solve_time < 0.1

@test norm(states(nlp)[end] - prob.xf) < 1e-10
@test norm(states(nlp)[1] - prob.x0) < 1e-10
