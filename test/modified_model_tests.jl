# import TrajectoryOptimization: dynamics
# import TrajectoryOptimization: Model, LQRCost, Problem, Objective, rollout!, iLQRSolverOptions,
#     AbstractSolver, jacobian!, _backwardpass!, _backwardpass_sqrt!, AugmentedLagrangianSolverOptions, ALTROSolverOptions,
#     goal_constraint, update_constraints!, update_active_set!, jacobian!, update_problem,
#     line_trajectory, total_time, generate_jacobian, _check_dynamics, AnalyticalModel, _test_jacobian,
#     f_augmented!, add_slack_controls, add_min_time_controls
#
# ## Augment dynamics
# model = Dynamics.pendulum_model
# model_d = discretize_model(model,:rk4)
#
# # Add slack controls
# model_inf = add_slack_controls(model_d)
#
# @test model_inf.n == model_d.n
# @test model_inf.m == model_d.m + model_d.n
#
# x = rand(model_d.n)
# u = rand(model_d.m)
# u_inf = rand(model_inf.m); u_inf[1:model_d.m] .= u
# x_d = zeros(model_d.n)
# x_inf = zeros(model_inf.n)
# Z_d = zeros(model_d.n,model_d.n+model_d.m+1)
# Z_inf = zeros(model_inf.n,model_inf.n+model_inf.m+1)
#
# evaluate!(x_d,model_d,x,u,1.0)
# evaluate!(x_inf,model_inf,x,u_inf,1.0)
#
# evaluate!(Z_d,rand(model_d.n),model_d,x,u,1.0)
# model_inf.∇f(Z_inf,x,u_inf,1.0)
#
# @test x_inf == x_d + u_inf[model_d.m+1:end]
# @test Z_d == Z_inf[1:model_d.n,[(1:model_d.n+model_d.m)...,2model_d.n+model_d.m+1]]
# @test Z_inf[1:model_d.n,(1:model_d.n) .+ (model_d.n+model_d.m)] == Diagonal(1.0I,model_d.n)
#
# # Add minimum time controls
# model_mintime = add_min_time_controls(model_d)
#
# @test model_mintime.n == model_d.n + 1
# @test model_mintime.m == model_d.m + 1
#
# dt = 0.75
# x = rand(model_d.n)
# u = rand(model_d.m)
# u_mintime = sqrt(dt)*ones(model_mintime.m); u_mintime[1:model_d.m] .= u
# x_d = zeros(model_d.n)
# x_mintime = rand(model_mintime.n); x_mintime[1:model_d.n] .= x
# x₊_mintime = zeros(model_mintime.n)
# Z_d = zeros(model_d.n,model_d.n+model_d.m+1)
# Z_mintime = zeros(model_mintime.n,model_mintime.n+model_mintime.m)
#
# evaluate!(x_d,model_d,x,u,dt)
# evaluate!(x₊_mintime,model_mintime,x_mintime,u_mintime,Inf) # note we can use Inf here since dt input is not used
#
# evaluate!(Z_d,rand(model_d.n),model_d,x,u,dt)
# model_mintime.∇f(Z_mintime,x_mintime,u_mintime,Inf)
#
# @test isapprox(x₊_mintime[1:model_d.n],x_d)
#
# @test isapprox(x₊_mintime[1:model_d.n],x_d)
# @test x₊_mintime[end] == u_mintime[end]
# @test isapprox(Z_d[1:model_d.n,1:model_d.n+model_d.m],Z_mintime[1:model_d.n,[(1:model_d.n)...,((1:model_d.m) .+ (model_d.n+1))...]])
# @test isapprox(Z_mintime[1:model_d.n,model_d.n+model_d.m+2],Z_d[:,end]*2.0*u_mintime[end])
# @test Z_mintime[model_d.n+1,model_d.n+model_d.m+2] == 1.0
#
# # Add infeasible and minimum time controls
# model_altro = add_min_time_controls(model_inf)
#
# dt = 0.75
# x = rand(model_d.n)
# u = rand(model_d.m)
# u_altro = sqrt(dt)*ones(model_altro.m); u_altro[1:model_d.m] .= u
# u_altro
# x_d = zeros(model_d.n)
# x_altro = rand(model_altro.n); x_altro[1:model_d.n] .= x
# x₊_altro = zeros(model_altro.n)
# Z_d = zeros(model_d.n,model_d.n+model_d.m+1)
# Z_altro = zeros(model_altro.n,model_altro.n+model_altro.m)
#
# @test model_altro.n == model_d.n + 1
# @test model_altro.m == model_d.m + model_d.n + 1
#
# evaluate!(x_d,model_d,x,u,dt)
# evaluate!(x₊_altro,model_altro,x_altro,u_altro,Inf) # note we can use Inf here since dt input is not used
#
# evaluate!(Z_d,rand(model_d.n),model_d,x,u,dt)
# model_altro.∇f(Z_altro,x_altro,u_altro,Inf)
#
# @test isapprox(x₊_altro[1:model_d.n],x_d + u_altro[(1:model_d.n) .+ model_d.n])
# @test x₊_altro[end] == u_altro[end]
# @test isapprox(Z_d[1:model_d.n,1:model_d.n+model_d.m],Z_altro[1:model_d.n,[(1:model_d.n)...,((1:model_d.m) .+ (model_d.n+1))...]])
# @test isapprox(Z_altro[1:model_d.n,model_altro.n+model_altro.m],Z_d[:,end]*2.0*u_altro[end])
# @test Z_altro[model_d.n+1,model_altro.n+model_altro.m] == 1.0

import TrajectoryOptimization: dynamics
import TrajectoryOptimization: Model, LQRCost, Problem, Objective, rollout!, iLQRSolverOptions,
    AbstractSolver, jacobian!, _backwardpass!, _backwardpass_sqrt!, AugmentedLagrangianSolverOptions, ALTROSolverOptions,
    goal_constraint, update_constraints!, update_active_set!, jacobian!, update_problem,
    line_trajectory, total_time, generate_jacobian, _check_dynamics, AnalyticalModel, _test_jacobian,
    f_augmented!, add_slack_controls, add_min_time_controls, Nominal

## Augment dynamics
model = Dynamics.pendulum_model
model_d = discretize_model(model,:rk4)

# Add slack controls
model_inf = add_slack_controls(model_d)

@test model_inf.n == model_d.n
@test model_inf.m == model_d.m + model_d.n

x = rand(model_d.n)
u = rand(model_d.m)
u_inf = rand(model_inf.m); u_inf[1:model_d.m] .= u
x_d = zeros(model_d.n)
x_inf = zeros(model_inf.n)
Z_d = zeros(model_d.n,model_d.n+model_d.m+1)
Z_inf = zeros(model_inf.n,model_inf.n+model_inf.m+1)

evaluate!(x_d,model_d,x,u,1.0)
evaluate!(x_inf,model_inf,x,u_inf,1.0)

jacobian!(Z_d,model_d,x,u,1.0)
model_inf.∇f(Z_inf,x,u_inf,1.0)

@test x_inf == x_d + u_inf[model_d.m+1:end]
@test Z_d == Z_inf[1:model_d.n,[(1:model_d.n+model_d.m)...,2model_d.n+model_d.m+1]]
@test Z_inf[1:model_d.n,(1:model_d.n) .+ (model_d.n+model_d.m)] == Diagonal(1.0I,model_d.n)

# Add minimum time controls
model_mintime = add_min_time_controls(model_d)

@test model_mintime.n == model_d.n + 1
@test model_mintime.m == model_d.m + 1

dt = 0.75
x = rand(model_d.n)
u = rand(model_d.m)
u_mintime = sqrt(dt)*ones(model_mintime.m); u_mintime[1:model_d.m] .= u
x_d = zeros(model_d.n)
x_mintime = rand(model_mintime.n); x_mintime[1:model_d.n] .= x
x₊_mintime = zeros(model_mintime.n)
Z_d = zeros(model_d.n,model_d.n+model_d.m+1)
Z_mintime = zeros(model_mintime.n,model_mintime.n+model_mintime.m)

evaluate!(x_d,model_d,x,u,dt)
evaluate!(x₊_mintime,model_mintime,x_mintime,u_mintime,Inf) # note we can use Inf here since dt input is not used

jacobian!(Z_d,model_d,x,u,dt)
model_mintime.∇f(Z_mintime,x_mintime,u_mintime,Inf)

@test isapprox(x₊_mintime[1:model_d.n],x_d)

@test isapprox(x₊_mintime[1:model_d.n],x_d)
@test x₊_mintime[end] == u_mintime[end]
@test isapprox(Z_d[1:model_d.n,1:model_d.n+model_d.m],Z_mintime[1:model_d.n,[(1:model_d.n)...,((1:model_d.m) .+ (model_d.n+1))...]])
@test isapprox(Z_mintime[1:model_d.n,model_d.n+model_d.m+2],Z_d[:,end]*2.0*u_mintime[end])
@test Z_mintime[model_d.n+1,model_d.n+model_d.m+2] == 1.0

# Add infeasible and minimum time controls
model_altro = add_min_time_controls(model_inf)

dt = 0.75
x = rand(model_d.n)
u = rand(model_d.m)
u_altro = sqrt(dt)*ones(model_altro.m); u_altro[1:model_d.m] .= u
x_d = zeros(model_d.n)
x_altro = rand(model_altro.n); x_altro[1:model_d.n] .= x
x₊_altro = zeros(model_altro.n)
Z_d = zeros(model_d.n,model_d.n+model_d.m+1)
Z_altro = zeros(model_altro.n,model_altro.n+model_altro.m)

@test model_altro.n == model_d.n + 1
@test model_altro.m == model_d.m + model_d.n + 1

evaluate!(x_d,model_d,x,u,dt)
evaluate!(x₊_altro,model_altro,x_altro,u_altro,Inf) # note we can use Inf here since dt input is not used

jacobian!(Z_d,model_d,x,u,dt)
model_altro.∇f(Z_altro,x_altro,u_altro,Inf)

@test isapprox(x₊_altro[1:model_d.n],x_d + u_altro[(1:model_d.n) .+ model_d.n])
@test x₊_altro[end] == u_altro[end]
@test isapprox(Z_d[1:model_d.n,1:model_d.n+model_d.m],Z_altro[1:model_d.n,[(1:model_d.n)...,((1:model_d.m) .+ (model_d.n+1))...]])
@test isapprox(Z_altro[1:model_d.n,model_altro.n+model_altro.m],Z_d[:,end]*2.0*u_altro[end])
@test Z_altro[model_d.n+1,model_altro.n+model_altro.m] == 1.0
