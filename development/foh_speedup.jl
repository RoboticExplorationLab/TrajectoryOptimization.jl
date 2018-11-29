using Test
using BenchmarkTools

##

model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!

u_min_pendulum = -2
u_max_pendulum = 2
x_min_pendulum = [-20;-20]
x_max_pendulum = [20; 20]

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)

model = model_pendulum
obj = obj_con_pendulum
u_max = u_max_pendulum
u_min = u_min_pendulum

# Solver
intergrator_foh = :rk3_foh

dt = 0.005
solver1 = Solver(model,obj,integration=intergrator_foh,dt=dt)
U1 = ones(solver1.model.m,solver1.N)
X1 = rand(solver1.model.n,solver1.N)
solver2 = Solver(model,obj,integration=intergrator_foh,dt=dt)
U2 = ones(solver2.model.m,solver2.N)
X2 = rand(solver2.model.n,solver2.N)

results1 = init_results(solver1,X1,U1)
results2 = init_results(solver2,X2,U2)
rollout!(results1,solver1)
rollout!(results2,solver2)
calculate_jacobians!(results1, solver1)
calculate_jacobians!(results2, solver2)

println("TEST")
@time v1 = _backwardpass_foh!(results1,solver1)
@time v2 = _backwardpass_foh_speedup!(results2,solver2)

@test isapprox(v1[1:2], v2[1:2])
@test isapprox(to_array(results1.K),to_array(results2.K))
@test isapprox(to_array(results1.b),to_array(results2.b))
@test isapprox(to_array(results1.d),to_array(results2.d))

@btime _backwardpass_foh!(results1,solver1)
@btime _backwardpass_foh_speedup!(results2,solver2)
println("\n")

# Benchmark
# 1. eliminate redundant L calculations
# 189.564 ms (614530 allocations: 35.21 MiB)
# 161.772 ms (576530 allocations: 31.78 MiB)
# 2. distribute dt calculations
# 148.805 ms (614530 allocations: 35.21 MiB)
# 144.791 ms (570530 allocations: 31.57 MiB)
# 3. view into s,S
# 153.813 ms (614530 allocations: 35.21 MiB)
# 134.082 ms (548530 allocations: 30.94 MiB)
# 4. eliminate vec()
# 161.563 ms (614530 allocations: 35.21 MiB)
# 142.320 ms (547522 allocations: 30.92 MiB)
# 5. distribute discrete dynamics calculation
# 156.674 ms (614530 allocations: 35.21 MiB)
# 119.566 ms (491524 allocations: 27.89 MiB)
# 6. don't index into U matrix
# 159.905 ms (614530 allocations: 35.21 MiB)
# 122.204 ms (490524 allocations: 27.80 MiB)
# 7. view into U matrix and calculate transposes only once
# 159.132 ms (614530 allocations: 35.21 MiB)
# 104.974 ms (456524 allocations: 26.30 MiB)
# 8. replace more transposes
# 156.551 ms (614530 allocations: 35.21 MiB)
# 89.473 ms (424524 allocations: 25.34 MiB)
# 9. linear indexing -note, I found that using lin. idx on views was slower
# 160.612 ms (614530 allocations: 35.21 MiB)
# 89.392 ms (418534 allocations: 25.10 MiB)
# 10. more intermediate calculation replacements
# 172.848 ms (614530 allocations: 35.21 MiB)
# 88.955 ms (413533 allocations: 24.56 MiB)


# RK3, RK3 FOH, RK3 FOH speedup
# .0069ms, .0193ms, .0148ms
# 23% speedup
# previously (on Brian's computer) .00413, .0121 -> 2.929x slower
# now 2.14x slower


model, obj = Dynamics.cartpole_analytical
n,m = model.n, model.m
N = 51
dt = 0.1

obj.x0 = [0;0;0;0]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.0
U0 = ones(m,N)
solver = Solver(model,obj,N=N,opts=opts,integration=:rk3_foh)
res_i, stat_i = solve(solver,U0)

k = 10
time_per_iter = zeros(k)
for i = 1:k
  res_i, stat_i = solve(solver,U0)
  time_per_iter[i] = stat_i["runtime"]/stat_i["iterations"]
end
println(mean(time_per_iter))
