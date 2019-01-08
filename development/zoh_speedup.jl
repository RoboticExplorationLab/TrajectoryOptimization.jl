using Test
using BenchmarkTools

##

model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!

u_min_pendulum = -2
u_max_pendulum = 2
x_min_pendulum = [-20;-20]
x_max_pendulum = [20; 20]

model_dubins, obj_uncon_dubins = TrajectoryOptimization.Dynamics.dubinscar!

# dubins car
u_min_dubins = [-1; -1]
u_max_dubins = [1; 1]
x_min_dubins = [0; -100; -100]
x_max_dubins = [1.0; 100; 100]
obj_con_dubins = ConstrainedObjective(obj_uncon_dubins, u_min=u_min_dubins, u_max=u_max_dubins, x_min=x_min_dubins, x_max=x_max_dubins)

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)

model = model_dubins
obj = obj_con_dubins
u_max = u_max_dubins
u_min = u_min_dubins

obj = TrajectoryOptimization.update_objective(obj, tf=:min, c=0.0, Q = 1e-3*Diagonal(I,model.n), R = 1e-3*Diagonal(I,model.m), Qf = Diagonal(I,model.n))
obj = obj_con_dubins
# Solver
intergrator_foh = :rk3

dt = 0.005
solver1 = Solver(model,obj,integration=intergrator_foh,N=51)
solver1.opts.minimum_time = false
solver1.opts.infeasible = false
solver1.opts.constrained = true
X0 = line_trajectory(solver1)
U0 = ones(solver1.model.m,solver1.N)
# U0 = [U0; ones(1,solver1.N)]
# u0 = infeasible_controls(solver1,X0)
# U0 = [U0;u0]
solver2 = Solver(model,obj,integration=intergrator_foh,N=51)
solver2.opts.minimum_time = false
solver2.opts.infeasible = false
solver2.opts.constrained = true
solver2.opts.R_infeasible = dt*solver1.opts.R_infeasible*tr(solver1.obj.R)

results1 = init_results(solver1,X0,U0)
results2 = init_results(solver2,X0,U0)
rollout!(results1,solver1)
rollout!(results2,solver2)
update_jacobians!(results1, solver1)
update_jacobians!(results2, solver2)

println("TEST")
@time v1 = _backwardpass!(results1,solver1)
@time v2 = _backwardpass_speedup!(results2,solver2)

@test isapprox(v1[1:2], v2[1:2])
@test isapprox(to_array(results1.K),to_array(results2.K))
@test isapprox(to_array(results1.b),to_array(results2.b))
@test isapprox(to_array(results1.d),to_array(results2.d))

@benchmark _backwardpass!($results1,$solver1)
@benchmark _backwardpass_speedup!($results2,$solver2)
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

# 12-3-2018
# 1.
# 466.429 ms (511527 allocations: 38.27 MiB)
# 289.674 ms (338527 allocations: 26.21 MiB)
# 2. Use pre-allocated S, s
# 295.870 ms (511527 allocations: 38.27 MiB)
# 194.656 ms (338523 allocations: 26.21 MiB)
# 3. Using views into all L,l is slow and more allocations
# --- Using minimum_time and infeasible
# 4.
# 22.327 ms (43878 allocations: 4.40 MiB)
# 17.161 ms (35044 allocations: 3.71 MiB)
# 5. - reduced number of calculations (also found small error in Lu)
# 21.847 ms (43878 allocations: 4.40 MiB)
# 18.010 ms (33769 allocations: 3.55 MiB)
# 6. - more calculation reductions
# 22.863 ms (43878 allocations: 4.40 MiB)
# 17.340 ms (33361 allocations: 3.51 MiB)
# 7. -views into preallocated L,l for minimum_time
# 22.817 ms (43878 allocations: 4.40 MiB)
# 16.899 ms (29273 allocations: 3.21 MiB)
# 8. -view into preallocated L,l for infeasible start
# 22.136 ms (43878 allocations: 4.40 MiB)
# 13.648 ms (25710 allocations: 3.01 MiB)
# 9. -removed duplicate assignments
# 22.393 ms (43878 allocations: 4.40 MiB)
# 13.492 ms (24996 allocations: 2.99 MiB)
# 10. -smarter indexing
# 39.727 ms (43878 allocations: 4.40 MiB)
# 19.778 ms (24639 allocations: 2.97 MiB)

model, obj = Dynamics.cartpole_analytical
n,m = model.n, model.m
N = 51
dt = 0.1

obj.x0 = [0;0;0;0]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.0
U0 = ones(m,N)
solver_foh = Solver(model,obj,N=N,opts=opts,integration=:rk3_foh)
solver_zoh = Solver(model,obj,N=N,opts=opts,integration=:rk3)

k = 10
time_per_iter_foh = zeros(k)
time_per_iter_zoh = zeros(k)

for i = 1:k
  res_foh, stat_foh = solve(solver_foh,U0)
  res_zoh, stat_zoh = solve(solver_zoh,U0)
  time_per_iter_foh[i] = stat_foh["runtime"]/stat_foh["iterations"]
  time_per_iter_zoh[i] = stat_zoh["runtime"]/stat_zoh["iterations"]
end
println("Time per iter (foh): $(mean(time_per_iter_foh))")
println("Time per iter (zoh): $(mean(time_per_iter_zoh))")
