# Infeasible
model = RobotZoo.DubinsCar()
inf = TO.InfeasibleModel(model)
@test inf._u == 1:2
@test inf._ui == 3:5
@test size(inf) == (3,5)
x = rand(inf)[1]
u0 = @SVector rand(2)
ui = @SVector rand(3)
u = [u0; ui]
dt = 0.1
@test length(rand(inf)[2]) == 5
@test_throws ErrorException dynamics(inf, x, u)
@test discrete_dynamics(RK3, inf, KnotPoint(x, u, dt)) ==
    (discrete_dynamics(RK3, model, x, u0, 0.0, dt) + ui)

function inf_allocs(inf)
    x,u = rand(inf)
    n,m = size(inf)
    dt = 0.1
    z = KnotPoint(x,u,0.1)
    # allocs = @allocated discrete_dynamics(RK3, inf, x, u, dt)
    ∇c = zeros(n,n+m+1)
    allocs = @allocated discrete_dynamics(RK3, inf, z)
    allocs += @allocated discrete_jacobian!(RK3, ∇c, inf, z)
end
@test inf_allocs(inf) == 0
