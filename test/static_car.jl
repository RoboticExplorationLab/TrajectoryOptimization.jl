car = Dynamics.DubinsCar()
n,m = size(car)
x = @SVector rand(n); u = @SVector rand(m);
z = KnotPoint(x,u,0.1)

@btime dynamics($car, $z)

discrete_dynamics(RK3, car, x, u, 0.1)
@btime discrete_dynamics(RK3, $car,$x,$u, 0.1)
@btime discrete_dynamics(RK3, $car, $z)
@btime discrete_jacobian(RK3, $car, $z)
