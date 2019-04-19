# Car

function car_dynamics!(xdot,x,u)
    xdot[1] = u[1]*cos(x[3])
    xdot[2] = u[1]*sin(x[3])
    xdot[3] = u[2]
    return nothing
end
n,m = 3,2

car_model = Model(car_dynamics!,n,m)
