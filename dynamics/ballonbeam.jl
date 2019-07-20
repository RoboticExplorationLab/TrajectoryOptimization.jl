## Ball on beam
#TODO make inplace
function ballonbeam_dynamics(x::AbstractVector{T},u::AbstractVector{T}) where T
    g = 9.81
    m1 = .35
    m2 = 2
    l = 0.5

    z = x[1];
    theta = x[2];
    zdot = x[3];
    thetadot = x[4];
    F = u[1];
    zddot = z*thetadot^2-g*sin(theta);
    thetaddot = (F*l*cos(theta) - 2*m1*z*zdot*thetadot - m1*g*z*cos(theta) - (m2*g*l*cos(theta))/2) / (m2*l^2/3 + m1*z^2);

    sys = [zdot; thetadot; zddot; thetaddot];
end
n,m = 4,1
ballonbeam = Model(ballonbeam_dynamics,n,m)
