using DiffResults

f = Dynamics.dubins_dynamics!
n,m = 3,2
f_aug! = f_augmented!(f,n,m)

fd_aug! = rk3(f_aug)
nm1 = n + m + 1

In = 1.0*Matrix(I,n,n)

# Initialize discrete and continuous dynamics Jacobians



# Get continuous dynamics jacobian
function gen_discrete_jacobians(fd_aug!,n,m,dt)
    nm1 = n+m+1
    Jd = zeros(nm1, nm1)
    s = zeros(nm1)
    sdot = zero(s)
    Fd!(Jd,Sdotd,Sd) = ForwardDiff.jacobian!(Jd,fd_aug!,Sdotd,Sd)
    s[end] = sqrt(dt)

    xi = 1:n
    ui = n .+ (1:m)

    Jdi = CartesianIndices(Jd)
    xx = Jdi[xi,xi]
    xu = Jdi[xi,ui]
    xu2 = Jdi[xi,1:m]

    function fd_jacobians!(fdx,fdu,x,u)
        # Assign state, control (and dt) to augmented vector
        s[1:n] = x
        s[n+1:n+m] = u[1:m]

        # Calculate Jacobian
        Fd!(Jd,sdot,s)

        fdx[1:n,1:n] = Jd[1:n,1:n]
        fdu[1:n,1:m] = Jd[1:n,n.+(1:m)]

    end

    function fd_jacobians2!(fdx,fdu,x,u)
        # Assign state, control (and dt) to augmented vector
        s[xi] = x
        s[ui] = u

        # Calculate Jacobian
        Fd!(Jd,sdot,s)

        fdx[xx] = Jd[xx]
        fdu[xu2] = Jd[xu]

    end
    return fd_jacobians!, fd_jacobians2!
end





function gen_continuous_jacobians(f_aug!,n,m,dt)
    xi = 1:n
    ui = n .+ (1:m)

    Jc = zeros(n+m,n+m)
    z = zeros(n+m)
    zdot = zeros(n+m)
    A = view(Jc,xi,xi)
    B = view(Jc,xi,ui)
    function fc_jacobians2!(x,u)
        z[xi] = x
        z[ui] = u
        ForwardDiff.jacobian!(Jc,f_aug!,zdot,z)
        return A,B
    end

    function fc_jacobians!(x,u)
        # infeasible = size(u,1) != m̄
        z[1:n] = x
        z[n+1:n+m] = u[1:m]
        Fc!(Jc,zdot,z)
        return Jc[1:n,1:n], Jc[1:n,n+1:n+m] # fx, fu
    end

    function rk3_jacobian!(x,u)
        z[xi] = x
        z[ui] = u
        A1,B1 = fc_jacobians2!(x,u)
        k1 = zdot*dt
        A2,B2 = fc_jacobians2!(x+k1[xi]/2,u)
        k2 = zdot*dt
        A3,B3 = fc_jacobians2!(x-k1[xi]+2*k2[xi],u)
        k3 = zdot*dt

        dk1_x = A1*dt
        dk2_x = (I + dk1_x/2)'A2*dt
        dk3_x = (I - dk1_x + 2*dk2_x)'A3*dt
        Ad = I + (dk1_x + 4*dk2_x + dk3_x)/6

        dk1_u = B1*dt
        dk2_u = A2'*(dk1_u/2)*dt + B2*dt
        dk3_u = A3'*(-dk1_u + 2*dk2_u)*dt + B3*dt
        Bd = (dk1_u + 4*dk2_u + dk3_u)/6
        return Ad,Bd
    end

    return fc_jacobians!, fc_jacobians2!, rk3_jacobian!
end

f_aug! = f_augmented!(f,n,m)
dt = 0.1
fc_jacob, fc_jacob2, fd_jacob = gen_continuous_jacobians(f_aug!,n,m,dt)

x,u = rand(n), rand(m)
xdot = zero(x)
z = [x;u]
zdot = zero(z)
s = [z;u;sqrt(dt)]
sdot = zero(s)
fc_jacob(x,u) == fc_jacob2(x,u)
fd_jacob(x,u)


# Discrete jacobians
fd! = rk3(f,dt)
fd_aug! = rk3(f_aug)
fd!(xdot,x,u)
fd_aug!(sdot,s)
xdot == sdot[1:n]

fd_jacob!, fd_jacob2! = gen_discrete_jacobians(fd_aug!,n,m,dt)
fdx = zeros(n,n)
fdu = zeros(n,m)
fd_jacob!(fdx,fdu,x,u) == fd_jacob2!(fdx,fdu,x,u)

@btime fc_jacob($x,$u)
@btime fc_jacob2($x,$u)
@btime fc_jacob2($x,$u)


@btime fd_jacob(x,u)
@btime fd_jacob!(fdx,fdu,x,u)


nm1 = n+m+1
Jd = zeros(nm1, nm1)

xi = 1:n
ui = n .+ (1:m)

Jdi = CartesianIndices(Jd)
xx = Jdi[xi,xi]
xu = Jdi[xi,ui]
