
function double_integrator_dynamics!(ẋ,x,u)
    ẋ[1] = x[2]
    ẋ[2] = u[1]
end

n = 2
m = 1

model_admm = Model(double_integrator_dynamics!,n,m)

# initial and goal states
x0 = [1.;0.]
xf = [0.;0.]

# costs
Q = (1e-2)*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.1

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

double_integrator = [model, obj_uncon]

# double double integrator
function double_double_integrator!(x_::AbstractArray,x::AbstractArray,u::AbstractArray,Δt=0.1)::Nothing

    m1 = 1 # mass of body 1
    m2 = 1 # mass of mass

    M1 = [m1 0; 0 m1]
    M2 = [m2 0; 0 m2]

    M1inv = [1/m1 0; 0 1/m1]
    M2inv = [1/m2 0; 0 1/m2]

    # body 1
    y = x[1:2]
    ẏ = x[3:4]

    # mass
    z = x[5:6]
    ż = x[7:8]

    # body 1 control
    uy = u[1:2]
    uz = u[3:4]

    ## implicit euler
    # body 1 update
    x_[3:4] = ẏ + Δt*M1inv*(uy)
    x_[1:2] = y + Δt*x_[3:4]

    # mass update
    x_[7:8] = ż + Δt*M2inv*(uz)
    x_[5:6] = z + Δt*x_[7:8]

    return nothing
end

# model = Model(double_double_integrator!,8,4)
# tf = 1.0
# y0 = [0.;1.]
# ẏ0 = [0.;0.]
# z0 = [0.;0.]
# ż0 = [0.;0.]
# x0 = [y0;ẏ0;z0;ż0]
#
# yf = [10.;1.]
# ẏf = ẏ0
# zf = [10.;0.]
# żf = ż0
# xf = [yf;ẏf;zf;żf]
#
# Q = Diagonal(0.0001I,model.n)
# R = Diagonal(0.0001I,model.m)
# Qf = Diagonal(100.0I,model.n)
#
# function cE(c,x::AbstractArray,u::AbstractArray)
#     c[1] = norm(x[1:2] - x[5:6])^2 - d^2
#     c[2] = u[3] - u[4]
# end
#
# obj = LQRObjective(Q, R, Qf, tf, x0, xf)#,cE=cE,use_xf_equality_constraint=false)
# solver = Solver(model,obj,integration=:none,dt=0.1)
# solver.opts.verbose = true
# solver.opts.cost_tolerance = 1e-8
# results, stats = solve(solver,rand(model.m,solver.N-1))
# plot(to_array(results.U)')
# update_jacobians!(results,solver)
#
# A = zeros(model.n,model.n)
# B = zeros(model.n,model.m)
# xx = rand(model.n)
# uu = rand(model.m)
# solver.Fd(A,B,xx,uu)
# yy = zeros(model.n)
# model.f(yy,xx,uu)
# yy
# f_aug = f_augmented!(model.f,model.n,model.m)
# using ForwardDiff
# ForwardDiff.jacobian(f_aug,rand(model.n),[xx;uu])
#
# A
# B
# Constrained Double Integrators ()
n1,m1 = 4,3
n2,m2 = 4,1
N = n1+n2
M = m1 + m2

bodies = (:a1,:m)

part_x = create_partition((n1,n2),bodies)
part_u = create_partition((m1,m2),bodies)
y0 = [0,1.,0,0]
v0 = zeros(m1)
z0 = [0.5,0,0,0]
w0 = zeros(m2)
f0 = 0
x0 = [y0;z0]
d = 1
u0 = [v0;w0;0;0]

Δt = 0.1

function double_integrator_constrained_system!(x_::AbstractArray,x::AbstractArray,u::AbstractArray,Δt=0.1)::Nothing

    m1 = 1 # mass of body 1
    m2 = 1 # mass of mass

    M1 = [m1 0; 0 m1]
    M2 = [m2 0; 0 m2]

    M1inv = [1/m1 0; 0 1/m1]
    M2inv = [1/m2 0; 0 1/m2]

    # body 1
    y = x[1:2]
    ẏ = x[3:4]

    # mass
    z = x[5:6]
    ż = x[7:8]

    # body 1 control
    uy = u[1:2]

    # constraint force
    fy = u[3]
    fz = u[4]

    # constraint Jacobians
    jy = (y - 2z)'
    jz = (-2y + z)'

    ## implicit euler
    # body 1 update
    x_[3:4] = ẏ + Δt*M1inv*(uy + jy'*fy)
    x_[1:2] = y + Δt*x_[3:4]

    # mass update
    x_[7:8] = ż + Δt*M2inv*(jz'*fz)
    x_[5:6] = z + Δt*x_[7:8]

    return nothing
end

model_admm = Model(double_integrator_constrained_system!,N,M)

tf = 1.0
y0 = [0.;1.]
ẏ0 = [0.;0.]
z0 = [0.;0.]
ż0 = [0.;0.]
x0 = [y0;ẏ0;z0;ż0]

yf = [10.;1.]
ẏf = ẏ0
zf = [10.;0.]
żf = ż0
xf = [yf;ẏf;zf;żf]

Q = Diagonal(0.0001I,model_admm.n)
R = Diagonal(0.0001I,model_admm.m)
Qf = Diagonal(100.0I,model_admm.n)

function double_integrator_constrained_system2!(x_::AbstractArray,x::AbstractArray,u::AbstractArray,Δt=0.1)::Nothing
    g = [0;9.81]
    m1 = 1 # mass of body 1
    m2 = 1 # mass of mass

    M1 = [m1 0; 0 m1]
    M2 = [m2 0; 0 m2]

    M1inv = [1/m1 0; 0 1/m1]
    M2inv = [1/m2 0; 0 1/m2]

    # body 1
    y = x[1:2]
    ẏ = x[3:4]

    # mass
    z = x[5:6]
    ż = x[7:8]

    # body 1 control
    uy = u[1:2]

    # constraint force
    fy = u[3:4]
    fz = u[5:6]

    # constraint Jacobians
    jy = (y - 2z)'
    jz = (-2y + z)'

    ## implicit euler
    # body 1 update
    x_[3:4] = ẏ + Δt*M1inv*(uy + fy -m1*g)
    x_[1:2] = y + Δt*x_[3:4]

    # mass update
    x_[7:8] = ż + Δt*M2inv*(fz -m2*g)
    x_[5:6] = z + Δt*x_[7:8]

    return nothing
end
n1,m1 = 4,4
n2,m2 = 4,2
N = n1+n2
M = m1 + m2
model_admm2 = Model(double_integrator_constrained_system2!,N,M)


function agents2_mass1_2D!(x_::AbstractArray,x::AbstractArray,u::AbstractArray,Δt=0.1)::Nothing
    g = [0;9.81]
    mb = 1 # mass of load
    ma1 = 1 # mass of agent 1
    ma2 = 1 # mass of agent 2

    Mbinv = Diagonal([1/mb; 1/mb])
    Ma1inv = Diagonal([1/ma1; 1/ma1])
    Ma2inv = Diagonal([1/ma2; 1/ma2])

    # body 1
    z = x[1:2]
    ż = x[3:4]

    # agent 1
    y1 = x[5:6]
    ẏ1 = x[7:8]

    # agent 2
    y2 = x[9:10]
    ẏ2 = x[11:12]

    # constraint force
    fz1 = u[1:2]
    fz2 = u[3:4]

    # body 1 control
    uy1 = u[5:6]
    fy1 = u[7:8]

    # body 2 control
    uy2 = u[9:10]
    fy2 = u[11:12]

    ## implicit euler

    # mass update
    x_[3:4] = ż + Δt*Mbinv*(fz1 + fz2 -mb*g)
    x_[1:2] = z + Δt*x_[3:4]

    # body 1 update
    x_[7:8] = ẏ1 + Δt*Ma1inv*(uy1 + fy1 -ma1*g)
    x_[5:6] = y1 + Δt*x_[7:8]

    #body 2 update
    x_[11:12] = ẏ2 + Δt*Ma2inv*(uy2 + fy2 -ma2*g)
    x_[9:10] = y2 + Δt*x_[11:12]

    return nothing
end

nb,mb = 4,4
na1,ma1 = 4,4
na2,ma2 = 4,4
N = nb+na1+na2
M = mb+ma1+ma2
model_a2_m1 = Model(agents2_mass1_2D!,N,M)


function agents3_mass1_3D!(x_::AbstractArray,x::AbstractArray,u::AbstractArray,Δt=0.1)::Nothing
    g = [0.;0.;9.81]
    mb = 1 # mass of load
    ma1 = 1 # mass of agent 1
    ma2 = 1 # mass of agent 2
    ma3 = 1 # mass of agent 3

    Mbinv = Diagonal(1/mb*ones(3))
    Ma1inv = Diagonal(1/ma1*ones(3))
    Ma2inv = Diagonal(1/ma2*ones(3))
    Ma3inv = Diagonal(1/ma3*ones(3))

    # body 1
    z = x[1:3]
    ż = x[4:6]

    # agent 1
    y1 = x[7:9]
    ẏ1 = x[10:12]

    # agent 2
    y2 = x[13:15]
    ẏ2 = x[16:18]

    # agent 2
    y3 = x[19:21]
    ẏ3 = x[22:24]


    # constraint force
    fz1 = u[1:3]
    fz2 = u[4:6]
    fz3 = u[7:9]

    # body 1 control
    uy1 = u[10:12]
    fy1 = u[13:15]

    # body 2 control
    uy2 = u[16:18]
    fy2 = u[19:21]

    # body 3 control
    uy3 = u[22:24]
    fy3 = u[25:27]

    ## implicit euler
    # mass update
    x_[4:6] = ż + Δt*Mbinv*(fz1 + fz2 -mb*g)
    x_[1:3] = z + Δt*x_[4:6]

    # body 1 update
    x_[10:12] = ẏ1 + Δt*Ma1inv*(uy1 + fy1 -ma1*g)
    x_[7:9] = y1 + Δt*x_[10:12]

    #body 2 update
    x_[16:18] = ẏ2 + Δt*Ma2inv*(uy2 + fy2 -ma2*g)
    x_[13:15] = y2 + Δt*x_[16:18]

    #body 3 updat6
    x_[22:24] = ẏ3 + Δt*Ma3inv*(uy3 + fy3 -ma3*g)
    x_[19:21] = y3 + Δt*x_[22:24]

    return nothing
end

nb,mb = 6,9
na1,ma1 = 6,6
na2,ma2 = 6,6
na3,ma3 = 6,6
N = nb+na1+na2+na3
M = mb+ma1+ma2+ma3
model_a3_m1 = Model(agents3_mass1_3D!,N,M)
