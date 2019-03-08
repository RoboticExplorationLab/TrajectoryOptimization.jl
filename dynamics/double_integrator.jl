
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
