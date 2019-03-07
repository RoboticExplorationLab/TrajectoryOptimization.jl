
function double_integrator_dynamics!(ẋ,x,u)
    ẋ[1] = x[2]
    ẋ[2] = u[1]
end

n = 2
m = 1

model = Model(double_integrator_dynamics!,n,m)

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

# Constrained Double Integrators ()
n1,m1 = 4,2
n2,m2 = 4,0
mf = 2
N = n1+n2
M = m1 + m2 + mf

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
function double_integrator_constrained_system!(x_::AbstractArray,x::AbstractArray,u::AbstractArray)::Nothing

    m1 = 1 # mass of body 1
    m2 = 1 # mass of mass

    M1 = [m1 0; 0 m1]
    M2 = [m2 0; 0 m2]

    M1inv = [1/m1 0; 0 1/m1]
    M2inv = [1/m2 0; 0 1/m2]

    # body 1
    y = x[part_x.a1][1:2]
    ẏ = x[part_x.a1][3:4]

    # mass
    z = x[part_x.m][1:2]
    ż = x[part_x.m][3:4]

    # body 1 control
    uy = u[1:2]

    # constraint force
    fy = u[3]
    fz = u[4]

    # constraint Jacobians
    jy = 2*(y - z)'
    jz = -2*(y + z)'

    ## implicit euler
    # body 1 update
    x_[part_x.a1][1:2] = y + Δt*ẏ
    x_[part_x.a1][3:4] = ẏ + Δt*M1inv*(uy + jy'*fy)

    # mass update
    x_[part_x.m][1:2] = z + Δt*ż
    x_[part_x.m][3:4] = ż + Δt*M2inv*(jz'*fz)

    return nothing
end

model_admm = Model(double_integrator_constrained_system!,N,M)

x0 = [10;0;0;0
Q = (1e-2)*Diagonal(I,N)
Qf = 100.0*Diagonal(I,N)
R = (1e-2)*Diagonal(I,N)

# simulation
tf = 5.0

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

solver = Solver(model_admm,obj,dt=Δt,integration=:none)
