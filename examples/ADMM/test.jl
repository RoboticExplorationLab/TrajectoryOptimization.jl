T = Float64

## assemble batch model
# params = (m=1.0,)
# params_load = (m=1.0,)
# _doubleintegrator3D = Model(double_integrator_3D_dynamics!,6,3,params)
# actuated_models = [_doubleintegrator3D,_doubleintegrator3D,_doubleintegrator3D]
# load_model = Model(double_integrator_3D_dynamics!,6,3,params_load)

actuated_models = [doubleintegrator3D,doubleintegrator3D,doubleintegrator3D]
num_act_models = length(actuated_models)
load_model = doubleintegrator3D

function gen_batch_model(actuated_models,load_model,n_slack=3)
    num_act_models = length(actuated_models)
    nn = zeros(Int,num_act_models)
    mm = zeros(Int,num_act_models)

    for i = 1:num_act_models
        nn[i] = actuated_models[i].n
        mm[i] = actuated_models[i].m
    end

    nn_tol = sum(nn)
    mm_tol = sum(mm)
    n_batch = nn_tol + load_model.n
    m_batch = mm_tol + n_slack*num_act_models

    function batch_dynamics!(ẋ,x,u)
        n_shift = 0
        m_shift = 0
        m_load_shift = copy(mm_tol)

        u_load_tol = zeros(eltype(u),3)

        # update actuated models
        for i = 1:num_act_models
            x_idx = n_shift .+ (1:nn[i])
            u_idx = m_shift .+ (1:mm[i])
            u_load = u[m_load_shift .+ (1:n_slack)]

            actuated_models[i].f(view(ẋ,x_idx), x[x_idx], u[u_idx] + u_load)

            n_shift += nn[i]
            m_shift += mm[i]
            m_load_shift += n_slack
            u_load_tol += u_load
        end

        # update load
        x_load_idx = (nn_tol .+ (1:load_model.n))
        load_model.f(view(ẋ,x_load_idx),x[x_load_idx],-1.0*u_load_tol)

        return nothing
    end

    Model(batch_dynamics!,n_batch,m_batch)
end

batch_model = gen_batch_model(actuated_models,load_model)

# batch_model.f(zeros(24),zeros(24),zeros(18))
# batch_model_d = rk3(batch_model)
# batch_model_d.f(zeros(24),zeros(24),zeros(18),0.5)
#
# F = zeros(24,24+18)
# batch_model.∇f(F,zeros(24),zeros(18))
# Fd = zeros(24,24+18+1)
# batch_model_d.∇f(Fd,zeros(24),zeros(18),1.0)

r_act = [0.2, 0.2, 0.2]
r_load = 0.2

function gen_batch_load_constraints(actuated_models,load_model,d,n_slack=3)
    num_act_models = length(actuated_models)
    nn = zeros(Int,num_act_models)
    mm = zeros(Int,num_act_models)

    for i = 1:num_act_models
        nn[i] = actuated_models[i].n
        mm[i] = actuated_models[i].m
    end

    nn_tol = sum(nn)
    n_batch = nn_tol + load_model.n
    m_batch = sum(mm) + n_slack*num_act_models
    idx_load_pos = (nn_tol .+ (1:load_model.n))[1:n_slack]

    function con(c,x,u=zeros(m_batch))
        n_shift = 0
        x_load_pos = x[idx_load_pos]

        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:n_slack]
            x_pos = x[idx_pos]
            c[i] = norm(x_pos - x_load_pos)^2 - d[i]^2
            n_shift += nn[i]
        end
    end

    function ∇con(C,x,u=zeros(m_batch))
        n_shift = 0
        x_load_pos = x[idx_load_pos]

        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:n_slack]
            x_pos = x[idx_pos]
            dif = x_pos - x_load_pos
            C[i,idx_pos] = 2*dif
            C[i,idx_load_pos] = -2*dif
            n_shift += nn[i]
        end
    end

    Constraint{Equality}(con,∇con,n_batch,m_batch,num_act_models,:load)
end

function gen_batch_self_collision_constraints(actuated_models,load_model,r_act,n_slack=3)
    num_act_models = length(actuated_models)
    nn = zeros(Int,num_act_models)
    mm = zeros(Int,num_act_models)

    for i = 1:num_act_models
        nn[i] = actuated_models[i].n
        mm[i] = actuated_models[i].m
    end

    nn_tol = sum(nn)
    n_batch = nn_tol + load_model.n
    m_batch = sum(mm) + n_slack*num_act_models

    p_con = 0
    for i = 1:num_act_models
        if i < num_act_models
            for j = (i+1):num_act_models
                p_con += 1
            end
        end
    end

    function col_con(c,x,u=zeros(m_batch))
        n_shift = 0
        p_shift = 1
        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:n_slack]
            x_pos = x[idx_pos]
            n_shift2 = n_shift + nn[i]
            if i < num_act_models
                for j = (i+1):num_act_models
                    idx_pos2 = (n_shift2 .+ (1:nn[j]))[1:n_slack]
                    x_pos2 = x[idx_pos2]
                    c[p_shift] = (r_act[i] + r_act[j])^2 - norm(x_pos - x_pos2)^2
                    n_shift2 += nn[j]
                    p_shift += 1
                end
            end
            n_shift += nn[i]
        end
        @assert p_shift-1 == p_con
    end

    function ∇col_con(C,x,u=zeros(m_batch))
        n_shift = 0
        p_shift = 1
        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:n_slack]
            x_pos = x[idx_pos]

            n_shift2 = n_shift + nn[i]
            if i < num_act_models
                for j = (i+1):num_act_models
                    idx_pos2 = (n_shift2 .+ (1:nn[j]))[1:n_slack]
                    x_pos2 = x[idx_pos2]
                    dif = x_pos - x_pos2
                    C[p_shift,idx_pos] = -2*dif
                    C[p_shift,idx_pos2] = 2*dif
                    n_shift2 += nn[j]
                    p_shift += 1
                end
            end
            n_shift += nn[i]
        end
        @assert p_shift-1 == p_con
    end

    Constraint{Inequality}(col_con,∇col_con,n_batch,m_batch,p_con,:col)
end

self_col = gen_batch_self_collision_constraints(actuated_models,load_model,r_act)
# pp = zeros(3)
# PP = zeros(3,n_batch+m_batch)
# ppN = zeros(3)
# PPN = zeros(3,n_batch)
#
# self_col.c(pp,rand(n_batch),m_batch)
# self_col.c(ppN,rand(n_batch))
# self_col.∇c(PP,rand(n_batch),m_batch)
# self_col.∇c(PPN,rand(n_batch))
#
# pp
# ppN
# PP
# PPN

n_slack = 3
nn = zeros(Int,num_act_models)
mm = zeros(Int,num_act_models)

for i = 1:num_act_models
    nn[i] = actuated_models[i].n
    mm[i] = actuated_models[i].m
end

nn_tol = sum(nn)
mm_tol = sum(mm)
n_batch = nn_tol + load_model.n
m_batch = mm_tol + n_slack*num_act_models

# _cyl = ((5.,.75,0.5),(6.,.75,0.5),(4.,.75,0.5),(5.,1.,0.5),(6.,1.,0.5),(4.,1.,0.5),(5.,-.75,0.5),(6.,-.75,0.5),(4.,-.75,0.5),(5.,-1.,0.5),(6.,-1.,0.5),(4.,-1.,0.5))
r_cylinder = 0.5
_cyl = []
l1 = 3

for i = range(4,stop=5,length=l1)
    push!(_cyl,(i, .75,r_cylinder))
end
for i = range(4,stop=5,length=l1)
    push!(_cyl,(i, -.75,r_cylinder))
end
for i = range(4,stop=5,length=l1)
    push!(_cyl,(i, 1.,r_cylinder))
end
for i = range(4,stop=5,length=l1)
    push!(_cyl,(i, -1.,r_cylinder))
end

function cI_cylinder(c,x,u)
    c_shift = 1
    n_slack = 3
    for p = 1:length(_cyl)
        n_shift = 0
        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:3]
            c[c_shift] = circle_constraint(x[idx_pos],_cyl[p][1],_cyl[p][2],_cyl[p][3] + r_act[i])
            c_shift += 1
            n_shift += nn[i]
        end
        c[c_shift] = circle_constraint(x[nn_tol .+ (1:load_model.n)],_cyl[p][1],_cyl[p][2],_cyl[p][3] + r_load)
        c_shift += 1
    end
end
cyl = Constraint{Inequality}(cI_cylinder,n_batch,m_batch,(num_act_models+1)*length(_cyl),:cyl)

shift_ = zeros(n)
shift_[1:3] = [0.0;0.0;1.0]
scaling = 1.
x10 = zeros(n)
x10[1:3] = scaling*[sqrt(8/9);0.;4/3]
x10 += shift_
x20 = zeros(n)
x20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
x20 += shift_
x30 = zeros(n)
x30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
x30 += shift_
xload0 = zeros(n)
xload0 += shift_

x0_batch = [x10;x20;x30;xload0]

_shift = zeros(n)
_shift[1:3] = [10.0;0.0;0.0]

norm(xload0[1:3]-x10[1:3])
norm(xload0[1:3]-x20[1:3])
norm(xload0[1:3]-x30[1:3])

xloadf = xload0 + _shift
x1f = copy(x10) + _shift
x2f = copy(x20) + _shift
x3f = copy(x30) + _shift
xf_batch = [x1f;x2f;x3f;xloadf]

d1 = norm(xloadf[1:3]-x1f[1:3])
d2 = norm(xloadf[1:3]-x2f[1:3])
d3 = norm(xloadf[1:3]-x3f[1:3])

d = [d1, d2, d3]
load_con = gen_batch_load_constraints(actuated_models,load_model,d)


# costs
Q = 1.0*Diagonal(I,n)
Qf = 1.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
Q_batch = Diagonal(cat(Q,Q,Q,Q,dims=(1,2)))
R_batch = Diagonal(cat(R,R,R,Diagonal(1.0e-6*ones(9)),dims=(1,2)))
Qf_batch = Diagonal(cat(Qf,Qf,Qf,Qf,dims=(1,2)))

N = 51
dt = 0.1

u_lim = Inf*ones(18)
u_lim[1:9] .= 15.
bnd = BoundConstraint(n_batch,m_batch,u_min=-1.0*u_lim,u_max=u_lim)

batch_obj = LQRObjective(Q_batch,R_batch,Qf_batch,xf_batch,N)
batch_constraints = Constraints([load_con],N)
for k = 1:N-1
    batch_constraints[k] += bnd + cyl + self_col
end
batch_constraints[N] += goal_constraint(xf_batch)

doubleintegrator_batch = TrajectoryOptimization.Problem(batch_model_d, batch_obj,constraints=batch_constraints, x0=x0_batch, xf=xf_batch, N=N, dt=dt)
initial_controls!(doubleintegrator_batch, 0.01*rand(batch_model.m,N-1))
# initial_controls!(doubleintegrator_batch, zeros(batch_model.m,N-1))

# rollout!(doubleintegrator_batch)
plot(doubleintegrator_batch.X)
plot(doubleintegrator_batch.U)

solve!(doubleintegrator_batch,ALTROSolverOptions{T}(verbose=true))
plot(doubleintegrator_batch.X)
max_violation(doubleintegrator_batch)

using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using Plots

vis = Visualizer()
open(vis)

# geometries
# sphere_small = HyperSphere(Point3f0(0), convert(Float32,r_int)) # trajectory points
# sphere_medium = HyperSphere(Point3f0(0), convert(Float32,1.0))

# Set camera location
settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))

function cable_transform(y,z)
    v1 = [0,0,1]
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1,v2)
    ang = acos(v1'v2)
    R = AngleAxis(ang,ax...)
    compose(Translation(z),LinearMap(R))
end

function plot_cylinder(vis,c1,c2,radius,mat,name="")
    geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
    setobject!(vis["cyl"][name],geom,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
end

function addcylinders!(vis,cylinders,height=1.5)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder(vis,[cyl[1],cyl[2],0],[cyl[1],cyl[2],height],cyl[3],MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0)),"cyl_$i")
    end
end

function visualize_batch_system(vis,prob,actuated_models,load_model,n_slack=3)
    num_act_models = length(actuated_models)
    nn = zeros(Int,num_act_models)
    mm = zeros(Int,num_act_models)

    for i = 1:num_act_models
        nn[i] = actuated_models[i].n
        mm[i] = actuated_models[i].m
    end

    nn_tol = sum(nn)

    # intialized system
    for i = 1:num_act_models
        setobject!(vis["agent$i"],HyperSphere(Point3f0(0), convert(Float32,r_act[i])) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))

        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    addcylinders!(vis,_cyl,3.)

    anim = MeshCat.Animation(24)
    for k = 1:prob.N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob.X[k][nn_tol .+ (1:load_model.n)][1:n_slack]
            n_shift = 0
            for i = 1:num_act_models
                x_idx = n_shift .+ (1:actuated_models[i].n)
                x_ = prob.X[k][x_idx][1:n_slack]
                settransform!(frame["cable"]["$i"], cable_transform(x_,x_load))
                settransform!(frame["agent$i"], Translation(x_...))
                n_shift += actuated_models[i].n
            end
            settransform!(frame["load"], Translation(x_load...))
        end
    end
    MeshCat.setanimation!(vis,anim)
end

visualize_batch_system(vis,doubleintegrator_batch,actuated_models,load_model)

# plot(doubleintegrator_batch.U,1:3)
#
# doubleintegrator_batch.constraints[N]
#
# a = String(:string)
# a[1:3]
# cc = pop!(doubleintegrator_batch.constraints[N],:load)
