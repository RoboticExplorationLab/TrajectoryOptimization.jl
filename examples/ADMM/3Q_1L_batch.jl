include(joinpath(dirname(@__FILE__),"../../dynamics/quaternions.jl"))

function quadrotor_lift_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,params) where T
      #TODO change concatentations to make faster!
      # Quaternion representation
      # Modified from D. Mellinger, N. Michael, and V. Kumar,
      # "Trajectory generation and control for precise aggressive maneuvers with quadrotors",
      # In Proceedings of the 12th International Symposium on Experimental Robotics (ISER 2010), 2010.

      ## States: X ∈ R^13; q = [s;v]
      # x
      # y
      # z
      # q0
      # q1
      # q2
      # q3
      # xdot
      # ydot
      # zdot
      # omega1
      # omega2
      # omega3

      # x = X[1:3]
      q = normalize(Quaternion(view(x,4:7)))
      # q = view(x,4:7)
      # normalize!(q)
      v = view(x,8:10)
      omega = view(x,11:13)

      # Parameters
      m = params[:m] # mass
      J = params[:J] # inertia matrix
      Jinv = params[:Jinv] # inverted inertia matrix
      g = params[:gravity] # gravity
      L = params[:motor_dist] # distance between motors

      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      f_load = u[5:7]

      kf = params[:kf]; # 6.11*10^-8;
      F1 = kf*w1;
      F2 = kf*w2;
      F3 = kf*w3;
      F4 = kf*w4;
      F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame

      km = params[:km]
      M1 = km*w1;
      M2 = km*w2;
      M3 = km*w3;
      M4 = km*w4;
      tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

      ẋ[1:3] = v # velocity in world frame
      # ẋ[4:7] = 0.5*qmult(q,[0;omega]) #quaternion derivative
      ẋ[4:7] = SVector(0.5*q*Quaternion(zero(x[1]), omega...))
      ẋ[8:10] = g + (1/m)*(q*F + f_load) #acceleration in world frame
      ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
      return tau, omega, J, Jinv
end

quad_params = (m=0.85,
             J=SMatrix{3,3}(Diagonal([0.0023, 0.0023, 0.004])),
             Jinv=SMatrix{3,3}(Diagonal(1.0./[0.0023, 0.0023, 0.004])),
             gravity=SVector(0,0,-9.81),
             motor_dist=0.1750,
             kf=1.0,
             km=0.0245)

quadrotor_lift = Model(quadrotor_lift_dynamics!, 13, 7, quad_params)

actuated_models = [quadrotor_lift, quadrotor_lift, quadrotor_lift]
num_act_models = length(actuated_models)
load_model = Dynamics.doubleintegrator3D

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
    m_batch = mm_tol #+ n_slack*num_act_models

    function batch_dynamics!(ẋ,x,u)
        n_shift = 0
        m_shift = 0
        # m_load_shift = copy(mm_tol)

        u_load_tol = zeros(eltype(u),3)

        # update actuated models
        for i = 1:num_act_models
            x_idx = n_shift .+ (1:nn[i])
            u_idx = m_shift .+ (1:mm[i])
            u_load = u[u_idx][end-(n_slack-1):end]

            actuated_models[i].f(view(ẋ,x_idx), x[x_idx], u[u_idx])

            n_shift += nn[i]
            m_shift += mm[i]
            # m_load_shift += n_slack
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

# batch_model.f(zeros(45),zeros(45),zeros(21))
batch_model_d = midpoint(batch_model)
# batch_model_d.f(zeros(45),zeros(45),zeros(21),0.5)
# #
# F = zeros(45,45+21)
# batch_model.∇f(F,zeros(45),zeros(21))
# Fd = zeros(45,45+21+1)
# batch_model_d.∇f(Fd,zeros(45),zeros(21),1.0)

r_act = [0.275, 0.275, 0.275]
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
    m_batch = sum(mm) #+ n_slack*num_act_models
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
    m_batch = sum(mm)# + n_slack*num_act_models

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
num_lift = 3
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
m_batch = mm_tol #+ n_slack*num_act_models

# _cyl = ((5.,.75,0.5),(6.,.75,0.5),(4.,.75,0.5),(5.,1.,0.5),(6.,1.,0.5),(4.,1.,0.5),(5.,-.75,0.5),(6.,-.75,0.5),(4.,-.75,0.5),(5.,-1.,0.5),(6.,-1.,0.5),(4.,-1.,0.5))
# r_cylinder = 0.5
# _cyl = []
# l1 = 3
#
# for i = range(4,stop=5,length=l1)
#     push!(_cyl,(i, .75,r_cylinder))
# end
# for i = range(4,stop=5,length=l1)
#     push!(_cyl,(i, -.75,r_cylinder))
# end
# for i = range(4,stop=5,length=l1)
#     push!(_cyl,(i, 1.,r_cylinder))
# end
# for i = range(4,stop=5,length=l1)
#     push!(_cyl,(i, -1.,r_cylinder))
# end

r_cylinder = 0.5

_cyl = []
# l1 = 6

push!(_cyl,(5.,1.,r_cylinder))
push!(_cyl,(5.,-1.,r_cylinder))
# push!(_cyl,(5.,1.25,r_cylinder))
# push!(_cyl,(5.,-1.25,r_cylinder))
# push!(_cyl,(5.,1.5,r_cylinder))
# push!(_cyl,(5.,-1.5,r_cylinder))
# push!(_cyl,(5.,1.75,r_cylinder))
# push!(_cyl,(5.,-1.75,r_cylinder))

function cI_cylinder(c,x,u)
    c_shift = 1
    n_slack = 3
    for p = 1:length(_cyl)
        n_shift = 0
        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:3]
            c[c_shift] = circle_constraint(x[idx_pos],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*r_act[i])
            c_shift += 1
            n_shift += nn[i]
        end
        c[c_shift] = circle_constraint(x[nn_tol .+ (1:load_model.n)],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*r_load)
        c_shift += 1
    end
end
cyl = Constraint{Inequality}(cI_cylinder,n_batch,m_batch,(num_act_models+1)*length(_cyl),:cyl)

n = quadrotor_lift.n
m = quadrotor_lift.m

n_load = Dynamics.doubleintegrator3D.n
m_load = Dynamics.doubleintegrator3D.m
shift_ = zeros(n)
shift_[1:3] = [0.0;0.0;0.25]
scaling = 1.
x10 = zeros(n)
x10[4] = 1.
x10[1:3] = scaling*[sqrt(8/9);0.;4/3]
x10 += shift_
x20 = zeros(n)
x20[4] = 1.
x20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
x20 += shift_
x30 = zeros(n)
x30[4] = 1.
x30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
x30 += shift_
xload0 = zeros(n_load)
xload0[3] = 4/6
xload0[1:3] += shift_[1:3]

x0_batch = [x10;x20;x30;xload0]

_shift = zeros(n)
_shift[1:3] = [7.5;0.0;0.0]

norm(xload0[1:3]-x10[1:3])
norm(xload0[1:3]-x20[1:3])
norm(xload0[1:3]-x30[1:3])

xloadf = zeros(n_load)
xloadf[1:3] = xload0[1:3] + _shift[1:3]
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

q_diag = ones(n)

q_diag1 = copy(q_diag)
q_diag2 = copy(q_diag)
q_diag3 = copy(q_diag)
q_diag1[1] = 1.0
q_diag2[1] = 1.5e-2
q_diag3[1] = 1.0e-3

r_diag = ones(m)
r_diag[1:4] .= 1.0e-6
r_diag[5:7] .= 1.0e-6
Q_lift = [1.0e-1*Diagonal(q_diag1), 1.0e-1*Diagonal(q_diag2), 1.0e-1*Diagonal(q_diag3)]
Qf_lift = [1000.0*Diagonal(q_diag), 1000.0*Diagonal(q_diag), 1000.0*Diagonal(q_diag)]
R_lift = Diagonal(r_diag)
Q_load = 0.0*Diagonal(I,n_load)
Qf_load = 0.0*Diagonal(I,n_load)
R_load = 1.0e-6*Diagonal(I,m_load)


Q_batch = Diagonal(cat(Q_lift[1],Q_lift[2],Q_lift[3],Q_load,dims=(1,2)))
R_batch = Diagonal(cat(R_lift,R_lift,R_lift,dims=(1,2)))
Qf_batch = Diagonal(cat(Qf_lift[1],Qf_lift[2],Qf_lift[3],Qf_load,dims=(1,2)))

N = 101
dt = 0.1

u_lim_l = -Inf*ones(m_batch)
u_lim_u = Inf*ones(m_batch)
u_lim_l[1:4] .= 0.
u_lim_l[8:11] .= 0.
u_lim_l[15:18] .= 0.

u_lim_u[1:4] .= 12/4.
u_lim_u[8:11] .= 12/4.
u_lim_u[15:18] .= 12/4.

x_lim_l = -Inf*ones(n_batch)
x_lim_l[3] = 0.
x_lim_l[13+3] = 0.
x_lim_l[26+3] = 0.

bnd1 = BoundConstraint(n_batch,m_batch,u_min=u_lim_l,u_max=u_lim_u)
bnd2 = BoundConstraint(n_batch,m_batch,u_min=u_lim_l,u_max=u_lim_u,x_min=x_lim_l)

batch_obj = LQRObjective(Q_batch,R_batch,Qf_batch,xf_batch,N)
batch_constraints = Constraints(N)

batch_constraints[1] += bnd1
for k = 2:N-1
    batch_constraints[k] += cyl + self_col + bnd2 + load_con
end
batch_constraints[N] += goal_constraint(xf_batch)

quad_batch = TrajectoryOptimization.Problem(batch_model_d, batch_obj,constraints=batch_constraints, x0=x0_batch, xf=xf_batch, N=N, dt=dt)

u_ = [9.81*(quad_params.m)/4.;9.81*(quad_params.m)/4.;9.81*(quad_params.m)/4.;9.81*(quad_params.m)/4.]
u_load = [0.;0.;-9.81*0.35/num_lift]
initial_controls!(quad_batch, [[u_;u_load;u_;u_load;u_;u_load] for k = 1:N-1])
# initial_controls!(doubleintegrator_batch, zeros(batch_model.m,N-1))

# rollout!(quad_batch)
# plot(quad_batch.X,1:3)
# plot(quad_batch.U)

@info "Solving batch problem"
verbose=false
opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=500, max_cost_value=1e12)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    penalty_initial=1.0e-3)
    # penalty_scaling=2.0)

@time solve!(quad_batch,opts_al)

visualize = false
if visualize
	# plot(quad_batch.X,1:3)
	# max_violation(quad_batch)

	using MeshCat
	using GeometryTypes
	using CoordinateTransformations
	using FileIO
	using MeshIO
	using Plots

	# geometries
	# sphere_small = HyperSphere(Point3f0(0), convert(Float32,r_int)) # trajectory points
	# sphere_medium = HyperSphere(Point3f0(0), convert(Float32,1.0))


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

	    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
	    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
	    obj = joinpath(urdf_folder, "quadrotor_base.obj")

	    quad_scaling = 0.1
	    robot_obj = FileIO.load(obj)
	    robot_obj.vertices .= robot_obj.vertices .* quad_scaling

	    # intialized system
	    for i = 1:num_act_models
		setobject!(vis["agent$i"]["sphere"],HyperSphere(Point3f0(0), convert(Float32,r_act[i])) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.35)))
		setobject!(vis["agent$i"]["robot"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))

		cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
		setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
	    end
	    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

	    addcylinders!(vis,_cyl,3.)

	    settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))


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
			settransform!(frame["agent$i"], compose(Translation(x_...),LinearMap(Quat(prob.X[k][x_idx][4:7]...))))

			n_shift += actuated_models[i].n
		    end
		    settransform!(frame["load"], Translation(x_load...))
		end
	    end
	    MeshCat.setanimation!(vis,anim)
	end

	vis = Visualizer()
	open(vis)
	visualize_batch_system(vis,quad_batch,actuated_models,load_model)
end
