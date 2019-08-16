using LinearAlgebra
using TrajectoryOptimization
const TO = TrajectoryOptimization
using Plots

include("3Q_1L_problem.jl")
include("batch_methods.jl")

num_lift = 2
x0 = [0,  0.0,  0.3]
xf = [6.0, 0.0, 0.3]

probs = Problem{Float64,Discrete}[]
prob_load = build_quad_problem(:load,x0,xf,true,false,num_lift)
for i = 1:num_lift
    push!(probs, build_quad_problem(i,x0,xf,true,false,num_lift))
end
prob_load.model.info[:radius]
N = probs[1].N
dt = probs[1].dt

r_lift = 0.275
r_load = 0.2

num_lift = length(probs)
n_batch = sum([probs[i].model.n for i = 1:num_lift]) + prob_load.model.n
m_batch = sum([probs[i].model.m for i = 1:num_lift])

actuated_models = [quadrotor_lift for i = 1:num_lift]
load_model = Dynamics.doubleintegrator3D

# create batch model
batch_model = gen_batch_model(actuated_models,load_model)
batch_model_d = midpoint(batch_model)

# initial and final conditions
x0_batch = vcat([probs[i].x0 for i = 1:num_lift]...,prob_load.x0)
xf_batch = vcat([probs[i].xf for i = 1:num_lift]...,prob_load.xf)


# initial controls
U0 = [vcat([probs[i].U[k] for i = 1:num_lift]...) for k = 1:N-1]


# load constraints
d = [norm(prob_load.xf[1:3] - probs[i].xf[1:3]) for i = 1:num_lift]
load_con = gen_batch_load_constraints(actuated_models,load_model,d)

# self collision constraints
self_col = gen_batch_self_collision_constraints(actuated_models,load_model,0.5)

# bound constraints
bnds = TO.GeneralConstraint[]
for k = 1:N
    x_min = Float64[]
    x_max = Float64[]
    u_min = Float64[]
    u_max = Float64[]
    for i = 1:num_lift
        if :bound in labels(probs[i].constraints[k])
            c = bounds(probs[i].constraints[k])[1]
            append!(x_min,c.x_min)
            append!(x_max,c.x_max)
            append!(u_min,c.u_min)
            append!(u_max,c.u_max)
        else
            append!(x_min,-Inf*ones(probs[i].model.n))
            append!(x_max,Inf*ones(probs[i].model.n))
            append!(u_min,-Inf*ones(probs[i].model.m))
            append!(u_max,Inf*ones(probs[i].model.m))
        end
    end
    if :bound in labels(prob_load.constraints[k])
        c = bounds(prob_load.constraints[k])[1]
        append!(x_min,c.x_min)
        append!(x_max,c.x_max)
    else
        append!(x_min,-Inf*ones(prob_load.model.n))
        append!(x_max,Inf*ones(prob_load.model.n))
    end

    push!(bnds,BoundConstraint(n_batch,m_batch,u_min=u_min,u_max=u_max,x_min=x_min,x_max=x_max))
end

# batch objective
_cost = TO.CostFunction[]
for k = 1:N
    Q_batch = zeros(0,0)
    q_batch = zeros(0)
    R_batch = zeros(0,0)
    r_batch = zeros(0)
    c_batch = 0.
    for i = 1:num_lift
        Q_batch = Diagonal(cat(Q_batch,probs[i].obj.cost[k].Q,dims=(1,2)))
        q_batch = vcat(q_batch,probs[i].obj.cost[k].q)
        R_batch = Diagonal(cat(R_batch,probs[i].obj.cost[k].R,dims=(1,2)))
        r_batch = vcat(r_batch,probs[i].obj.cost[k].r)
        c_batch += probs[i].obj.cost[k].c
    end
    Q_batch = Diagonal(cat(Q_batch,prob_load.obj.cost[k].Q,dims=(1,2)))
    q_batch = vcat(q_batch,prob_load.obj.cost[k].q)
    c_batch += prob_load.obj.cost[k].c
    push!(_cost,QuadraticCost(Q_batch,R_batch,zeros(m_batch,n_batch),q_batch,r_batch,c_batch))
end

batch_obj = Objective(_cost)
batch_constraints = Constraints(N)

batch_constraints[1] += bnds[1]
for k = 2:N-1
    batch_constraints[k] +=  bnds[k] #+ load_con + self_col
end
batch_constraints[N] += goal_constraint(xf_batch)

quad_batch = TrajectoryOptimization.Problem(batch_model_d, batch_obj,constraints=batch_constraints, x0=x0_batch, xf=xf_batch, N=N, dt=dt)
initial_controls!(quad_batch,U0)

@info "Solving batch problem"
verbose=false
opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=500,
	  max_cost_value=1e12)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr)#,

function solve_batch(prob,opts)
	_batch, _solver = solve(prob,opts) # solve without system level constraints
	for k = 2:N-1
		_batch.constraints[k] += load_con + self_col # add system level constraints
	end
	_batch_, _solver_ = solve(_batch,opts)

	return _batch_, _solver_
end

@btime sol, _solver = solve_batch(quad_batch, opts_al)

visualize = true
if visualize
	using MeshCat
	using GeometryTypes
	using CoordinateTransformations
	using FileIO
	using MeshIO
	using Plots

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
			setobject!(vis["agent$i"]["sphere"],HyperSphere(Point3f0(0), convert(Float32,r_lift)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.35)))
			setobject!(vis["agent$i"]["robot"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))

			cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
			setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
	    end
	    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

	    settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))


	    anim = MeshCat.Animation(convert(Int,floor(1/prob_load.dt)))
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
	visualize_batch_system(vis,sol,actuated_models,load_model)
end
