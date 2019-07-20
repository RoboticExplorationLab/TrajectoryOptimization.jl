using BenchmarkTools, SNOPT7, Plots
using FileIO, MeshIO, GeometryTypes, CoordinateTransformations, MeshCat

# Quadrotor in Maze
T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    iterations=300)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-8,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-4)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    opts=Dict(:max_iter=>10000),
    feasibility_tolerance=1.0e-3)

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    feasibility_tolerance=1.0e-3,
    opts=Dict(:Iterations_limit=>500000,
        :Major_iterations_limit=>1000))

# ALTRO w/ Newton
prob_altro = copy(Problems.quadrotor_maze)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation_direct(p1)
X1 = to_array(p1.X)
plot(X1[1:3,:]',title="Quadrotor position (ALTRO)")
plot(p1.U,title="Quadrotor control (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = update_problem(copy(Problems.quadrotor_maze),model=Dynamics.quadrotor_euler) # get continuous time model
p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation_direct(p2)
X2 = to_array(p2.X)
plot(X2[1:3,:]',title="Quadrotor position (Ipopt)")
plot(p2.U,title="Quadrotor control (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = update_problem(copy(Problems.quadrotor_maze),model=Dynamics.quadrotor_euler) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
X3 = to_array(p3.X)
plot(X3[1:3,:]',title="Quadrotor position (SNOPT)")
plot(p3.U,title="Quadrotor control (SNOPT)")

# c_max plot
# Max constraint plot
t_pn = s1.stats[:time_al]
t_span_al = range(0,stop=s1.stats[:time_al],length=s1.solver_al.stats[:iterations])
t_span_pn = range(t_pn,stop=s1.stats[:time],length=s1.solver_pn.stats[:iterations]+1)
t_span = [t_span_al;t_span_pn[2:end]]
c_span = [s1.solver_al.stats[:c_max]...,s1.solver_pn.stats[:c_max]...]

p = plot(t_pn*ones(100),range(1.0e-9,stop=1.0,length=100),color=:red,linestyle=:dash,label="Projected Newton",width=2)

# note: make sure snopt.out was updated by running with verbose=true
snopt_res = parse_snopt_summary(joinpath(pwd(),"snopt.out"))
t_span_snopt = range(0,stop=s3.stats[:time],length=length(snopt_res[:c_max]))
p = plot!(t_span_snopt,snopt_res[:c_max],marker=:circle,yscale=:log10,ylim=[1.0e-9,1.0],color=:green,label="SNOPT")

p = plot!(t_span,c_span,title="Quadrotor Maze c_max",xlabel="time (s)",marker=:circle,color=:orange,width=2,yscale=:log10,ylim=[1.0e-9,1.0],label="ALTRO")

savefig(p,joinpath(pwd(),"examples/IROS_2019/quadrotor_maze_c_max.png"))


# Visualization
function plot_cylinder(vis,c1,c2,radius,mat,name="")
    geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
    setobject!(vis["cyl"][name],geom,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
end

function addcylinders!(vis,cylinders,height=1.5)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder(vis,[cyl[1],cyl[2],0],[cyl[1],cyl[2],height],cyl[3],MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0)),"cyl_$i")
    end
end

function animate_quadrotor_maze(prob)
    vis = Visualizer()
    open(vis)

    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.7
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling


    sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.25)) # trajectory points
    sphere_med = HyperSphere(Point3f0(0), convert(Float32,0.5));
    sphere_quad = HyperSphere(Point3f0(0), convert(Float32,2.0));


    obstacles = vis["obs"]
    traj = vis["traj"]
    robot = vis["robot"]
    setobject!(vis["robot"]["quad"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)));
    # setobject!(vis["robot"]["ball"],sphere_medium,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.5)));

    settransform!(vis["/Cameras/default"], compose(Translation(0., 72., 60.),LinearMap(RotX(pi/7.5)*RotZ(pi/2))))
    addcylinders!(vis,Problems.quadrotor_maze_objects,16.0)
    traj = vis["traj"]

    for i = 1:prob.N
        setobject!(vis["traj"]["t$i"],sphere_small,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))
        settransform!(vis["traj"]["t$i"], Translation(prob.X[i][1], prob.X[i][2], prob.X[i][3]))
    end

    anim = MeshCat.Animation(24)
    for i = 1:prob.N
        MeshCat.atframe(anim,vis,i) do frame
            settransform!(frame["robot"], compose(Translation(prob.X[i][1:3]...),LinearMap(Quat(prob.X[i][4:7]...))))
        end
    end
    MeshCat.setanimation!(vis,anim)
end

function ghost_quadrotor_maze(prob)
    vis = Visualizer()
    open(vis)

    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.7
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling


    sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.25)) # trajectory points
    sphere_med = HyperSphere(Point3f0(0), convert(Float32,0.5));
    sphere_quad = HyperSphere(Point3f0(0), convert(Float32,2.0));


    obstacles = vis["obs"]
    traj = vis["traj"]
    robot = vis["robot"]
    # setobject!(vis["robot"]["quad"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)));
    # setobject!(vis["robot"]["ball"],sphere_medium,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.5)));

    settransform!(vis["/Cameras/default"], compose(Translation(0., 72., 60.),LinearMap(RotX(pi/7.5)*RotZ(pi/2))))
    addcylinders!(vis,Problems.quadrotor_maze_objects,16.0)
    traj = vis["traj"]

    for i = 1:prob.N
        setobject!(vis["traj"]["t$i"],sphere_small,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))
        settransform!(vis["traj"]["t$i"], Translation(prob.X[i][1], prob.X[i][2], prob.X[i][3]))
    end

    traj_idx = [1;12;20;30;40;50;prob.N]
    n_robots = length(traj_idx)
    for i = 1:n_robots
        robot = vis["robot_$i"]
        setobject!(vis["robot_$i"]["quad"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
        settransform!(vis["robot_$i"], compose(Translation(prob.X[traj_idx[i]][1:3]...),LinearMap(Quat(prob.X[traj_idx[i]][4:7]...))))    end
end

ghost_quadrotor_maze(p1)

animate_quadrotor_maze(p1)
