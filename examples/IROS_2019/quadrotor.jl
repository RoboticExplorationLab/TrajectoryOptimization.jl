using BenchmarkTools, SNOPT7, Plots
using FileIO, MeshIO, GeometryTypes, CoordinateTransformations, MeshCat

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
    projected_newton_tolerance=1.0e-3)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    opts=Dict(:max_iter=>10000),
    feasibility_tolerance=max_con_viol)

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    feasibility_tolerance=max_con_viol,
    opts=Dict(:Iterations_limit=>500000,
        :Major_iterations_limit=>1000))

# ALTRO w/ Newton
prob_altro = copy(Problems.quadrotor)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation_direct(p1)
X1 = to_array(p1.X)
plot(X1[1:3,:]',title="Quadrotor position (ALTRO)")
plot(p1.U,title="Quadrotor control (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.quadrotor)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.quadrotor_euler) # get continuous time model
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation_direct(p2)
X2 = to_array(p2.X)
plot(X2[1:3,:]',title="Quadrotor position (Ipopt)")
plot(p2.U,title="Quadrotor control (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.quadrotor)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.quadrotor_euler) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
X3 = to_array(p3.X)
plot(X3[1:3,:]',title="Quadrotor position (SNOPT)")
plot(p3.U,title="Quadrotor control (SNOPT)")

# Visualization
vis = Visualizer()
open(vis)
function animate_quadrotor_line(vis,prob)

    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor.obj")

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

animate_quadrotor_line(p1)
