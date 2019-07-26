
using BenchmarkTools, SNOPT7, Plots
using FileIO, MeshIO, GeometryTypes, CoordinateTransformations, MeshCat
using Colors, Plots

#################################
#        VISUALIZATION          #
#################################
function plot_cylinder(vis,c1,c2,radius,mat,name="")
    geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
    setobject!(vis["cyl"][name],geom,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
end

function addcylinders!(vis,cylinders,height=1.5)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder(vis,[cyl[1],cyl[2],0],[cyl[1],cyl[2],height],cyl[3],MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0)),"cyl_$i")
    end
end

function addspheres!(vis,spheres)
    for (i,sphere) in enumerate(spheres)
        geom = HyperSphere(Point3f0(sphere[1:3]), convert(Float32,sphere[4]))
        setobject!(vis["sph"]["sph_$i"],geom,MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0)))
    end
end

function plot_scene(vis=Visualizer())
    addcylinders!(vis,Problems.cyl_obs,20.0)
    addspheres!(vis,Problems.sph_obs)
end

function plot_quad(x0=prob.x0, xf=prob.xf)
    traj_folder = TrajectoryOptimization.root_dir()
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.7
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling

    setobject!(vis["robot"]["quad"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
    settransform!(vis["robot"], compose(Translation(x0[1:3]...),LinearMap(Quat(x0[4:7]...))))

    setobject!(vis["robot_goal"]["quad"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.3)))
    settransform!(vis["robot_goal"], compose(Translation(xf[1:3]...),LinearMap(Quat(xf[4:7]...))))

    settransform!(vis["/Cameras/default"], compose(Translation(-20., 72., 60.),LinearMap(RotX(pi/7.5)*RotZ(pi/2))))
end

function animate_quad!(vis, prob::Problem)
    anim = MeshCat.Animation(24)
    for i = 1:prob.N
        MeshCat.atframe(anim,vis,i) do frame
            settransform!(frame["robot"], compose(Translation(prob.X[i][1:3]...),LinearMap(Quat(prob.X[i][4:7]...))))
        end
    end
    MeshCat.setanimation!(vis,anim)
end

vis = Visualizer()
open(vis)
plot_scene()
plot_quad()

#################################
#      SOLVING THE PROBLEM      #
#################################
T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

prob = copy(Problems.quad_obs)

# iLQR
opts_ilqr = iLQRSolverOptions{T}(verbose=true,
    iterations=300)
@time r0, s0 = solve(prob, opts_ilqr)
animate_quad!(vis, r0)
plot(res.U)

# AL-iLQR
prob = copy(Problems.quad_obs)
opts_ilqr.verbose = false
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=true,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-3,
    constraint_tolerance=1e-4,
    penalty_scaling=10.,
    penalty_initial=0.1)
@time r1, s1 = solve(prob, opts_al)
plot(r1.U)
animate_quad!(vis, r1)

# ALTRO w/ Newton
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
@time r2, s2 = solve(prob, opts_altro)
plot(r2.U)
animate_quad!(vis, r2)
max_violation_direct(p2)

# Ipopt
opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    opts=Dict(:max_iter=>10000),
    feasibility_tolerance=max_con_viol)
r3, s3 = solve(prob, opts_ipopt)
plot(r3.U)
animate_quad!(vis, r3)

# SNOPT
opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=SNOPT7.Optimizer(),
    feasibility_tolerance=max_con_viol,
    opts=Dict(:Iterations_limit=>500000,
        :Major_iterations_limit=>1000))
r4, s4 = solve(prob, opts_snopt)
