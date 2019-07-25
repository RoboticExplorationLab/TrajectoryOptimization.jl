using BenchmarkTools, Plots, SNOPT7
using MeshCatMechanisms, MeshCat, GeometryTypes

T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    iterations=300,
    live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=20,
    cost_tolerance=1.0e-6,
    cost_tolerance_intermediate=1.0e-5,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=0.01)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-5)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    feasibility_tolerance=max_con_viol)

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    feasibility_tolerance=max_con_viol,
    opts=Dict(:Iterations_limit=>100000,
        :Major_iterations_limit=>1000))

# ALTRO w/ Newton
prob_altro = copy(Problems.kuka_problem)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation_direct(p1)
plot(p1.X,title="Kuka state (ALTRO)")
plot(p1.U,title="Kuka control (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.kuka_obstacles)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.kuka)
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation_direct(p2)
plot(p2.X,title="Kuka state (Ipopt)")
plot(p2.U,title="Kuka control (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.kuka_obstacles)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.kuka) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
plot(p3.X,title="Kuka state (SNOPT)")
plot(p3.U,title="Kuka control (SNOPT)")

# Visualization
kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)
kuka_visuals = URDFVisuals(Dynamics.urdf_kuka)

function plot_sphere(vis::MechanismVisualizer,frame::CartesianFrame3D,center,radius,mat,name="")
    geom = HyperSphere(Point3f0(center), convert(Float32,radius))
    setelement!(vis,frame,geom,mat,name)
end

function plot_cylinder(vis::MechanismVisualizer,frame::CartesianFrame3D,c1,c2,radius,mat,name="")
    geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
    setelement!(vis,frame,geom,mat,name)
end

function addcircles!(vis,circles,robot)
    world = root_frame(robot)
    for (i,circle) in enumerate(circles)
        p = Point3D(world,collect(circle[1:3]))
        setelement!(vis,p,circle[4],"obs$i")
    end
end

function addcylinders!(vis,cylinders,robot,height=1.5,clr=MeshPhongMaterial(color=RGBA(0, 0, 1, 0.5)))
    world = root_frame(robot)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder(vis,world,[cyl[1],cyl[2],0],[cyl[1],cyl[2],height],cyl[3],clr,"cyl_$i")
    end
end

function visualize_kuka_obstacles(prob,circles_kuka,cylinders_kuka,kuka_points=false)
    N = length(prob.X)
    vis = Visualizer()
    open(vis)
    mvis = MechanismVisualizer(kuka, kuka_visuals, vis[:base])
    addcircles!(mvis,circles_kuka,kuka)
    addcylinders!(mvis,cylinders_kuka,kuka)

    if kuka_points
        plot=true
        bodies = [3,4,5,6]
        radii = [0.1,0.12,0.09,0.09]
        ee_body,ee_point = Dynamics.get_kuka_ee(kuka)
        ee_radii = 0.05
        body_collision = MeshPhongMaterial(color=RGBA(1, 0, 0, 0.25))

        points = Point3D[]
        frames = CartesianFrame3D[]
        for (idx,radius) in zip(bodies,radii)
            body = findbody(kuka,"iiwa_link_$idx")
            frame = default_frame(body)
            point = Point3D(frame,0.,0.,0.)
            plot ? plot_sphere(mvis,frame,0,radius,body_collision,"body_$idx") : nothing
            plot ? setelement!(mvis,point,radius,"body_$idx") : nothing
            push!(points,point)
            push!(frames,frame)
        end
    end

    q = [prob.X[k][1:convert(Int,prob.model.n/2)] for k = 1:N]
    t = range(0,stop=prob.tf,length=N)
    setanimation!(mvis,t,q)
end

visualize_kuka_obstacles(p2,Problems.kuka_obstacles_objects[1],Problems.kuka_obstacles_objects[2],true)
