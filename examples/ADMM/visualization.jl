using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using LinearAlgebra
import TrajectoryOptimization: AbstractSolver, solve_aula!

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
    setobject!(vis["cyl"][name],geom,mat)
end

function addcylinders!(vis,cylinders,height=1.5)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder(vis,[cyl[1],cyl[2],0],[cyl[1],cyl[2],height],cyl[3],MeshPhongMaterial(color=RGBA(1.0, 0, 0, 1.0)),"cyl_$i")
    end
end

function movecylinders!(vis, shifts)
    for (i,cyl) in enumerate(shifts)
        settransform!(vis["cyl"]["cyl_$i"], Translation(cyl[1], cyl[2], cyl[3]))
    end
end

function visualize_DI_lift_system(vis,sol)
    prob_lift = sol[2:end]
    prob_load = sol[1]
    r_lift = prob_lift[1].model.info[:radius]
    r_load = prob_load.model.info[:radius]
    visualize_DI_lift_system(vis,prob_lift,prob_load,r_lift,r_load,3)
end

function visualize_DI_lift_system(vis,prob_lift,prob_load,r_lift,r_load,n_slack=3)
    num_lift = length(prob_lift)
    d = [norm(prob_lift[i].x0[1:n_slack] - prob_load.x0[1:n_slack]) for i = 1:num_lift]
    _cyl = DI_obstacles()

    # camera angle
    settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))

    # intialize system
    for i = 1:num_lift
        setobject!(vis["lift$i"],HyperSphere(Point3f0(0), convert(Float32,r_lift)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))

        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    addcylinders!(vis,_cyl,3.)

    anim = MeshCat.Animation(convert(Int,floor(1.0/prob_lift[1].dt)))
    for k = 1:prob_lift[1].N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob_load.X[k][1:n_slack]
            for i = 1:num_lift
                x_lift = prob_lift[i].X[k][1:n_slack]
                settransform!(frame["cable"]["$i"], cable_transform(x_lift,x_load))
                settransform!(frame["lift$i"], Translation(x_lift...))
            end
            settransform!(frame["load"], Translation(x_load...))
        end
    end
    MeshCat.setanimation!(vis,anim)
    return anim
end

function visualize_quadrotor_lift_system(vis, probs; door=:middle, n_slack=3)
    prob_load = probs[1]
    prob_lift = probs[2:end]
    r_lift = prob_lift[1].model.info[:radius]::Float64
    r_load = prob_load.model.info[:radius]::Float64
    ceiling = bounds(prob_lift[1].obj.constraints[2])[1].x_max[3]


    num_lift = length(prob_lift)::Int
    d = [norm(prob_lift[i].x0[1:n_slack] - prob_load.x0[1:n_slack]) for i = 1:num_lift]

    if door != :false
        _cyl, = quad_obstacles()
        addcylinders!(vis, _cyl, ceiling)
        settransform!(vis["cyl"], Translation(0, door_location(door), 0))

        # # Pedestal
        xf_load = prob_load.xf
        mat = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
        plot_cylinder(vis, [xf_load[1], 0, 0], [xf_load[1], 0, xf_load[3] - r_load], 0.3, mat, "pedestal")
    end
    # camera angle
    # settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))

    # load in quad mesh
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.085
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling

    # intialize system
    for i = 1:num_lift
        # setobject!(vis["lift$i"]["sphere"],HyperSphere(Point3f0(0), convert(Float32,r_lift)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.25)))

        # setobject!(vis["lift$i"]["cyl_top"],Cylinder(Point3f0([0,0,-1*r_lift]),Point3f0([0,0,3*r_lift]),convert(Float32,r_lift)),MeshPhongMaterial(color=RGBA(1, 0, 0, 0.25)))
        # setobject!(vis["lift$i"]["cyl_bottom"],Cylinder(Point3f0([0,0,-3r_lift]),Point3f0([0,0,-1*r_lift]),convert(Float32,r_lift)),MeshPhongMaterial(color=RGBA(1, 0, 0, 0.25)))
        # setobject!(vis["lift$i"]["sphere"],HyperSphere(Point3f0(0), convert(Float32,r_lift)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.25)))

        setobject!(vis["lift$i"]["robot"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))

        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int,floor(1/prob_lift[1].dt)))
    for k = 1:prob_lift[1].N
        MeshCat.atframe(anim,vis,k) do frame
            plot_quad_scene(frame, k, probs)
        end
    end
    MeshCat.setanimation!(vis,anim)
    return anim
end

function plot_quad_scene(frame, k, probs)
    prob_load = probs[1]
    prob_lift = probs[2:end]

    # cables
    x_load = prob_load.X[k][1:n_slack]
    for i = 1:length(prob_lift)
        x_lift = prob_lift[i].X[k][1:n_slack]
        settransform!(frame["cable"]["$i"], cable_transform(x_lift,x_load))
        settransform!(frame["lift$i"], compose(Translation(x_lift...),LinearMap(Quat(prob_lift[i].X[k][4:7]...))))

    end
    settransform!(frame["load"], Translation(x_load...))
end


function visualize_door_change(vis, sol1, sol2, door1, door2, k_switch; n_slack=3)
    r_lift = prob_lift[1].model.info[:radius]::Float64
    r_load = prob_load.model.info[:radius]::Float64
    ceiling = bounds(sol[2].obj.constraints[2])[1].x_max[3]
    _cyl, = quad_obstacles()

    num_lift = length(sol) - 1
    d = [norm(sol1[i+1].x0[1:n_slack] - sol1[1].x0[1:n_slack]) for i = 1:num_lift]

    # # Pedestal
    xf_load = prob_load.xf
    mat = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
    plot_cylinder(vis, [xf_load[1], xf_load[2]*0, 0], [xf_load[1], xf_load[2]*0, xf_load[3] - r_load], 0.3, mat, "pedestal")

    # camera angle
    # settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))

    # load in quad mesh
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.085
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling

    # intialize system
    for i = 1:num_lift
        # setobject!(vis["lift$i"]["sphere"],HyperSphere(Point3f0(0), convert(Float32,r_lift)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.25)))

        # setobject!(vis["lift$i"]["cyl_top"],Cylinder(Point3f0([0,0,-1*r_lift]),Point3f0([0,0,3*r_lift]),convert(Float32,r_lift)),MeshPhongMaterial(color=RGBA(1, 0, 0, 0.25)))
        # setobject!(vis["lift$i"]["cyl_bottom"],Cylinder(Point3f0([0,0,-3r_lift]),Point3f0([0,0,-1*r_lift]),convert(Float32,r_lift)),MeshPhongMaterial(color=RGBA(1, 0, 0, 0.25)))
        # setobject!(vis["lift$i"]["sphere"],HyperSphere(Point3f0(0), convert(Float32,r_lift)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.25)))

        setobject!(vis["lift$i"]["robot"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))

        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    addcylinders!(vis, _cyl, ceiling)

    anim = MeshCat.Animation(convert(Int,floor(1/prob_lift[1].dt)))
    for k = 1:k_switch
        MeshCat.atframe(anim,vis,k) do frame
            plot_quad_scene(frame, k, sol1)
            settransform!(frame["cyl"], Translation(0, door_location(door1), 0))
        end
    end
    for k = 1:sol2[1].N
        MeshCat.atframe(anim,vis,k+k_switch) do frame
            plot_quad_scene(frame, k, sol2)
            settransform!(frame["cyl"], Translation(0, door_location(door2), 0))
        end
    end
    MeshCat.setanimation!(vis,anim)
    return anim
end

function ghost_quadrotor_lift_system(vis, probs, index, tp=[]; door=:middle, n_slack=3)
    prob_load = probs[1]
    prob_lift = probs[2:end]
    r_lift = prob_lift[1].model.info[:radius]::Float64
    r_load = prob_load.model.info[:radius]::Float64
    ceiling = bounds(prob_lift[1].obj.constraints[2])[1].x_max[3]
    _cyl, = quad_obstacles()

    num_lift = length(prob_lift)::Int
    d = [norm(prob_lift[i].x0[1:n_slack] - prob_load.x0[1:n_slack]) for i = 1:num_lift]

    addcylinders!(vis, _cyl, ceiling)
    settransform!(vis["cyl"], Translation(0, door_location(door), 0))

    # # Pedestal
    xf_load = prob_load.xf
    mat = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
    plot_cylinder(vis, [xf_load[1], 0, 0], [xf_load[1], 0, xf_load[3] - r_load], 0.3, mat, "pedestal")

    # camera angle
    settransform!(vis["/Cameras/default"], compose(Translation(4., 0., 8.5),LinearMap(RotY(-pi/2))))

    # load in quad mesh
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.085
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling


    for (kk,k) in enumerate(index)
        x_load = prob_load.X[k][1:n_slack]

        for i = 1:num_lift
            x_lift = prob_lift[i].X[k][1:n_slack]

            setobject!(vis["lift$k$i"]["robot"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, tp[kk])))
            settransform!(vis["lift$k$i"], Translation(x_lift...))

            cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
            setobject!(vis["cable$k"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, tp[kk])))
            settransform!(vis["cable$k"]["$i"], cable_transform(x_lift,x_load))

        end
        setobject!(vis["load$k"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, tp[kk])))
        settransform!(vis["load$k"], Translation(x_load...))

    end
end
