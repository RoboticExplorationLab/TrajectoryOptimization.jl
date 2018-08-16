using Plots

function println(level::Symbol, msg::String)::Void
    if level_priorities[level] ≥ level_priorities[debug_level::Symbol]
        println(msg)
    end
end

print_info(msg) = println(:info,msg)
print_debug(msg) = println(:debug,msg)

#TODO finish
function plot_cost(results::ResultsCache)
    index_outerloop = find(x -> x == 1, results.iter_type)
end

"""
@(SIGNATURES)

Generate an animation of a trajectory as it evolves during a solve
"""
function trajectory_animation(results::ResultsCache;traj::String="state",ylim=[-10;10],title::String="Trajectory Evolution",filename::String="trajectory.gif",fps::Int=1)::Void
    anim = @animate for i=1:results.termination_index
        if traj == "state"
            t = results.result[i].X'
        elseif traj == "control"
            t = results.result[i].U'
        end
        plot(t,ylim=(ylim[1],ylim[2]),size=(200,200),label="",width=1,title=title)
        if results.iter_type[i] == 2
            plot!(xlabel="Infeasible->Feasible") # note the transition from an infeasible to feasible solve
        end
    end

    path = joinpath(Pkg.dir("TrajectoryOptimization"),filename)
    gif(anim,path,fps=fps)
    return nothing
end

"""
@(SIGNATURES)

Generate an animation of a pendulum
"""
function pendulum_animation(results::SolverResults;filename::String="pendulum.gif",fps::Int=1)::Void
    anim = @animate for i=1:size(results1.X,2)
        x = cos(results1.X[1,i] - pi/2)
        y = sin(results1.X[1,i] - pi/2)
        plot([0,x],[0,y],xlims=(-1.5,1.5),ylims=(-1.5,1.5),color="black",size=(300,300),label="",title="Pendulum")
    end
    path = joinpath(Pkg.dir("TrajectoryOptimization"),filename)
    gif(anim,path,fps=fps)
    return nothing
end

"""
@(SIGNATURES)

    Convert ZYX Euler angles to quaternion [v;s]
"""
function eul2quat(eul)
    ## Translate the given Euler angle with a specified axis rotation sequence into the corresponding quaternion:

    quat = zeros(4,1)
    # case 'ZYX'
	c_1 = cos(eul[1,1]*0.5)
	s_1 = sin(eul[1,1]*0.5)
	c_2 = cos(eul[2,1]*0.5)
	s_2 = sin(eul[2,1]*0.5)
	c_3 = cos(eul[3,1]*0.5)
	s_3 = sin(eul[3,1]*0.5)

    #TODO As noted in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf, the original algorithm has q = [s;v], I've updated the equation so that q = [v;s]; needs to be tested
	quat[2,1] = c_1*c_2*c_3 + s_1*s_2*s_3
	quat[3,1] = c_1*c_2*s_3 - s_1*s_2*c_3
	quat[4,1] = c_1*s_2*c_3 + s_1*c_2*s_3
	quat[1,1] = s_1*c_2*c_3 - c_1*s_2*s_3

    quat
end

"""
@(SIGNATURES)

Plot a 3D trajectory (static)
"""
function plot_3D_trajectory(results::ResultsCache, solver::Solver;xlim=[-10.0;10.0],ylim=[-10.0;10.0],zlim=[-10.0;10.0],title::String="3D Trajectory")::Void
    # initialize a 3D plot with 1 empty series
    plt = path3d(1, xlim=(xlim[1],xlim[2]), ylim=(ylim[1],ylim[2]), zlim=(zlim[1],zlim[2]),
                    xlab = "", ylab = "", zlab = "",
                    title=title,legend=nothing,color="blue",width=2)

    # build 3D plot
    for i=1:solver.N
        push!(plt, results.X[1,i], results.X[2,i], results.X[3,i])
    end
    plot!((solver.obj.x0[1],solver.obj.x0[2],solver.obj.x0[3]),marker=(:circle,"red"))
    plot!((solver.obj.xf[1],solver.obj.xf[2],solver.obj.xf[3]),marker=(:circle,"green"))
    display(plt)
    return nothing
end
