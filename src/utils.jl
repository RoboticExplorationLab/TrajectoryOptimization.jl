using Plots

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
