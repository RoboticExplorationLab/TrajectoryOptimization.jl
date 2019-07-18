
col_altro = "orange"
col_ipopt = "blue"
col_snopt = "green!80!black"
col_alilqr = "cyan"

function trajectory_plot(prob::Problem; kwargs...)
    x = [x[1] for x in prob.X]
    y = [x[2] for x in prob.X]
    PGF.Plots.Linear(x,y; kwargs...)
end

function trajectory_plot_flip(prob::Problem; kwargs...)
    x = [x[2] for x in prob.X]
    y = [x[1] for x in prob.X]
    PGF.Plots.Linear(x,y; kwargs...)
end
