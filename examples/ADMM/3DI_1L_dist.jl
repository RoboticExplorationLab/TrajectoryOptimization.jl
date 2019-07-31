using Distributed
using TimerOutputs
addprocs(3)

using TrajectoryOptimization
@everywhere using TrajectoryOptimization
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include("examples/ADMM/3DI_problem.jl")
@everywhere const TO = TrajectoryOptimization


# Set up
n_slack = 3
n_lift = 6
m_lift = 3
num_lift = 3
n_load = 6
m_load = n_slack * num_lift

# Initial state
scaling = 1.
x0_lift = [zeros(n_lift) for i in 1:num_lift]

shift_ = zeros(n_lift)
shift_[1:3] = [0.0;0.0;1.]

x0_lift[1][1:3] .= scaling*[sqrt(8/9);0.;4/3]
x0_lift[2][1:3] .= scaling*[-sqrt(2/9);sqrt(2/3);4/3]
x0_lift[3][1:3] .= scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
x0_lift .+= [shift_ for i in 1:num_lift]

x0_load = zeros(n_load)
x0_load += shift_


# goal state
_shift = zeros(n_lift)
_shift[1:3] = [10.;0.0;0.0]

xf_lift = x0_lift .+ [_shift for i in 1:num_lift]
xf_load = x0_load + _shift

d1 = norm(xf_load[1:3]-xf_lift[1][1:3])
d2 = norm(xf_load[1:3]-xf_lift[2][1:3])
d3 = norm(xf_load[1:3]-xf_lift[3][1:3])

d = [d1, d2, d3]

# Costs
Q_lift = [1.0e-2*Diagonal(I,n_lift), 10.0e-2*Diagonal(I,n_lift), 0.1e-2*Diagonal(I,n_lift)]

# Obstacles
r_lift = 0.1
r_load = 0.1
r_cylinder = 0.75
_cyl = NTuple{3,Float64}[]
push!(_cyl,(5.,1.,r_cylinder))
push!(_cyl,(5.,-1.,r_cylinder))


# Initialize problems
verbose = false
distributed = true
opts_ilqr = iLQRSolverOptions(verbose=verbose,iterations=500)
opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-5,
    iterations=100,
    penalty_scaling=2.0,
    penalty_initial=10.)
opts = opts_al

distributed = true
distributed = false
if distributed
    probs = ddata(T=Problem{Float64,Discrete});
    @sync for i in workers()
        j = i - 1
        @spawnat i probs[:L] = build_lift_problem(x0_lift[j], xf_lift[j], Q_lift[j], r_lift, _cyl, num_lift)
    end
    prob_load = build_load_problem(x0_load, xf_load, r_load, _cyl, num_lift)
else
    probs = Problem{Float64,Discrete}[]
    prob_load = build_load_problem(x0_load, xf_load, r_load, _cyl, num_lift)
    for i = 1:num_lift
        push!(probs, build_lift_problem(x0_lift[i], xf_lift[i], Q_lift[i], r_lift, _cyl, num_lift))
    end
end

@time sol = solve_admm(prob_load, probs, opts_al)

vis = Visualizer()
open(vis)
visualize_lift_system(vis, sol, r_lift, r_load)


function solve_admm(prob_load, probs, opts::AugmentedLagrangianSolverOptions)
    prob_load = copy(prob_load)
    probs = copy_probs(probs)
    X_cache, U_cache, X_lift, U_lift = init_cache(probs)
    solve_admm!(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts)
    combine_problems(prob_load, probs)
end

function solve_init!(prob_load, probs::DArray, X_cache, U_cache, X_lift, U_lift, opts)
    # for i in workers()
    #     @spawnat i solve!(probs[:L], opts)
    # end
    futures = [@spawnat w solve!(probs[:L], opts_al) for w in workers()]
    solve!(prob_load, opts)
    wait.(futures)

    # Get trajectories
    X_lift0 = fetch.([@spawnat w probs[:L].X for w in workers()])
    U_lift0 = fetch.([@spawnat w probs[:L].U for w in workers()])
    for i = 1:num_lift
        X_lift[i] .= X_lift0[i]
        U_lift[i] .= U_lift0[i]
    end
    update_load_problem(prob_load, X_lift, U_lift, d)

    # Send trajectories
    @sync for w in workers()
        for i = 2:4
            @spawnat w begin
                X_cache[:L][i] .= X_lift0[i-1]
                U_cache[:L][i] .= U_lift0[i-1]
            end
        end
        @spawnat w begin
            X_cache[:L][1] .= prob_load.X
            U_cache[:L][1] .= prob_load.U
        end
    end

    # Update lift problems
    @sync for w in workers()
        agent = w - 1
        @spawnat w update_lift_problem(probs[:L], X_cache[:L], U_cache[:L], agent, d[agent], r_lift)
    end
end

function solve_init!(prob_load, probs::Vector{<:Problem}, X_cache, U_cache, X_lift, U_lift, opts)
    num_lift = length(probs)
    for i = 1:num_lift
        solve!(probs[i], opts)
    end
    solve!(prob_load, opts)

    # Get trajectories
    X_lift0 = [prob.X for prob in probs]
    U_lift0 = [prob.U for prob in probs]
    for i = 1:num_lift
        X_lift[i] .= X_lift0[i]
        U_lift[i] .= U_lift0[i]
    end
    update_load_problem(prob_load, X_lift, U_lift, d)

    # Send trajectories
    for w = 2:4
        for i = 2:4
            X_cache[w-1][i] .= X_lift0[i-1]
            U_cache[w-1][i] .= U_lift0[i-1]
        end
        X_cache[w-1][1] .= prob_load.X
        U_cache[w-1][1] .= prob_load.U
    end

    # Update lift problems
    for w = 2:4
        agent = w - 1
        update_lift_problem(probs[agent], X_cache[agent], U_cache[agent], agent, d[agent], r_lift)
    end
end

function solve_admm!(prob_load, probs::Vector{<:Problem}, X_cache, U_cache, X_lift, U_lift, opts)
    num_left = length(probs) - 1

    # Solve the initial problems
    solve_init!(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts_al)

    # create augmented Lagrangian problems, solvers
    solvers_al = AugmentedLagrangianSolver{Float64}[]
    for i = 1:num_lift
        solver = AugmentedLagrangianSolver(probs[i],opts)
        probs[i] = AugmentedLagrangianProblem(probs[i],solver)
        push!(solvers_al, solver)
    end
    solver_load = AugmentedLagrangianSolver(prob_load, opts)
    prob_load = AugmentedLagrangianProblem(prob_load, solver_load)

    for ii = 1:opts.iterations
        # Solve each AL problem
        for i = 1:num_lift
            TO.solve_aula!(probs[i], solvers_al[i])
        end

        # Get trajectories
        for i = 1:num_lift
            X_lift[i] .= probs[i].X
            U_lift[i] .= probs[i].U
        end

        # Solve load with updated lift trajectories
        TO.solve_aula!(prob_load, solver_load)

        # Send trajectories
        for i = 1:num_lift  # loop over agents
            for j = 1:num_lift
                i != j || continue
                X_cache[i][j+1] .= X_lift[j]
            end
            X_cache[i][1] .= prob_load.X
        end

        max_c = maximum(max_violation.(solvers_al))
        max_c = max(max_c, max_violation(solver_load))
        println(max_c)
        if max_c < opts.constraint_tolerance
            break
        end
    end
end

function solve_admm!(prob_load, prob::DArray, X_cache, U_cache, X_lift, U_lift, opts)
    solve_init!(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts_al)

    # create augmented Lagrangian problems, solvers
    solvers_al = ddata(T=AugmentedLagrangianSolver{Float64});
    @sync for w in workers()
        @spawnat w begin
            solvers_al[:L] = AugmentedLagrangianSolver(probs[:L], opts)
            probs[:L] = AugmentedLagrangianProblem(probs[:L],solvers_al[:L])
        end
    end
    solver_load = AugmentedLagrangianSolver(prob_load, opts)
    prob_load = AugmentedLagrangianProblem(prob_load, solver_load)

    for ii = 1:opts.iterations
        # Solve each AL lift problem
        future = [@spawnat w TO.solve_aula!(probs[:L], solvers_al[:L]) for w in workers()]
        wait.(future)

        # Get trajectories
        X_lift0 = fetch.([@spawnat w probs[:L].X for w in workers()])
        U_lift0 = fetch.([@spawnat w probs[:L].U for w in workers()])
        for i = 1:num_lift
            X_lift[i] .= X_lift0[i]
            U_lift[i] .= U_lift0[i]
        end
        TO.solve_aula!(prob_load, solver_load)

        # Send trajectories
        @sync for w in workers()
            for i = 2:4
                @spawnat w begin
                    X_cache[:L][i] .= X_lift0[i-1]
                    U_cache[:L][i] .= U_lift0[i-1]
                end
            end
            @spawnat w begin
                X_cache[:L][1] .= prob_load.X
                U_cache[:L][1] .= prob_load.U
            end
        end
        max_c = maximum(fetch.([@spawnat w max_violation(solvers_al[:L]) for w in workers()]))
        max_c = max(max_c, max_violation(solver_load))
        println(max_c)
        if max_c < opts.constraint_tolerance
            break
        end
    end
end

copy_probs(probs::Vector{<:Problem}) = copy.(probs)
function copy_probs(probs::DArray)
    probs2 = ddata(T=eltype(probs))
    @sync for w in workers()
        @spawnat w probs2[:L] = copy(probs[:L])
    end
    return probs2
end


combine_problems(prob_load, probs::Vector{<:Problem}) = [[prob_load]; probs]
function combine_problems(prob_load, probs::DArray)
    problems = fetch.([@spawnat w probs[:L] for w in workers()])
    combine_problems(prob_load, problems)
end

function init_cache(probs::Vector{<:Problem})
    num_lift = length(probs)
    X_lift = [deepcopy(prob.X) for prob in probs]
    U_lift = [deepcopy(prob.U) for prob in probs]
    X_traj = [[prob_load.X]; X_lift]
    U_traj = [[prob_load.U]; U_lift]
    X_cache = [deepcopy(X_traj) for i=1:num_lift]
    U_cache = [deepcopy(U_traj) for i=1:num_lift]
    return X_cache, U_cache, X_lift, U_lift
end

function init_cache(probs::DArray)
    # Initialize state and control caches
    X_lift = fetch.([@spawnat w deepcopy(probs[:L].X) for w in workers()])
    U_lift = fetch.([@spawnat w deepcopy(probs[:L].U) for w in workers()])
    X_traj = [[prob_load.X]; X_lift]
    U_traj = [[prob_load.U]; U_lift]

    X_cache = ddata(T=Vector{Vector{Vector{Float64}}});
    U_cache = ddata(T=Vector{Vector{Vector{Float64}}});
    @sync for w in workers()
        @spawnat w begin
            X_cache[:L] = X_traj
            U_cache[:L] = U_traj
        end
    end
    return X_cache, U_cache, X_lift, U_lift
end
