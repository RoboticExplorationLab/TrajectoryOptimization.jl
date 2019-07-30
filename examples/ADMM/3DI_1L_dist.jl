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
distributed = true
if distributed
    probs = ddata(T=Problem{Float64,Discrete});
    @sync for i in workers()
        @spawnat i probs[:L] = build_lift_problem(x0_lift[j], xf_lift[j], Q_lift[j], r_lift, _cyl, num_lift)
    end
    prob_load = build_load_problem(x0_load, xf_load, r_load, _cyl, num_lift)

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
else
    probs = Problem{Float64,Discrete}[]
    push!(probs, build_load_problem(x0_load, xf_load, r_load, _cyl, num_lift))
    for i = 1:num_lift
        push!(probs, build_lift_problem(x0_lift[i], xf_lift[i], Q_lift[i], r_lift, _cyl, num_lift))
    end

    X_traj = [deepcopy(prob.X) for prob in probs]
    U_traj = [deepcopy(prob.U) for prob in probs]
    X_cache = [deepcopy(X_traj) for i=1:4]
    U_cache = [deepcopy(U_traj) for i=1:4]
end

# Solve the initial problem
verbose = false
distributed = false
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

@time solve_admm2(probs, X_cache, U_cache, opts_al)

@time solve_admm_dist(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts)


# Solve with ADMM


function solve_init_distributed(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts)
    for i in workers()
        @spawnat i solve!(probs[:L], opts_al)
    end
    futures = [@spawnat w solve!(probs[:L], opts_al) for w in workers()]
    solve!(prob_load, opts_al)
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

function solve_init(probs, X_cache, U_cache, opts)
    num_lift = length(probs) - 1

    @timeit "copy probs" probs = copy.(probs)

    # @sync for i in workers()
    #     @spawnat i solve(probs[:L], opts_al)
    # end

    for i = 1:num_lift + 1
        @timeit "solve" solve!(probs[i], opts_al)
    end

    # Get trajectories from solve
    @timeit "copy" begin
        for i = 1:num_lift + 1
            for j = 1:num_lift + 1
                X_cache[j][i] .= probs[i].X
                U_cache[j][i] .= probs[i].U
            end
        end
    end

    # Add constraints to problems
    @timeit "update problem" begin
        X_load = X_cache[2][1]
        U_load = U_cache[2][1]
        for i = 1:num_lift
            update_lift_problem(probs[i+1], X_cache[i+1], U_cache[i+1], i, d[i], r_lift)
        end
        update_load_problem(probs[1], X_cache[1][2:4], U_cache[1][2:4], d)
    end
    return probs
end

function solve_admm2(probs0, X_cache, U_cache, opts)
    num_left = length(probs0) - 1

    # Solve the initial problems
    probs = solve_init(probs0, X_cache, U_cache, opts_al)

    # create augmented Lagrangian problems, solvers
    solvers_al = AugmentedLagrangianSolver{Float64}[]
    for i = 1:num_lift + 1
        solver = AugmentedLagrangianSolver(probs[i],opts)
        probs[i] = AugmentedLagrangianProblem(probs[i],solver)

        push!(solvers_al, solver)
    end

    for ii = 1:opts.iterations
        # Solve each AL problem
        for i = 1:num_lift
            w = i + 1
            TO.solve_aula!(probs[w], solvers_al[w])
        end

        # Get trajectories
        for i = 1:num_lift
            w = i + 1
            X_cache[1][w] .= probs[w].X
            U_cache[1][w] .= probs[w].U
        end

        # Solve load with updated lift trajectories
        TO.solve_aula!(probs[1], solvers_al[1])
        X_cache[1][1] .= probs[1].X
        U_cache[1][1] .= probs[1].U

        # Send trajectories
        for i = 1:num_lift
            w = i + 1
            for j = 1:num_lift + 1
                if j == w
                    continue
                end
                X_cache[w][j] .= X_cache[1][j]
            end
        end

        max_c = maximum(max_violation.(solvers_al))
        println(max_c)
        if max_c < opts.constraint_tolerance
            break
        end
    end
end


function solve_admm_dist(prob_load, prob, X_cache, U_cache, X_lift, U_lift, opts)
    solve_init_distributed(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts_al)

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
        solve!(prob_load, solver_load)

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
