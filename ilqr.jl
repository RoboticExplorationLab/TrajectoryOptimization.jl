module iLQR
    using RigidBodyDynamics
    using ForwardDiff

    include("model.jl")
    include("solver.jl")
    include("ilqr_algorithm.jl")
end
<<<<<<< HEAD

function solve(solver::Solver, iterations::Int64=100, eps::Float64=1e-3; control_init::String="random")
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    X = zeros(n,N)
    X_ = zeros(n,N)

    if control_init == "random"
        U = 1.0*rand(m,N-1)
    else
        U = zeros(m,N-1)
    end
    U_ = zeros(m,N-1)

    K = zeros(m,n,N-1)
    d = zeros(m,N-1)

    X = rollout(solver, X, U)
    J_prev = cost(solver, X, U)
    println("Initial Cost: $J_prev\n")

    for i = 1:iterations
        println("*** Iteration: $i ***")
        K, d, v1, v2 = backwardpass(solver,X,U,K,d)
        X, U, J = forwardpass(solver,X,U,K,d,J_prev,v1,v2)

        if abs(J-J_prev) < eps
          println("-----SOLVED-----")
          println("eps criteria met at iteration: $i")
          break
        end
        J_prev = copy(J)
    end

    return X, U
end
=======
>>>>>>> beb2cb6b42af27237156cc191bd63d0ac3ce2623
