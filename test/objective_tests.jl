using TrajectoryOptimization: state, control

@testset "Objectives" begin
    n, m = rand(10:20), rand(5:10)
    N = 11

    Q = Diagonal(@SVector fill(0.1, n))
    R = Diagonal(@SVector fill(0.01, m))
    H = rand(m, n)
    q = @SVector rand(n)
    r = @SVector rand(m)
    c = rand()
    Qf = Diagonal(@SVector fill(10.0, n))
    xf = @SVector ones(n)

    @testset "Constructors" begin
        # Simple copy objective constructor
        qcost = QuadraticCost(Q, R, H, q, r, c)
        obj = Objective(qcost, N)
        @test length(obj) == N
        @test state_dim(obj) == n
        @test control_dim(obj) == m
        @test TO.get_J(obj) === obj.J
        @test obj[end].terminal == false

        # Maintains associativity
        @test obj[1].Q === obj[2].Q
        @test obj[1].q === obj[2].q
        qcost.r[1] = 1
        @test obj[3].r[1] ≈ 1

        # Terminal cost constructor
        qterm = QuadraticCost(Qf, zero(R), q = q, c = c, terminal = true)
        dterm = DiagonalCost(Qf, zero(R), q = q, c = c, terminal = true)
        obj = Objective(qcost, qterm, N)
        @test eltype(obj) <: QuadraticCost
        @test obj[1].Q === obj[2].Q
        @test obj[1].q === obj[2].q
        @test obj[1].Q ≈ Q
        @test obj[end].Q ≈ Qf
        @test obj[end].R ≈ zero(R)
        @test obj[end].q ≈ q
        @test obj[end].terminal == true

        obj = Objective(qcost, dterm, N)
        @test eltype(obj) == QuadraticCost{
            n,
            m,
            Float64,
            Diagonal{Float64,SVector{n,Float64}},
            Diagonal{Float64,SVector{m,Float64}},
        }
        @test obj[1].Q === obj[2].Q
        @test obj[1].q === obj[2].q
        @test obj[1].Q ≈ Q
        @test obj[end].Q ≈ Qf
        @test obj[end].R ≈ zero(R)
        @test obj[end].q ≈ q
        @test obj[end].terminal == true

        dcost = DiagonalCost(Q, R, q = q, r = r)
        obj = Objective(dcost, dterm, N)
        @test eltype(obj) == DiagonalCost{n,m,Float64}
        @test obj[1].Q == Q
        @test obj[end].Q == Qf

        # Cost vector constructor
        costs = [copy(k < N ? qcost : qterm) for k = 1:N]
        obj = Objective(costs)
        @test length(obj) == N
        @test eltype(obj) == QuadraticCost{
            n,
            m,
            Float64,
            Diagonal{Float64,MVector{n,Float64}},
            Diagonal{Float64,MVector{m,Float64}},
        }
        @test obj[1] !== obj[2]
        @test obj[1].Q !== obj[2].Q
        @test obj[1].q !== obj[2].q


        # LQR constructors
        xf = @SVector rand(n)
        obj = LQRObjective(Q, R, Qf, xf, N)
        @test length(obj) == N
        @test eltype(obj) == DiagonalCost{n,m,Float64}
        @test obj[1].Q ≈ Q
        @test obj[1].q ≈ -Q * xf
        @test obj[2].r ≈ zero(r)
        @test obj[2].c ≈ 0.5 * xf'Q * xf
        @test obj[end].c ≈ 0.5 * xf'Qf * xf

        obj = LQRObjective(Matrix(Q), R, Qf, xf, N)
        @test eltype(obj) <: QuadraticCost{n,m,Float64,<:SizedMatrix,<:Diagonal}
        @test obj[1].Q ≈ Q
        @test obj[1].q ≈ -Q * xf
        @test obj[2].r ≈ zero(r)
        @test obj[2].c ≈ 0.5 * xf'Q * xf
        @test obj[end].c ≈ 0.5 * xf'Qf * xf

        obj = LQRObjective(Q.diag, R.diag, Qf, xf, N)
        @test eltype(obj) <: DiagonalCost{n,m}
        @test obj[1].Q ≈ Q
        @test obj[1].q ≈ -Q * xf
        @test obj[2].r ≈ zero(r)
        @test obj[2].c ≈ 0.5 * xf'Q * xf
        @test obj[end].c ≈ 0.5 * xf'Qf * xf

        obj = LQRObjective(Diagonal(Vector(Q.diag)), R.diag, Vector(Qf.diag), xf, N)
        @test eltype(obj) <: DiagonalCost{n,m}

        # Test error expansion constructor
        model = Cartpole()
        E0 = TO.CostExpansion(n,m,N)
        E = TO.CostExpansion{Float32}(n,m,N)
        @test eltype(E[1].hess) == Float32
        E1 = TO.CostExpansion(E0, model)
        @test E1 === E0

        model = Quadrotor() 
        E0 = TO.CostExpansion(13,4,11)
        @test_throws AssertionError TO.CostExpansion(E0, model)
        E0 = TO.CostExpansion(12,4,11)
        E1 = TO.CostExpansion(E0, model)
        @test E1 !== E0
        @test size(E1[1].xx) == (13,13)
        @test size(E0[1].xx) == (12,12)
    end

    @testset "Evaluation and expansion" begin
        # Evaluation and expansion functions
        dt = 0.01
        N = 101
        Z = Traj([KnotPoint(rand(SVector{n}), rand(SVector{m}), dt * (k < N)) for k = 1:N])
        uref = @SVector rand(m)
        obj = LQRObjective(Q, R, Qf, xf, N, uf = uref)
        J = sum([
                0.5 * (x - xf)'Q * (x - xf) + 0.5 * (u - uref)'R * (u - uref)
                for (x, u) in zip(states(Z)[1:N-1], controls(Z))
            ]) * dt + 0.5 * (state(Z[end]) - xf)'Qf * (state(Z[end]) - xf)
        @test cost(obj, Z) ≈ J

        TO.cost!(obj, Z)
        @test sum(TO.get_J(obj)) ≈ J

        # E = TO.QuadraticObjective(n, m, N)
        E0 = TO.CostExpansion(n, m, N)
        TO.cost_gradient!(E0, obj, Z)
        @test (@allocated TO.cost_gradient!(E0, obj, Z)) == 0
        @test all([E0[k].q ≈ obj[k].Q * (state(Z[k]) - xf) * (k < N ? dt : 1.0) for k = 1:N])
        @test all([E0[k].r ≈ R * (control(Z[k]) - uref) * dt for k = 1:N-1])

        TO.cost_hessian!(E0, obj, Z, init=true, rezero=true)
        # @test (@allocated TO.cost_hessian!(E0, obj, Z, init=true)) == 0
        @test all([E0[k].Q ≈ obj[k].Q * (k < N ? dt : 1.0) for k = 1:N])
        @test all([E0[k].R ≈ R * dt for k = 1:N-1])

        # pass in cache
        cache = TO.ExpansionCache(obj[1])
        TO.cost_expansion!(E0, obj, Z)
        TO.cost_expansion!(E0, obj, Z, cache)

        # Test error expansion
        model = Cartpole()
        G = [zeros(n,n) for k = 1:N]
        E = TO.CostExpansion(E0, model)
        RobotDynamics.state_diff_jacobian!(G, model, Z)
        TO.error_expansion!(E, E0, model, Z, G)

        model = Quadrotor()
        G = [SizedMatrix{13,12}(zeros(13,12)) for k = 1:N]
        Z = Traj([KnotPoint(rand(model)..., dt) for k = 1:N])
        RobotDynamics.state_diff_jacobian!(G, model, Z)
        E0 = TO.CostExpansion(12, 4, N)
        E = TO.CostExpansion(E0, model)
        obj = LQRObjective(Diagonal(rand(13)), Diagonal(rand(4)), 
            Diagonal(rand(13)), rand(model)[1], N)
        TO.cost_expansion!(E, obj, Z, init=true, rezero=true)
        @test E[1].xx != zeros(13,13)
        TO.error_expansion!(E0, E, model, Z, G)
        @test (@allocated TO.error_expansion!(E0, E, model, Z, G)) == 0
    end
end
