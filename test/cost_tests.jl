@testset "Quadratic Costs" begin
    # Quadratic Costs
    n, m = rand(10:20), rand(5:10)
    Q = Diagonal(@SVector fill(0.1, n))
    R = Diagonal(@SVector fill(0.01, m))
    H = rand(m, n)
    q = @MVector rand(n)
    r = @MVector rand(m)
    c = rand()
    Qf = Diagonal(@SVector fill(10.0, n))
    xf = @SVector ones(n)

    @testset "Constructors" begin
        qcost = QuadraticCost(Q, R)
        @test qcost.Q == Q
        @test qcost.q == zeros(n)
        @test qcost.R == R
        @test qcost.r == zeros(m)
        @test qcost.c == 0
        @test state_dim(qcost) == n
        @test control_dim(qcost) == m
        @test TO.is_blockdiag(qcost) == true

        qcost = QuadraticCost(Q, R, H = H)
        @test qcost.H ≈ H
        @test qcost.Q == Q
        @test qcost.q == zero(q)

        qcost = QuadraticCost(Q, R, r = r, H = H, q = q, c = c, terminal = true)
        @test qcost.H ≈ H
        @test qcost.Q == Q
        @test qcost.q == q
        @test qcost.r == r
        @test qcost.c == c
        @test qcost.zeroH == false
        @test qcost.terminal == true

        qcost = QuadraticCost(Q, R, H, q, r, c, checks = false, terminal = false)
        @test qcost.H ≈ H
        @test qcost.Q == Q
        @test qcost.q == q
        @test qcost.r == r
        @test qcost.c == c
        @test qcost.zeroH == false
        @test qcost.terminal == false
        @test TO.is_blockdiag(qcost) == false

        qcost = QuadraticCost{Float64}(n, m)
        @test eltype(qcost.Q) == Float64
        @test qcost.Q == I(n)
        @test qcost.R == I(m)
        @test qcost.q == zero(q)
        @test qcost.r == zero(r)
        @test qcost.c == zero(c)

        qcost = QuadraticCost{Float32}(n, m)
        @test eltype(qcost.Q) == Float32
        @test eltype(qcost.r) == Float32
        @test eltype(qcost.c) == Float32

        @test_logs (:warn, "R is not positive definite") QuadraticCost(Q, R * 0, H, q, r, c)
        @test_nowarn QuadraticCost(Q * 0, R, H, q, r, c)
        @test_logs (:warn, "Q is not positive semidefinite") QuadraticCost(
            Q .- 0.2,
            R,
            H,
            q,
            r,
            c,
        )
        @test_logs (:warn, "R is not positive definite") QuadraticCost(Q, R * 0)
        @test_logs (:warn, "Q is not positive semidefinite") QuadraticCost(
            Q .- 0.2,
            R,
            q = q,
        )
        @test_nowarn QuadraticCost(Q .- 0.2, R, q = q, checks = false)

        # Diagonal Cost functions
        dcost = DiagonalCost(Q, R)
        @test dcost.Q === Q
        @test dcost.R === R
        @test dcost.r == zero(r)
        @test dcost.q == zero(q)

        dcost = DiagonalCost(Q, R, q = q)
        @test dcost.Q === Q
        @test dcost.R === R
        @test dcost.r == zero(r)
        @test dcost.q === q

        dcost = DiagonalCost(Q.diag, R.diag, q = q)
        @test dcost.Q === Q
        @test dcost.R === R
        @test dcost.r == zero(r)
        @test dcost.q === q

        dcost = DiagonalCost(Vector(Q.diag), Vector(R.diag), q = q)
        @test dcost.Q === Q
        @test dcost.R === R
        @test dcost.r == zero(r)
        @test dcost.q === q
        @test dcost.Q isa Diagonal{Float64,<:SVector}

        # Test convenience constructors
        quadcost = LQRCost(Q, R, xf)
        @test quadcost.Q == Q
        @test quadcost.q == -Q * xf
        @test quadcost.R == R
        @test quadcost.r == zeros(m)
        @test quadcost.c ≈ 0.5 * xf'Q * xf
        @test quadcost isa DiagonalCost
        @test TO.is_blockdiag(quadcost)
        @test TO.is_diag(quadcost)
    end

    @testset "Math Operations" begin
        # Test adding cost functions
        qcost = QuadraticCost(Q, R, H, q, r, c, checks = false, terminal = false)
        dcost = DiagonalCost(Vector(Q.diag), Vector(R.diag), q = q)
        addcost = dcost + qcost
        @test addcost.Q ≈ 2Q
        @test addcost.R ≈ 2R
        @test addcost.q ≈ 2q
        @test addcost.r ≈ r
        @test addcost.c ≈ c
        @test addcost isa QuadraticCost

        dcost = DiagonalCost(Vector(Q.diag), Vector(R.diag), q, r, c)
        addcost = dcost + dcost
        @test addcost.Q ≈ 2Q
        @test addcost.R ≈ 2R
        @test addcost.q ≈ 2q
        @test addcost.r ≈ 2r
        @test addcost.c ≈ 2c

        dcost2 = DiagonalCost(Q.diag, Q.diag, q, q, c)
        @test_throws AssertionError dcost2 + dcost

        # Test inversion
        dinv = inv(dcost)
        @test dinv.Q ≈ inv(Q)
        @test dinv.R ≈ inv(R)
        @test dinv.q == q

        qinv = inv(qcost)
        G = inv(Matrix([Q H'; H R]))
        @test G[1:n, 1:n] ≈ qinv.Q
        @test G[1:n, n.+(1:m)] ≈ qinv.H'
        @test G[n.+(1:m), n.+(1:m)] ≈ qinv.R
        @test dinv.q == q
        @test dinv.r == r
        @test dinv.c == c
        @test TO.is_diag(qinv) == false
        @test TO.is_diag(dinv) == true
        @test TO.is_blockdiag(qinv) == false
        @test TO.is_blockdiag(dinv) == true

        qinv = inv(QuadraticCost(Q, R))
        @test qinv.Q ≈ inv(Q)
        @test qinv.R ≈ inv(R)
        @test TO.is_blockdiag(qinv) == true
        @test TO.is_diag(qinv) == true

    end

    @testset "Conversion and Promotion" begin
        # Test promotion rules
        qcost = QuadraticCost(10Q, zero(R), q = q, c = c, terminal = true)
        dcost = DiagonalCost(2Q, zero(R), q = q, c = c, terminal = true)
        c1, c2 = promote(qcost, dcost)
        @test c1 isa QuadraticCost
        @test c2 isa QuadraticCost
        @test c1.Q ≈ qcost.Q
        @test c1.R ≈ qcost.R
        @test c2.Q ≈ dcost.Q

        c1, c2 = promote(dcost, dcost)
        @test c1 isa DiagonalCost
        @test c2 isa DiagonalCost
        @test c1.Q ≈ dcost.Q
        @test c1.R ≈ dcost.R
        @test c2.Q ≈ dcost.Q

        c1 = QuadraticCost(Q, R)
        c2 = QuadraticCost(2 * Matrix(Q), 2 * R)
        p1, p2 = promote(c1, c2)
        @test p1.Q ≈ Q
        @test p2.Q ≈ 2Q
        @test p1.R ≈ R
        @test p2.R ≈ 2R
        @test p1.Q isa SizedMatrix
        @test c1.Q isa Diagonal
        @test c2.Q isa Matrix
    end

    @testset "Evaluation and expansion" begin
        # Test cost functions and expansions
        x = @SVector rand(n)
        u = @SVector rand(m)
        dt = 0.1
        z = KnotPoint(x, u, dt)
        zterm = KnotPoint(x, u, 0.0)
        @test TO.is_terminal(zterm)

        qcost = QuadraticCost(Q, R, H, q, r, c)
        @test TO.stage_cost(qcost, z) ≈ TO.stage_cost(qcost, x, u) * dt
        @test TO.stage_cost(qcost, x, u) ≈
              0.5 * (x'Q * x + u'R * u) + q'x + r'u + c + u'H * x

        E = QuadraticCost{Float64}(n, m)
        TO.gradient!(E, qcost, zterm)
        @test E.q ≈ Q * x + q
        @test E.r ≈ zero(r)
        TO.gradient!(E, qcost, z)
        @test E.q ≈ Q * x + q + H'u
        @test E.r ≈ R * u + r + H * x

        TO.hessian!(E, qcost, zterm)
        @test E.Q ≈ Q
        @test E.R ≈ I(m)
        TO.hessian!(E, qcost, z) 
        @test E.R ≈ R
        @test E.H ≈ H
        @test (@allocated TO.gradient!(E, qcost, zterm)) == 0
        # @test (@allocated TO.gradient!(E, qcost, z)) == 0
        @test (@allocated TO.hessian!(E, qcost, zterm)) == 0
        @test (@allocated TO.hessian!(E, qcost, z)) == 0

        dcost = DiagonalCost(Q, R, q, r, c)
        @test TO.stage_cost(dcost, z) ≈ TO.stage_cost(dcost, x, u) * dt
        @test TO.stage_cost(dcost, x, u) ≈ 0.5 * (x'Q * x + u'R * u) + q'x + r'u + c

        E = QuadraticCost{Float64}(n, m)
        TO.gradient!(E, dcost, zterm)
        @test E.q ≈ Q * x + q
        @test E.r ≈ zero(r)
        TO.gradient!(E, dcost, z) 
        @test E.q ≈ Q * x + q
        @test E.r ≈ R * u + r

        TO.hessian!(E, dcost, zterm)
        @test E.Q ≈ Q
        @test E.R ≈ I(m)
        TO.hessian!(E, dcost, z) 
        @test E.R ≈ R
        @test E.H ≈ zero(H)
        @test (@allocated TO.gradient!(E, dcost, zterm)) == 0
        @test (@allocated TO.gradient!(E, dcost, z)) == 0
        @test (@allocated TO.hessian!(E, dcost, zterm)) == 0
        @test (@allocated TO.hessian!(E, dcost, z)) == 0
    end

end
