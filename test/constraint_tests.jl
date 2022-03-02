#--- Setup
function alloc_con(con,z)
    ∇c = TO.gen_jacobian(con)
    c = zeros(RD.output_dim(con))
    allocs  = @ballocated RD.evaluate($con, $z) samples=1 evals=1
    allocs += @ballocated RD.evaluate!($con, $c, $z) samples=1 evals=1
    allocs += @ballocated RD.jacobian!($con, $∇c, $c, $z) samples=1 evals=1
end

model = Cartpole()
n,m = RD.dims(model)
x,u = rand(model)
t,h = 1.1, 0.1
z = KnotPoint(x,u,t,h)

#--- Goal Constraint
@testset "Goal Constraint" begin
    xf = @SVector rand(n)
    goal = GoalConstraint(xf)
    c = zeros(n)
    @test RD.evaluate(goal, z) ≈ x - xf
    @test RD.evaluate(goal, x) ≈ x - xf
    RD.evaluate!(goal, c, x)
    @test c ≈ x - xf
    C = zeros(n,n)
    RD.jacobian!(goal, C, c, z)
    @test C ≈ I(n)

    @test RD.output_dim(goal) == n
    @test TO.upper_bound(goal) ≈ zero(xf)
    @test TO.lower_bound(goal) ≈ zero(xf)
    @test TO.is_bound(goal)
    @test TO.check_dims(goal, n, m)
    @test TO.check_dims(goal, n+1, m) == false
    @test TO.check_dims(goal, n, m+1) == true
    @test TO.widths(goal) == (n,)
    @test TO.widths(goal, n, m) == (n,)
    @test TO.widths(goal, n+1, m) == (n+1,)
    @test state_dim(goal) == n

    @test GoalConstraint(Vector(xf)).xf isa MVector{n}
end


#--- Linear Constraint
@testset "Linear Constraint" begin
    p = 5
    A = @SMatrix rand(p,n+m)
    b = @SVector rand(p)
    ∇c = zeros(p,n+m)
    c = zeros(p)
    lin = LinearConstraint(n,m,A,b, Inequality())
    @test RD.evaluate(lin, z) ≈ A*z.z - b
    RD.evaluate!(lin, c, z)
    @test c ≈ A*z.z - b
    RD.jacobian!(lin, ∇c, c, z)
    @test ∇c ≈ A

    lin2 = LinearConstraint(n,m, Matrix(A), Vector(b), Inequality())
    @test lin2.A isa SizedMatrix{p,n+m}
    @test lin2.b isa SVector{p}

    @test RD.output_dim(lin) == p
    @test TO.widths(lin) == (n+m,)
    @test state_dim(lin) == n
    @test control_dim(lin) == m
    @test TO.upper_bound(lin) ≈ zeros(p)
    @test TO.lower_bound(lin) ≈ fill(-Inf,p)
    @test TO.is_bound(lin) == false

    A = @SMatrix rand(p,n)
    lin2 = LinearConstraint(n,m, A,b, Equality(), 1:n)
    @test RD.evaluate(lin2, z) ≈ A*x - b
    ∇c = zeros(p,n+m) 
    TO.jacobian!(lin2, ∇c, c, z)
    @test ∇c ≈ [A zeros(p)]

    @test TO.widths(lin2) == (n+m,)
    @test alloc_con(lin2,z) == 0
    @test state_dim(lin2) == n
    @test control_dim(lin2) == m
    @test TO.upper_bound(lin2) ≈ zeros(p)
    @test TO.lower_bound(lin2) ≈ zeros(p)
    @test TO.is_bound(lin2) == false

    @test TO.sense(lin) == Inequality()
    @test TO.sense(lin2) == Equality()

    A = @SMatrix rand(p,m)
    lin3 = LinearConstraint(n,m, A,b, Inequality(), n .+ (1:m))
    @test RD.evaluate(lin3, z) ≈ A*u - b
    RD.evaluate!(lin3, c, z)
    c ≈ A*u - b
    ∇c = TO.gen_jacobian(lin3)
    RD.jacobian!(lin3, ∇c, c, z)
    @test ∇c ≈ [zeros(p,n) A]
end


#--- Circle/Sphere Constraints
@testset "Circle/Sphere Constraints" begin
    xc = SA[1,1,1]
    yc = SA[1,2,3]
    r  = SA[1,1,1]
    cir = CircleConstraint(n, xc, yc, r)
    c = zeros(3)
    @test RD.evaluate(cir, z) ≈ -((x[1] .- xc).^2 + (x[2] .- yc).^2 .- r.^2)
    RD.evaluate!(cir, c, z)
    @test c ≈ -((x[1] .- xc).^2 + (x[2] .- yc).^2 .- r.^2)
    ∇c = TO.gen_jacobian(cir)
    RD.jacobian!(cir, ∇c, c, z)
    @test ∇c ≈ hcat(-2*(x[1] .- xc), -2*(x[2] .- yc), zeros(3,n-2))
    @test cir isa CircleConstraint{3,Int}
    @test cir isa TO.StateConstraint
    @test RD.output_dim(cir) == 3
    @test state_dim(cir) == n
    @test_throws RobotDynamics.NotImplementedError control_dim(cir)
    @test TO.widths(cir) == (n,)

    cir_ = CircleConstraint(n, Float64.(xc), yc, r)
    @test cir_ isa CircleConstraint{3,Float64}
    cir_ = CircleConstraint{3,Float64}(n, xc, yc, r)
    @test cir_ isa CircleConstraint{3,Float64}

    cir2 = CircleConstraint(n, Float64.(xc), yc, r, 2, 3)
    @test RD.evaluate(cir2, z) ≈ -((x[2] .- xc).^2 + (x[3] .- yc).^2 .- r.^2)
    ∇c = TO.gen_jacobian(cir2)
    RD.jacobian!(cir2, ∇c, c, z)
    @test ∇c ≈ hcat(zeros(3), -2*(x[2] .- xc), -2*(x[3] .- yc), zeros(3,n-3))

    @test_throws AssertionError CircleConstraint(n, push(xc,2), yc, r)


    zc = SA[3,3,3]
    sph = SphereConstraint{3,Int}(n, xc, yc, zc, r)
    @test RD.evaluate(sph, z) ≈ -((x[1] .- xc).^2 + (x[2] .- yc).^2  .+ (x[3] .- zc).^2 .- r.^2)
    ∇c = TO.gen_jacobian(sph)
    RD.jacobian!(sph, ∇c, c, z)
    @test ∇c ≈ hcat(-2*(x[1] .- xc), -2*(x[2] .- yc), -2*(x[3] .- zc), zeros(3,n-3))
    @test sph isa SphereConstraint{3,Int}
    @test sph isa TO.StateConstraint
    @test RD.output_dim(sph) == 3

    sph_ = SphereConstraint(n, Float64.(xc), yc, zc, r)
    @test sph_ isa SphereConstraint{3,Float64}
    sph_ = SphereConstraint{3,Float64}(n, xc, yc, zc, r)
    @test sph_ isa SphereConstraint{3,Float64}

    sph2 = SphereConstraint(n, Float64.(xc), yc, zc, r, 2, 3, 1)
    @test sph2.zi == 1
    @test RD.evaluate(sph2, z) ≈ -((x[2] .- xc).^2 + (x[3] .- yc).^2 + (x[1] .- zc).^2 .- r.^2)
    RD.evaluate!(sph2, c, z) 
    @test c ≈ -((x[2] .- xc).^2 + (x[3] .- yc).^2 + (x[1] .- zc).^2 .- r.^2)
    ∇c = TO.gen_jacobian(sph2)
    RD.jacobian!(sph2, ∇c, c, z)
    @test ∇c ≈ hcat(-2*(x[1] .- zc), -2*(x[2] .- xc), -2*(x[3] .- yc), zeros(3,n-3))
end


#--- Collision Constraint
@testset "Collision Constraint" begin
    x1 = SA[1,2]
    x2 = SA[3,4]
    col = CollisionConstraint(n, x1, x2, 2.)
    d = x[x1] - x[x2]
    c = zeros(1)
    @test RD.evaluate(col, z) ≈ SA[4 - d'd]
    RD.evaluate!(col, c, z)
    @test c[1] ≈ 4 - d'd
    ∇c = TO.gen_jacobian(col)
    RD.jacobian!(col, ∇c, c, z)
    @test ∇c ≈ [-2d' 2d']
    @test RD.output_dim(col) == 1

    col_ = CollisionConstraint(n, x1, x2, 1)
    @test col_.radius isa Float64
    col_ = CollisionConstraint(n, 1:2, x2, 1)
    @test col_.x1 isa SVector{2,Int}
    @test_throws AssertionError CollisionConstraint(n, 1:2, 1:3, 1.0)
end


#--- Norm Constraint
@testset "Norm Constraint" begin
    ncon = NormConstraint(n,m, 2.0, Inequality(), 1:n)
    c = zeros(1)
    @test RD.evaluate(ncon, z) ≈ [x'x - 2^2]
    RD.evaluate!(ncon, c, z)
    @test c ≈ [x'x - 4]
    ∇c = TO.gen_jacobian(ncon)
    TO.jacobian!(ncon, ∇c, c, z)
    @test ∇c ≈ [2x; 0]'

    @test RD.output_dim(ncon) == 1
    @test TO.widths(ncon) == (n+m,)
    @test TO.sense(ncon) == Inequality()

    ncon2 = NormConstraint(n,m, 2.0, Inequality(), :state)
    @test RD.evaluate(ncon, z) ≈ RD.evaluate(ncon2, z)

    ncon2 = NormConstraint(n, m, 3.0, Equality(), :control)
    @test RD.evaluate(ncon2, z) ≈ [u'u - 3^2]
    ∇c = TO.gen_jacobian(ncon2)
    TO.jacobian!(ncon2, ∇c, c, z)
    @test ∇c ≈ [zeros(n); 2u]'

    ncon3 = NormConstraint(n, m, 4.0, Inequality(), SA[1,3,5])
    @test RD.evaluate(ncon3, z) ≈ [x[1]^2 + x[3]^2 + u'u - 4^2]
    ∇c = TO.gen_jacobian(ncon3)
    RD.jacobian!(ncon3, ∇c, c, z)
    @test ∇c ≈ [2x[1] 0 2x[3] 0 2u[1]]
end


#--- Bound Constraint
@testset "Bound Constraint" begin
    xmin = -@SVector rand(n)
    xmax = +@SVector rand(n)
    umin = -@SVector rand(m)
    umax = +@SVector rand(m)

    bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)
    c = zeros(RD.output_dim(bnd))
    @test RD.evaluate(bnd, z) ≈ [x - xmax; u - umax; xmin - x; umin - u]
    RD.evaluate!(bnd, c, z)
    bnd.i_max
    ∇c = TO.gen_jacobian(bnd)
    RD.jacobian!(bnd, ∇c, c, z)
    @test ∇c ≈ [I(n+m); -I(n+m)]
    @test RD.output_dim(bnd) == 2(n+m)
    @test TO.widths(bnd) == (n+m,)
    @test TO.upper_bound(bnd) == [xmax; umax]
    @test TO.lower_bound(bnd) == [xmin; umin]
    @test TO.is_bound(bnd) == true

    xmin = pop(pushfirst(xmin, -Inf))
    umax = popfirst(push(umax, Inf))
    bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)
    c = zeros(RD.output_dim(bnd))
    @test RD.evaluate(bnd, z) ≈ [x - xmax; u[1:end-1] - umax[1:end-1];
        xmin[2:end] - x[2:end]; umin - u]
    RD.evaluate!(bnd, c, z)
    @test c ≈ RD.evaluate(bnd, z)
    ∇c = TO.gen_jacobian(bnd)
    RD.jacobian!(bnd, ∇c, c, z)
    iz = ones(Bool,2(n+m))
    iz[n+1] = 0
    iz[n+m+1] = 0
    @test ∇c ≈ [I(n+m); -I(n+m)][iz, :]
    @test RD.output_dim(bnd) == 2(n+m) - 2
    @test TO.widths(bnd) == (n+m,)
    @test TO.upper_bound(bnd) == [xmax; umax]
    @test TO.lower_bound(bnd) == [xmin; umin]
    @test TO.is_bound(bnd) == true

    bnd_ = BoundConstraint(n,m, x_min=-10, x_max=10, u_min=umin, u_max=umax)
    c = zeros(RD.output_dim(bnd_))
    @test RD.evaluate(bnd_, z) ≈ [x .- 10; u[1:end-1] - umax[1:end-1];
        -10 .- x; umin - u]
    RD.evaluate!(bnd_, c, z)
    @test c ≈ RD.evaluate(bnd_, z)
    bnd_ = BoundConstraint(n,m, x_min=-10, x_max=10, u_min=Vector(umin), u_max=umax)
    @test RD.evaluate(bnd_, z) ≈ [x .- 10; u[1:end-1] - umax[1:end-1];
        -10 .- x; umin - u]
    bnd_ = BoundConstraint(n,m, x_min=-10, x_max=10, u_min=Vector(umin), u_max=MVector(umax))
    @test RD.evaluate(bnd_, z) ≈ [x .- 10; u[1:end-1] - umax[1:end-1];
        -10 .- x; umin - u]

    xmin = -rand(1:10,n)
    xmax = rand(1:10,n)
    bnd_ = BoundConstraint(n,m, x_min=xmin, x_max=xmax)
    @test RD.evaluate(bnd_, z) ≈ [x .- xmax; xmin .- x]

    @test_throws ArgumentError BoundConstraint(n,m, x_min=10, x_max=-10, u_min=umin, u_max=umax)
end


#--- Indexed Constraint
@testset "Indexed Constraint" begin
    xmin = -@SVector rand(n)
    xmax = +@SVector rand(n)
    umin = -@SVector rand(m)
    umax = +@SVector rand(m)

    bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)

    n2,m2 = 2n, 2m
    x2,u2 = [x; 2x], [u; 2u]
    z2 = KnotPoint(x2,u2,z.dt,z.t)

    idx = TO.IndexedConstraint(n2,m2, bnd)
    c = zeros(RD.output_dim(idx))
    @test RD.evaluate(idx, z2) ≈ RD.evaluate(bnd, z)
    RD.evaluate!(idx, c, z2)
    @test c ≈ RD.evaluate(bnd, z)

    ∇c = TO.gen_jacobian(idx)
    ∇c0 = TO.gen_jacobian(bnd)
    RD.jacobian!(idx, ∇c, c, z2)
    RD.jacobian!(bnd, ∇c0, c, z)
    @test ∇c ≈ [∇c0[:,1:n] zeros(RD.output_dim(bnd), n) ∇c0[:,n+1:end] zeros(RD.output_dim(bnd), m)]

    @test RD.output_dim(idx) == RD.output_dim(bnd)
    @test TO.sense(idx) == TO.sense(bnd)
    @test TO.state_dim(idx) == 2n
    @test TO.control_dim(idx) == 2m
    @test TO.upper_bound(idx) == TO.upper_bound(bnd)
    @test TO.lower_bound(idx) == TO.lower_bound(bnd)
    @test TO.is_bound(idx) == true


    xc = SA[1,1,1]
    yc = SA[1,2,3]
    r  = SA[1,1,1]
    cir = CircleConstraint(n, xc, yc, r)

    idx = TO.IndexedConstraint(n2,m2, cir, n+1:2n, m+1:2m)
    c = zeros(RD.output_dim(idx))
    @test size(idx.A) == size(idx.∇c)
    @test isempty(idx.B)
    @test RD.evaluate(idx, z2) ≈ RD.evaluate(cir, 2z)
    RD.evaluate!(idx, c, z2)
    @test c ≈ RD.evaluate(cir, 2z)
    ∇c  = TO.gen_jacobian(idx)
    ∇c0 = TO.gen_jacobian(cir)
    TO.jacobian!(idx, ∇c, c, z2)
    TO.jacobian!(cir, ∇c0, c, 2z)
    @test ∇c ≈ [zeros(RD.output_dim(cir), n) ∇c0[:,1:n] zeros(RD.output_dim(cir), 2)]

    @test RD.output_dim(idx) == RD.output_dim(cir)
    @test TO.sense(idx) == TO.sense(cir)
    @test TO.state_dim(idx) == 2n
    @test TO.control_dim(idx) == 2m
    @test TO.upper_bound(idx) == TO.upper_bound(cir)
    @test TO.lower_bound(idx) == TO.lower_bound(cir)
    @test TO.is_bound(idx) == TO.is_bound(cir)

    # TODO: test IndexedConstraint with a ControlConstraint
end

using Rotations
@testset "QuatVecEq" begin
    model = Quadrotor()
    n,m = RD.dims(model)
    qf = Rotations.expm([1,0,0]*deg2rad(45))
    qcon = TO.QuatVecEq(n,m,qf)
    @test qcon.qind == 4:7
    @test_nowarn TO.QuatVecEq(n,m,MRP(qf))
    qcon2 = TO.QuatVecEq(n,m,qf,[1,2,3,4])
    @test qcon2.qind === SA[1,2,3,4]

    x,u = rand(model)
    z = RD.KnotPoint(x,u,0.0,NaN)
    q = RD.orientation(model, x)
    dq = RD.params(qf)'RD.params(q)
    sign(dq)
    Rotations.vector(q)
    @test RD.evaluate(qcon, x) ≈ -(sign(dq)*Rotations.vector(qf) - Rotations.vector(q))
    c = zeros(3)
    RD.evaluate!(qcon, c, x)
    @test c ≈ -(sign(dq)*Rotations.vector(qf) - Rotations.vector(q))
    c .= 0
    TO.evaluate_constraint!(RD.StaticReturn(), qcon, c, z)
    @test c ≈ -(sign(dq)*Rotations.vector(qf) - Rotations.vector(q))

    J = zeros(3,n)
    TO.constraint_jacobian!(RD.StaticReturn(), RD.ForwardAD(), qcon, J, c, z)
    @test J ≈ ForwardDiff.jacobian(x->RD.evaluate(qcon,x), RD.state(z))
    TO.constraint_jacobian!(RD.InPlace(), RD.ForwardAD(), qcon, J, c, z)
    @test J ≈ ForwardDiff.jacobian(x->RD.evaluate(qcon,x), RD.state(z))

    J .= 0
    TO.constraint_jacobian!(RD.InPlace(), RD.FiniteDifference(), qcon, J, c, z)
    @test J ≈ ForwardDiff.jacobian(x->RD.evaluate(qcon,x), RD.state(z)) atol=1e-6
end