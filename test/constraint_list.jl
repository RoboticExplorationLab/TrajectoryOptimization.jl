
@testset "Constraint List" begin
    model = Cartpole()
    n,m = RD.dims(model)
    N = 11
    x,u = rand(model)
    t,h = 1.1, 0.1
    z = KnotPoint(x,u,t,h)

    #--- Generate some constraints
    # Circle Constraint
    xc = SA[1,1,1]
    yc = SA[1,2,3]
    r  = SA[1,1,1]
    cir = CircleConstraint(n, xc, yc, r)

    # Goal Constraint
    xf = @SVector rand(n)
    goal = GoalConstraint(xf)

    # Linear Constraint
    p = 5
    A = @SMatrix rand(p,n+m)
    b = @SVector rand(p)
    lin = LinearConstraint(n,m,A,b, Inequality())

    # Bound Constraint
    xmin = -@SVector rand(n)
    xmax = +@SVector rand(n)
    umin = -@SVector rand(m)
    umax = +@SVector rand(m)
    bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)

    #--- Create a List
    cons = ConstraintList(n,m,N)
    add_constraint!(cons, cir, 1:N)
    @test cons.constraints[1] === cir
    @test cons[1] === cir
    @test cons.inds[1] == 1:N
    @test cons.p == fill(RD.output_dim(cir), N)

    add_constraint!(cons, goal, N)
    @test cons.constraints[2] === goal
    @test cons[2] === goal
    @test cons.inds[2] == N:N
    @test cons.p[1:N-1] == fill(RD.output_dim(cir), N-1)
    @test cons.p[end] == RD.output_dim(cir) + RD.output_dim(goal)

    add_constraint!(cons, lin, 1:4, 1)
    @test cons[1] === lin
    @test cons[2] === cir
    @test cons[end] === goal
    @test cons.inds[1] === 1:4
    @test cons.p[1:4] == fill(RD.output_dim(cir)+RD.output_dim(lin), 4)
    @test cons.p[5:N-1] == fill(RD.output_dim(cir), N-1-4)
    @test length(cons) == 3

    cons2 = copy(cons)
    add_constraint!(cons, bnd, 1:N-1)
    @test length(cons) == 4
    @test length(cons2) == 3

    @test TO.num_constraints(cons) === cons.p
    @test TO.num_constraints(cons2) !== cons.p
    @test cons[end] == bnd
    @test length(cons) == 4

    # Try adding a constraint with incorrect dimensions
    lin2 = LinearConstraint(2, 1, rand(3,2), rand(3), Inequality(), 1:2)
    @test_throws AssertionError add_constraint!(cons, lin2, 1:4)

    # Test iteration
    conlist = [lin, cir, goal, bnd]
    @test all(conlist .=== [con for con in cons])
    @test RD.output_dim.(cons) == RD.output_dim.(conlist)
    @test eltype(cons) == TO.AbstractConstraint
end
