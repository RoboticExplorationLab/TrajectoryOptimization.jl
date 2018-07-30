include("../iLQR.jl")
import iLQR: UnconstainedObjective, ConstrainedObjective
using Dynamics
using Base.Test

@testset "Objectives" begin
    """ Simple Pendulum """
    pendulum = Dynamics.pendulum[1]
    n = pendulum.n
    m = pendulum.m
    x0 = [0; 0];
    xf = [pi; 0]; # (ie, swing up)
    u0 = [1]
    Q = 1e-3*eye(n);
    Qf = 100*eye(n);
    R = 1e-3*eye(m);
    tf = 5

    @test_nowarn UnconstainedObjective(Q, R, Qf, tf, x0, xf)
    obj_uncon = iLQR.UnconstainedObjective(Q, R, Qf, tf, x0, xf)

    ### Constraints ###
    # Test defaults
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf)
    @test obj.u_min == [-Inf]
    @test obj.u_max == [Inf]
    @test obj.x_min == -[Inf,Inf]
    @test obj.x_max == [Inf,Inf]
    @test isa(obj.cI(x0,u0),Void)
    @test isa(obj.cE(x0,u0),Void)
    @test isa(obj.cI_N(x0),Void)
    @test isa(obj.cE_N(x0),Void)
    @test obj.p == 0
    @test obj.use_terminal_constraint == true
    @test obj.p_N == 2

    # Use scalar control constraints
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min=-1,u_max=1)
    @test obj.p == 2
    @test obj.p_N == 2

    # Single-sided
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_max=1)
    @test obj.p == 1
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min=1, u_max=Inf)
    @test obj.p == 1

    # Error testing
    @test_throws ArgumentError iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min=1, u_max=-1)
    @test_throws DimensionMismatch iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min=[1], u_max=[1,2])

    # State constraints
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        x_min=-[1,2], x_max=[1,2])
    @test obj.p == 4
    @test obj.pI == 4

    @test_throws DimensionMismatch obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        x_min=-[Inf,2,3,4], x_max=[1,Inf,3,Inf])
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        x_min=-[Inf,4], x_max=[3,Inf])
    @test obj.p == 2
    @test obj.pI == 2

    # Scalar to array constraint
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        x_min=-4, x_max=4)
    @test obj.p == 4


    # Custom constraints
    c(x,u) = x[2]+u[1]-2
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        cI=c)
    @test obj.p == 1
    @test obj.pI == 1
    obj = iLQR.ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        cE=c)
    @test obj.p == 1
    @test obj.pI == 0

    # Construct from unconstrained
    obj = iLQR.ConstrainedObjective(obj_uncon)
    @test obj.u_min == [-Inf]
    @test obj.u_max == [Inf]
    @test obj.x_min == -[Inf,Inf]
    @test obj.x_max == [Inf,Inf]
    @test isa(obj.cI(x0,u0),Void)
    @test isa(obj.cE(x0,u0),Void)
    @test isa(obj.cI_N(x0),Void)
    @test isa(obj.cE_N(x0),Void)
    @test obj.p == 0
    @test obj.use_terminal_constraint == true
    @test obj.p_N == 2

    obj = iLQR.ConstrainedObjective(obj_uncon, u_min=-1)
    @test obj.p == 1

    # Update objectve
    obj = iLQR.update_objective(obj, u_max=2, x_max = 4)
    @test obj.p == 4


end
