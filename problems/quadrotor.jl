# model
model = Dynamics.Quadrotor2{UnitQuaternion{Float64,VectorPart}}()
# model = Dynamics.quadrotor_euler
n,m = size(model)

# discretization
N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

q0 = @SVector [1,0,0,0]

x0_pos = @SVector [0., 0., 10.]
x0 = [x0_pos; q0; @SVector zeros(6)]

xf = zero(x0)
xf_pos = @SVector [0., 60., 10.]
xf = [xf_pos; q0; @SVector zeros(6)]

# cost
Qdiag = fill(1e-3,n)
Qdiag[4:7] .= 1e-2
Q = Diagonal(SVector{13}(Qdiag))
R = (1.0e-4)*Diagonal(@SVector ones(m))
Qf = 1000.0*Diagonal(@SVector ones(n))

u_min = 0.
u_max = 50.
x_max = Inf*ones(model.n)
x_min = -Inf*ones(model.n)

x_max[1:3] = [25.0; Inf; 20]
x_min[1:3] = [-25.0; -Inf; 0.]
bnd = BoundConstraint(n,m,u_min=u_min)

xf_no_quat_U = Vector(xf)
xf_no_quat_L = Vector(xf)
xf_no_quat_U[4:7] .= Inf
xf_no_quat_L[4:7] .= -Inf
xf_no_quat_U[8:10] .= 0.
xf_no_quat_L[8:10] .= 0.
bnd_xf = BoundConstraint(n,m, x_min=xf_no_quat_L, x_max=xf_no_quat_U)
inds_no_quat = SVector{n-4}(deleteat!(collect(1:n), 4:7))
goal = GoalConstraint(xf, inds_no_quat)

con_bnd = ConstraintVals(bnd, 1:N-1)
con_xf = ConstraintVals(bnd_xf, N:N)
con_goal = ConstraintVals(goal, N:N)
conSet = ConstraintSet(n,m,[con_bnd, con_goal], N)


U_hover = [0.5*9.81/4.0*(@SVector ones(m)) for k = 1:N-1] # initial hovering control trajectory
obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs

quadrotor_static = Problem(model, obj, xf, tf, constraints=conSet, U0=U_hover, x0=x0)


function gen_quadrotor_zigzag(Rot; use_rot=Rot<:UnitQuaternion, costfun=:Quadratic,
        normcon=false)
    model = Dynamics.Quadrotor2{Rot}(use_rot=use_rot)
    n,m = size(model)

    max_con_viol = 1.0e-3
    T = Float64
    verbose = true

    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
        cost_tolerance=1e-4,
        iterations=300)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
        opts_uncon=opts_ilqr,
        iterations=40,
        cost_tolerance=1.0e-5,
        cost_tolerance_intermediate=1.0e-4,
        constraint_tolerance=max_con_viol,
        penalty_scaling=10.,
        penalty_initial=1.)

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [0., -10., 1.]
    x0 = Dynamics.build_state(model, x0_pos, I(UnitQuaternion), zeros(3), zeros(3))

    # cost
    costfun == :QuatLQR ? sq = 0 : sq = 1
    rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
    Q_diag = Dynamics.fill_state(model, 1e-5, 1e-5*sq, 1e-3, 1e-3)
    Q = Diagonal(Q_diag)
    R = Diagonal(@SVector fill(1e-4,m))
    q_nom = I(UnitQuaternion)
    v_nom, ω_nom = zeros(3), zeros(3)
    x_nom = Dynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)

    if costfun == :QuatLQR
        cost_nom = QuatLQRCost(Q, R, x_nom, w=0.0)
    elseif costfun == :ErrorQuad
        cost_nom = ErrorQuadratic(model, Diagonal(Q_diag[rm_quat]), R, x_nom)
    else
        cost_nom = LQRCost(Q, R, x_nom)
    end

    # waypoints
    wpts = [(@SVector [10,0,1.]),
            (@SVector [-10,0,1.]),
            (@SVector [0,10,1.])]
    times = [33, 66, 101]
    Qw_diag = Dynamics.fill_state(model, 1e3,1*sq,1,1)
    Qf_diag = Dynamics.fill_state(model, 10., 100*sq, 10, 10)
    xf = Dynamics.build_state(model, wpts[end], I(UnitQuaternion), zeros(3), zeros(3))

    costs = map(1:length(wpts)) do i
        r = wpts[i]
        xg = Dynamics.build_state(model, r, q_nom, v_nom, ω_nom)
        if times[i] == N
            Q = Diagonal(Qf_diag)
            w = 40.0
        else
            Q = Diagonal(1e-3*Qw_diag)
            w = 0.1
        end
        if costfun == :QuatLQR
            QuatLQRCost(Q, R, xg, w=w)
        elseif costfun == :ErrorQuad
            Qd = diag(Q)
            ErrorQuadratic(model, Diagonal(Qd[rm_quat]), R, xg)
        else
            LQRCost(Q, R, xg)
        end
    end

    costs_all = map(1:N) do k
        i = findfirst(x->(x ≥ k), times)
        if k ∈ times
            costs[i]
        else
            cost_nom
        end
    end

    obj = Objective(costs_all)

    # Initialization
    u0 = @SVector fill(0.5*9.81/4, m)
    U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

    # Constaints
    conSet = ConstraintSet(n,m,N)
    if normcon
        if use_rot == :slack
            add_constraint!(conSet, QuatSlackConstraint(), 1:N-1)
        else
            add_constraint!(conSet, QuatNormConstraint(), 1:N-1)
            u0 = [u0; (@SVector [1.])]
        end
    end
    bnd = BoundConstraint(n,m, u_min=0.0, u_max=12.0)
    add_constraint!(conSet, bnd, 1:N-1)

    # Problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
    initial_controls!(prob, U_hover)

    return prob
end
