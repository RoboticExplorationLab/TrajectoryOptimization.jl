function Quadrotor(scenario=:zigzag, Rot=UnitQuaternion{Float64,CayleyMap};
        use_rot=Rot<:UnitQuaternion, costfun=:Quadratic, normcon=false)
    if scenario == :zigzag
        model = Dynamics.Quadrotor2{Rot}(use_rot=use_rot)
        n,m = size(model)

        opts = SolverOptions()

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
        rollout!(prob)

        return prob, opts
    end
end
