using Test

##
model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!

u_min_pendulum = -2
u_max_pendulum = 2
x_min_pendulum = [-20;-20]
x_max_pendulum = [20; 20]

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)

model = model_pendulum
obj = obj_con_pendulum

# Solver
intergration = :rk4
N = 25
solver1 = Solver(model,obj,integration=intergration,N=N)
solver1.opts.constrained = true
X0 = line_trajectory(solver1)
U0 = ones(solver1.model.m,solver1.N)

results1 = init_results(solver1,X0,U0)
rollout!(results1,solver1)
calculate_jacobians!(results1, solver1)

p,pI,pE = get_num_constraints(solver1)

idx = Array(1:p)

p_idx = idx[results1.active_set[1]]

a = 1

function _backwardpass_active_set!(res::SolverVectorResults,solver::Solver)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    pN = n
    W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf

    solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d;
    L = res.L; l = res.l

    # Useful linear indices
    idx = Array(1:2*(n+mm))
    Idx = reshape(Array(1:(n+mm)^2),n+mm,n+mm)

    x_idx = idx[1:n]
    u_idx = idx[n+1:n+m]
    ū_idx = idx[n+1:n+mm]

    p_idx = Array(1:p)

    S = [k != N ? zeros(n+p,n+p) : zeros(n+pN,n+pN) for k = 1:N]
    s = [k != N ? zeros(n+p) : zeros(n+pN) for k = 1:N]

    # Boundary Conditions
    C = res.C; Iμ = res.Iμ; λ = res.λ; CxN = res.Cx_N

    S[N][1:n,1:n] = Wf + CxN'*res.IμN*CxN
    S[N][1:n,n+1:n+pN] = Cx_N'
    S[N][n+1:n+pN,1:n] = Cx_N
    S[N][n+1:n+pN,n+1:n+pN] = zeros(pN,pN)

    s[N][1:n] = Wf*(X[N] - xf) + CxN'*(res.IμN*res.CN + res.λN)
    s[N][n+1:n+pN] = res.CN

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Backward pass
    k = N-1
    while k >= 1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        x = X[k]
        u = U[k][1:m]

        ℓ1 = ℓ(x,u,W,R,xf)
        ℓx = W*(x - xf)
        ℓu = R*u
        ℓxx = W
        ℓuu = R
        ℓux = zeros(m,n)

        # Assemble expansion

        Lx = dt*ℓx
        Lu = dt*ℓu
        Lxx = dt*ℓxx
        Luu = dt*ℓuu
        Lux = dt*ℓux

        # Minimum time expansion components
        if solver.opts.minimum_time
            h = U[k][m̄]

            l[n+m̄] = 2*h*(ℓ1 + R_minimum_time)

            tmp = 2*h*ℓu
            L[u_idx,m̄] = tmp
            L[m̄,u_idx] = tmp'
            L[m̄,m̄] = 2*(ℓ1 + R_minimum_time)

            L[m̄,x_idx] = 2*h*ℓx'
        end

        # Infeasible expansion components
        if solver.opts.infeasible
            l[n+m̄+1:n+mm] = R_infeasible*U[k][m̄+1:m̄+n]
            L[n+m̄+1:n+mm,n+m̄+1:n+mm] = R_infeasible
        end

        # Final expansion terms
        if solver.opts.minimum_time || solver.opts.infeasible
            l[u_idx] = Lu

            L[u_idx,u_idx] = Luu
            L[u_idx,x_idx] = Lux

            Lu = l[n+1:n+mm]
            Luu = L[n+1:n+mm,n+1:n+mm]
            Lux = L[n+1:n+mm,1:n]
        end

        # Constraints
        if res isa ConstrainedIterResults
            Cx, Cu = res.Cx[k], res.Cu[k]
            Lx += Cx'*(Iμ[k]*C[k] + λ[k])
            Lu += Cu'*(Iμ[k]*C[k] + λ[k])
            Lxx += Cx'*Iμ[k]*Cx
            Luu += Cu'*Iμ[k]*Cu
            Lux += Cu'*Iμ[k]*Cx
        end

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = Lx + fdx'*s[k+1][1:n]
        Qu = Lu + fdu'*s[k+1][1:n]
        Qxx = Lxx + fdx'*S[k+1][1:n,1:n]*fdx
        Quu = Luu + fdu'*S[k+1][1:n,1:n]*fdu
        Qux = Lux + fdu'*S[k+1][1:n,1:n]*fdx

        if solver.opts.regularization_type == :state
            Luu_reg = Luu + res.ρ[1]*fdu'*fdu
            Lux_reg = Lux + res.ρ[1]*fdu'*fdx
            Quu_reg = Quu + res.ρ[1]*fdu'*fdu
            Qux_reg = Qux + res.ρ[1]*fdu'*fdx
        elseif solver.opts.regularization_type == :control
            Quu_reg = Quu + res.ρ[1]*I
            Qux_reg = Qux
            Luu_reg = Luu + res.ρ[1]*I
            Lux_reg = Lux
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            regularization_update!(res,solver,:increase)

            # reset backward pass
            k = N-1
            Δv[1] = 0.
            Δv[2] = 0.
            continue
        end

        # linear dynamics for cost-to-go
        S̄xx = fdx'*Sxx*fdx
        S̄xλ = fdx'*Sxλ
        S̄xu = fdx'*Sxx*fdu
        S̄λx = Sλx*fdx
        S̄λλ = Sλλ
        S̄λu = Sλx*fdu
        S̄ux = fdu'*Sxx*fdx
        S̄uλ = fdu'*Sxλ
        S̄uu = fdu'*Sxx*fdu

        S̄x = fdx'*Sx
        S̄λ = Sλ
        S̄u = fdu'*Sx

        # calculate multiplier gains
        res.Kλ[k+1] = -(S̄λu*(Quu_reg\S̄λu'))\(S̄λu*(Quu_reg\Qux_reg) - S̄λx)
        res.Mλ[k+1] = -(S̄λu*(Quu_reg\S̄λu'))\(S̄λu*(Quu_reg\res.Cu[k]'))
        res.dλ[k+1] = -(S̄λu*(Quu_reg\S̄λu'))\(S̄λu*(Quu_reg\Qu) - res.C[k+1])

        if k != 1
            # calculate control gains
            res.K[k] = -Quu_reg\(Qux_reg + S̄λu'*res.Kλ[k+1])
            res.M[k] = -Quu_reg\(res.Cu[k] + S̄λu'*res.Mλ[k+1])
            res.d[k] = -Quu_reg\(Qu + S̄λu'*res.dλ[k+1])
        else
            res.Kλ[1] = -(res.Cu[k]*(Quu_reg\(res.Cu[k]' + S̄λu*res.Mλ[k+1])))\(res.Cu[k]*(Quu_reg\(Qux_reg + S̄λu*res.Kλ[k+1])) - S̄λx)
            # res.Mλ[1] = -()\()
            res.dλ[1] = -(res.Cu[k]*(Quu_reg\(res.Cu[k]' + S̄λu*res.Mλ[k+1])))\(res.Cu[k]*(Quu_reg\(Qu + S̄λu*res.dλ[k+1])) - res.C[k+1])

            res.K[1] = -Quu_reg\(Qux_reg + res.Cu[k]*res.Kλ[1] + S̄λu'*res.Kλ[k+1] + S̄λu'*res.Mλ[k+1]*res.Kλ[1])
            # res.M[1] = -()\()
            res.d[1] = -Quu_reg\(Qu + res.Cu[k]*res.dλ[1] + S̄λu'*res.dλ[k+1] + S̄λu'*res.Mλ[k+1]*res.dλ[1])
        end

        # plug in multiplier equation
        Ŝxx = S̄xx + S̄xλ*res.Kλ[k+1] + res.Kλ[k+1]'*S̄xλ + res.Kλ[k+1]'*S̄λλ*res.Kλ[k+1]
        Ŝxλ = S̄xλ*res.Mλ[k+1] + res.Kλ[k+1]'*S̄λλ*res.Mλ[k+1]
        Ŝxu = res.Kλ[k+1]'*S̄λu + S̄xu
        Ŝλx = res.Mλ[k+1]'*S̄λx + res.Mλ[k+1]'*S̄λλ*res.Kλ[k+1]
        Ŝλλ = res.Mλ[k+1]'*S̄λλ*res.Mλ[k+1]
        Ŝλu = res.Mλ[k+1]'*S̄λu
        Ŝux = S̄ux + S̄uλ*res.Kλ[k+1]
        Ŝuλ = S̄uλ*res.Mλ[k+1]
        Ŝuu = S̄uu

        Ŝx = res.dλ[k+1]'*S̄λx + res.dλ[k+1]'*S̄λλ*res.Kλ[k+1] + S̄x + S̄λ'*res.Kλ[k+1]
        Ŝλ = res.dλ[k+1]'*S̄λλ*res.Mλ[k+1] + S̄λ'*res.Mλ[k+1]
        Ŝu = res.dλ[k+1]'*S̄λu + S̄u'

        Δv[1] += res.dλ[k+1]'*S̄λ
        Δv[2] += 0.5*res.dλ[k+1]'*S̄λλ*res.dλ[k+1]

        # calculate cost-to-go
        Sxx = (Lxx + Ŝxx) + (Lxu + Ŝxu)*res.K[k] + res.K[k]'*(Lux + Ŝux) + res.K[k]'*(Luu + Ŝuu)*res.K[k]
        Sxλ = (Lxu + Ŝxu) + (Lxu + Ŝxu)*res.M[k] + rs.K[k]'*(Luλ + Ŝuλ) + res.K[k]'*(Luu + Ŝuu)*res.M[k]

        Sλx = Sxλ'
        Sλλ = (Lλλ + Ŝλλ) + res.M[k]'*(Luλ + Ŝuλ) + res.M[k]'*(Luu + Ŝuu)*res.M[k]

        Sx = (Lx + Ŝx) + res.d[k]'*(Lux + Ŝux) + res.d[k]*(Luu + Ŝuu)*res.K[k] + (Lu + Ŝu)'*res.K[k]
        Sλ = (Lλ + Ŝλ) + res.d[k]'*(Luλ + Ŝuλ) + res.d[k]*(Luu + Ŝuu)*res.M[k] + (Lu + Ŝu)'*res.M[k]

        S[k][1:n,1:n] = Sxx
        S[k][1:n,n+1:n+pN] = Sxλ
        S[k][n+1:n+pN,1:n] = Sλx
        S[k][n+1:n+pN,n+1:n+pN] = Sλλ

        s[k][1:n] = Sx
        s[k][n+1:n+pN] = Sλ

        Δv[1] += res.d[k]'*(Lu + Ŝu)
        Δv[2] += 0.5*res.d[k]'*(Luu + Ŝuu)*res.d[k]

        # # Compute gains
        # K[k] = -Quu_reg\Qux_reg
        # d[k] = -Quu_reg\Qu
        #
        # # Calculate cost-to-go (using unregularized Quu and Qux)
        # s[k] = Qx + K[k]'*Quu*d[k] + K[k]'*Qu + Qux'*d[k]
        # S[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'*K[k]
        # S[k] = 0.5*(S[k] + S[k]')
        #
        # # calculated change is cost-to-go over entire trajectory
        # Δv[1] += d[k]'*Qu
        # Δv[2] += 0.5*d[k]'*Quu*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end
