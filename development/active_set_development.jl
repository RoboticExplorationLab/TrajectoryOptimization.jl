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

_backwardpass_active_set!(results1,solver1)

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


    S = [k != N ? zeros(n+p,n+p) : zeros(n+pN,n+pN) for k = 1:N]
    s = [k != N ? zeros(n+p) : zeros(n+pN) for k = 1:N]

    # Boundary Conditions
    C = res.C; Iμ = res.Iμ; λ = res.λ; CxN = res.Cx_N

    Sxx = Wf + CxN'*res.IμN*CxN
    Sxλ = CxN'
    Sλλ = zeros(pN,pN)

    Sx = Wf*(X[N] - xf) + CxN'*(res.IμN*res.CN + res.λN)
    Sλ = res.CN

    S[N][1:n,1:n] = Sxx
    S[N][1:n,n+1:n+pN] = Sxλ
    S[N][n+1:n+pN,1:n] = Sxλ'
    S[N][n+1:n+pN,n+1:n+pN] = Sλλ

    s[N][1:n] = Sx
    s[N][n+1:n+pN] = Sλ

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
        Cx, Cu = res.Cx[k], res.Cu[k]
        Lx += Cx'*(Iμ[k]*C[k] + λ[k])
        Lu += Cu'*(Iμ[k]*C[k] + λ[k])
        Lxx += Cx'*Iμ[k]*Cx
        Luu += Cu'*Iμ[k]*Cu
        Lux += Cu'*Iμ[k]*Cx

        # multiplier expansion terms
        Lλ = res.C[k]
        Lxλ = res.Cx[k]'
        Lλλ = zeros(p,p)
        Lλu = res.Cu[k]

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = Lx + fdx'*s[k+1][1:n]
        Qu = Lu + fdu'*s[k+1][1:n]
        Qxx = Lxx + fdx'*S[k+1][1:n,1:n]*fdx
        Quu = Luu + fdu'*S[k+1][1:n,1:n]*fdu
        Qux = Lux + fdu'*S[k+1][1:n,1:n]*fdx

        # regularization
        Quu_reg = Quu + res.ρ[1]*I
        Qux_reg = Qux
        Luu_reg = Luu + res.ρ[1]*I
        Lux_reg = Lux

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

        # linearized dynamics for cost-to-go
        S̄xx = fdx'*Sxx*fdx
        S̄xλ = fdx'*Sxλ
        S̄xu = fdx'*Sxx*fdu
        S̄λλ = Sλλ
        S̄λu = Sxλ'*fdu
        S̄uu = fdu'*Sxx*fdu

        S̄x = fdx'*Sx
        S̄λ = Sλ
        S̄u = fdu'*Sx

        # get active set constraints and Jacobians
        active_set_idx_1 = res.active_set[k] # active set indices at index k
        k != N-1 ? active_set_idx_2 = res.active_set[k+1] : active_set_idx_2 = ones(Bool,pN) # active set indices at index k+1

        c_active_set_1 = res.C[k][active_set_idx_1] # active set constraints at index k
        k != N-1 ? c_active_set_2 = res.C[k+1][active_set_idx_2] : c_active_set_2 = res.CN[active_set_idx_2] # active set constraints at index k+1
        c_set_1 = res.C[k]
        k != N-1 ? c_set_2 = res.C[k+1] : c_set_2 = res.CN

        cx_active_set_1 = res.Cx[k][active_set_idx_1,:] # active set Jacobians at index k
        cx_active_set_2 = S̄xλ[:,active_set_idx_2]' # active set Jacobians at index k+1
        cx_set_1 = res.Cx[k]
        cx_set_2 = S̄xλ'

        cu_active_set_1 = res.Cu[k][active_set_idx_1,:] # active set Jacobians at index k
        cu_active_set_2 = S̄λu[active_set_idx_2,:] # active set Jacobians at index k+1
        cu_set_1 = res.Cu[k]
        cu_set_2 = S̄λu

        # calculate multiplier gains
        tmp = cu_active_set_2*(Quu_reg\cu_active_set_2')

        res.Kλ[k+1][active_set_idx_2,:] = -tmp\(cu_active_set_2*(Quu_reg\Qux_reg) - cx_active_set_2)
        res.Mλ[k+1][active_set_idx_2,active_set_idx_1] = -tmp\(cu_active_set_2*(Quu_reg\cu_active_set_1'))
        res.dλ[k+1][active_set_idx_2] = -tmp\(cu_active_set_2*(Quu_reg\Qu) - c_active_set_2)

        # reset inactive gains to zero (this may be unnecessary)
        res.Kλ[k+1][.!active_set_idx_2,:] .= 0.
        res.Mλ[k+1][.!active_set_idx_2,.!active_set_idx_1] .= 0.
        res.dλ[k+1][.!active_set_idx_2] .= 0.

        if k != 1
            # calculate control gains
            res.K[k] = -Quu_reg\(Qux_reg + cu_set_2'*res.Kλ[k+1])
            res.M[k] = -Quu_reg\(cu_set_1' + cu_set_2'*res.Mλ[k+1])
            res.d[k] = -Quu_reg\(Qu + cu_set_2'*res.dλ[k+1])
        else
            # calculate final multiplier gains
            tmp = cu_active_set_1*(Quu_reg\(cu_active_set_1' + cu_active_set_2'*res.Mλ[k+1][active_set_idx_2,active_set_idx_1]))
            res.Kλ[1][active_set_idx_1,:] = -tmp\(cu_active_set_1*(Quu_reg\(Qux_reg + cu_active_set_2'*res.Kλ[k+1][active_set_idx_2,:])) - cx_active_set_1)
            res.dλ[1][active_set_idx_1] = -tmp\(cu_active_set_1*(Quu_reg\(Qu + cu_active_set_2'*res.dλ[k+1][active_set_idx_2])) - c_active_set_1)

            # reset inactive gains to zero (this may be unnecessary)
            res.Kλ[1][.!active_set_idx_1,:] .= 0.
            res.dλ[1][.!active_set_idx_1] .= 0.

            # calculate final control gains
            res.K[1] = -Quu_reg\(Qux_reg + cu_set_1'*res.Kλ[1] + cu_set_2'*res.Kλ[k+1] + cu_set_2'*res.Mλ[k+1]*res.Kλ[1])
            res.d[1] = -Quu_reg\(Qu + cu_set_1'*res.dλ[1] + cu_set_2'*res.dλ[k+1] + cu_set_2'*res.Mλ[k+1]*res.dλ[1])
        end

        # plug in multiplier equation
        Ŝxx = S̄xx + S̄xλ*res.Kλ[k+1] + res.Kλ[k+1]'*S̄xλ' + res.Kλ[k+1]'*S̄λλ*res.Kλ[k+1]
        Ŝxλ = S̄xλ*res.Mλ[k+1] + res.Kλ[k+1]'*S̄λλ*res.Mλ[k+1]
        Ŝxu = res.Kλ[k+1]'*S̄λu + S̄xu
        Ŝλλ = res.Mλ[k+1]'*S̄λλ*res.Mλ[k+1]
        Ŝλu = res.Mλ[k+1]'*S̄λu
        Ŝuu = S̄uu

        Ŝx = S̄xλ*res.dλ[k+1] + res.Kλ[k+1]'*S̄λλ*res.dλ[k+1] + S̄x + res.Kλ[k+1]'*S̄λ
        Ŝλ = res.Mλ[k+1]'*S̄λλ*res.dλ[k+1] + res.Mλ[k+1]'*S̄λ
        Ŝu = S̄λu'*res.dλ[k+1] + S̄u

        Δv[1] += res.dλ[k+1]'*S̄λ
        Δv[2] += 0.5*res.dλ[k+1]'*S̄λλ*res.dλ[k+1]

        # calculate cost-to-go
        Q̂xx = Lxx + Ŝxx
        Q̂xλ = Lxλ + Ŝxλ
        Q̂ux = Lux_reg + Ŝxu'
        Q̂uu = Luu_reg + Ŝuu
        Q̂λu = Lλu + Ŝλu
        Q̂λλ = Lλλ + Ŝλλ
        Q̂x = Lx + Ŝx
        Q̂u = Lu + Ŝu
        Q̂λ = Lλ + Ŝλ

        Sxx = Q̂xx + Q̂ux'*res.K[k] + res.K[k]'*Q̂ux + res.K[k]'*Q̂uu*res.K[k]
        Sxλ = Q̂xλ + Q̂ux'*res.M[k] + res.K[k]'*Q̂λu' + res.K[k]'*Q̂uu*res.M[k]
        Sλλ = Q̂λλ + res.M[k]'*Q̂λu' + res.M[k]'*Q̂uu*res.M[k]

        Sx = Q̂x + Q̂ux'*res.d[k] + res.K[k]'*Q̂uu*res.d[k] + res.K[k]'*Q̂u
        Sλ = Q̂λ + Q̂λu*res.d[k] + res.M[k]'*Q̂uu*res.d[k] + res.M[k]'*Q̂u

        S[k][1:n,1:n] = Sxx
        S[k][1:n,n+1:n+p] = Sxλ
        S[k][n+1:n+p,1:n] = Sxλ'
        S[k][n+1:n+p,n+1:n+p] = Sλλ

        s[k][1:n] = Sx
        s[k][n+1:n+p] = Sλ

        Δv[1] += res.d[k]'*Q̂u
        Δv[2] += 0.5*res.d[k]'*Q̂uu*res.d[k]

        if k == 1
            Δv[1] += res.dλ[1]'*Sλ
            Δv[2] += 0.5*res.dλ[1]'*Sλλ*res.dλ[1]
        end

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

a = 1
