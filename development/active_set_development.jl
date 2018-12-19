using Test

##
model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!

u_min_pendulum = -4
u_max_pendulum = 2
x_min_pendulum = [-1;-3]
x_max_pendulum = [20; 20]

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)
obj_con_pendulum.tf = 3.0
model = model_pendulum
obj = obj_con_pendulum
obj.Qf[:,:] = 0.0*Matrix(I,2,2)
# Solver
intergration = :rk4
N = 25
solver1 = Solver(model,obj,integration=intergration,N=N)

# control and state constraints and indices
p,pI,pE = get_num_constraints(solver1)
pu = 2
ps = 4
pu_idx = Array(1:2)
ps_idx = Array(3:6)
pN = model.n

X0 = line_trajectory(solver1)
U0 = ones(solver1.model.m,solver1.N)

solver1.opts.constraint_tolerance = 1e-2
solver1.opts.active_set_flag = false
results2, stat2 = solve(solver1,U0)
max_violation(results2)
cost(solver1,results2)
plot(to_array(results2.X)')
plot(to_array(results2.U)')
plot(to_array(results2.λ)')

results2.λN

solver1.opts.active_set_flag = true
solver1.opts.active_constraint_tolerance = 0.0
update_constraints!(results2,solver1)
calculate_jacobians!(results2, solver1)
Δv = _backwardpass_active_set!(results2,solver1)

results2.Kλ
results2.dλ
results2.λN
# J = forwardpass!(results2, solver1, Δv)
rollout_active_set!(results2,solver1,0.01)
# results2.X .= deepcopy(results2.X_)
# results2.U .= deepcopy(results2.U_)
update_constraints!(results2,solver1,results2.X_,results2.U_)
cost(solver1,results2,results2.X_,results2.U_)

results2.X_[end]
_cost(solver1,results2,results2.X_,results2.U_)
cost_constraints(solver1, results2)
plot(to_array(results2.X_)')
plot(to_array(results2.U_)')
plot!(to_array(results2.λ)')
results2.λN
max_violation(results2)

a = 1
function _backwardpass_active_set!(res::SolverVectorResults,solver::Solver)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    pN = n
    W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf

    solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s
    L = res.L; l = res.l

    # Useful linear indices
    idx = Array(1:2*(n+mm))
    Idx = reshape(Array(1:(n+mm)^2),n+mm,n+mm)

    x_idx = idx[1:n]
    u_idx = idx[n+1:n+m]
    ū_idx = idx[n+1:n+mm]

    S = [k != N ? zeros(n+ps,n+ps) : zeros(n+pN,n+pN) for k = 1:N]
    s = [k != N ? zeros(n+ps) : zeros(n+pN) for k = 1:N]

    KK = res.Kλ
    dd = res.dλ
    KK[:] = [k != N-1 ? zeros(pu+ps,n) : zeros(pu+pN,n) for k = 1:N-1]
    dd[:] = [k != N-1 ? zeros(pu+ps) : zeros(pu+pN) for k = 1:N-1]

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
        Cx, Cu = res.Cx[k][ps_idx,:], res.Cu[k][pu_idx,:]
        Lx += Cx'*(Iμ[k][ps_idx,ps_idx]*C[k][ps_idx] + λ[k][ps_idx])
        Lu += Cu'*(Iμ[k][pu_idx,pu_idx]*C[k][pu_idx] + λ[k][pu_idx])
        Lxx += Cx'*Iμ[k][ps_idx,ps_idx]*Cx
        Luu += Cu'*Iμ[k][pu_idx,pu_idx]*Cu
        # Lux += Cu'*Iμ[k]*Cx

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = Lx + fdx'*s[k+1][1:n]
        Qu = Lu + fdu'*s[k+1][1:n]
        Qxx = Lxx + fdx'*S[k+1][1:n,1:n]*fdx
        Quu = Luu + fdu'*S[k+1][1:n,1:n]*fdu
        Qux = Lux + fdu'*S[k+1][1:n,1:n]*fdx

        if solver.opts.regularization_type == :state
            Quu_reg = Quu + res.ρ[1]*fdu'*fdu
            Qux_reg = Qux + res.ρ[1]*fdu'*fdx
        elseif solver.opts.regularization_type == :control
            Quu_reg = Quu + res.ρ[1]*I
            Qux_reg = Qux
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            regularization_update!(res,solver,:increase)

            Sxx = S[N][1:n,1:n]
            Sxλ = S[N][1:n,n+1:n+pN]
            Sλλ = S[N][n+1:n+pN,n+1:n+pN]

            Sx = s[N][1:n]
            Sλ = s[N][n+1:n+pN]
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
        active_set_idx_1u = res.active_set[k][pu_idx] # active set indices at index k
        active_set_idx_1s = res.active_set[k][ps_idx] # active set indices at index k

        k != N-1 ? active_set_idx_2 = res.active_set[k+1][ps_idx] : active_set_idx_2 = ones(Bool,pN) # active set indices at index k+1

        c_active_set_1u = res.C[k][pu_idx][active_set_idx_1u] # active set constraints at index k
        c_active_set_1s = res.C[k][ps_idx][active_set_idx_1s] # active set constraints at index k
        k != N-1 ? c_active_set_2 = S̄λ[active_set_idx_2] : c_active_set_2 = S̄λ # active set constraints at index k+1
        c_set_1u = res.C[k][pu_idx]
        c_set_1s = res.C[k][ps_idx]
        c_set_2 = S̄λ

        cx_active_set_1 = res.Cx[k][ps_idx,:][active_set_idx_1s,:] # active set Jacobians at index k
        cx_active_set_2 = (S̄xλ[:,active_set_idx_2])' # active set Jacobians at index k+1

        cx_set_1 = res.Cx[k][ps_idx,:]
        cx_set_2 = S̄xλ'

        cu_active_set_1 = res.Cu[k][pu_idx,:][active_set_idx_1u,:] # active set Jacobians at index k
        cu_active_set_2 = S̄λu[active_set_idx_2,:] # active set Jacobians at index k+1
        cu_set_1 = res.Cu[k][pu_idx,:]
        cu_set_2 = S̄λu

        # calculate control and multiplier gains
        tmp1 = [cu_active_set_1;cu_active_set_2]
        tmp2 = tmp1*(Quu_reg\tmp1')
        tmp3 = [zeros(length(c_active_set_1u),n);cx_active_set_2]
        tmp4 = [c_active_set_1u;c_active_set_2]
        tmp5 = [cu_set_1; cu_set_2]


        k != N-1 ? gains_active_idx = [res.active_set[k][pu_idx]; res.active_set[k+1][ps_idx]] : gains_active_idx = [res.active_set[k][pu_idx]; ones(Bool,pN)]




        if sum(gains_active_idx) != 0
            println("Iteration $k")
            println(tmp1)
            println(tmp2)
            println(tmp3)
            println(tmp4)
            # println(inv(tmp2))
            println("Cu")
            println(res.Cu[k])
            println("Cx")
            println(res.Cx[k])
            println("S̄λu:")
            println(S̄λu)
            println("Cx_N")
            println(res.Cx_N)
            println("Quu")
            println(Quu)
            println(Quu_reg)
            println(rank(tmp1))
            println(rank(tmp2))
            println(cond(tmp2))

            # println(res.Cx_N*fdu*inv(Quu_reg)*fdu'*res.Cx_N')
            iter = 0
            res.ρ[1] = 0.
            while rank(tmp2) < size(tmp2,1)
                tmp2 += res.ρ[1]*Matrix(I,size(tmp2,1),size(tmp2,1))
                regularization_update!(res,solver,:increase)

                iter += 1
                iter == 100 ? error("couldn't fix tmp2") : nothing
            end
            iter = 0

            KK[k][gains_active_idx,:] = -tmp2\(tmp1*(Quu_reg\Qux_reg) - tmp3)
            dd[k][gains_active_idx] = -tmp2\(tmp1*(Quu_reg\Qu) - tmp4)

            K[k] = -Quu_reg\(Qux_reg + tmp5'*KK[k])
            d[k] = -Quu_reg\(Qu + tmp5'*dd[k])

            println("KK: $(KK[k])")
            println("dd: $(dd[k])")
        end



        k != N-1 ? KKs = KK[k][ps_idx,:] : KKs = KK[k][Array(pu_idx[end]+1:pu_idx[end]+pN),:]
        KKu = KK[k][pu_idx,:]
        k != N-1 ? dds = dd[k][ps_idx] : dds = dd[k][Array(pu_idx[end]+1:pu_idx[end]+pN)]
        ddu = dd[k][pu_idx]

        Ŝxx = S̄xx + S̄xu*K[k] + S̄xλ*KKs + K[k]'*S̄xu' + K[k]'*S̄uu*K[k] + K[k]'*S̄λu'*KKs + KKs'*S̄xλ' + KKs'*S̄λu*K[k] + KKs'*S̄λλ*KKs
        Ŝx = S̄xu*d[k] + S̄xλ*dds + K[k]'*S̄uu*d[k] + KKs'*S̄λu*d[k] + K[k]'*S̄λu'*dds + KKs'*S̄λλ*dds + S̄x + K[k]'*S̄u + KKs'*S̄λ

        Δv[1] += S̄u'*d[k] + S̄λ'*dds
        Δv[2] += 0.5*d[k]'*S̄uu*d[k] + 0.5*d[k]'*S̄λu'*dds + 0.5*dds'*S̄λu*d[k] + 0.5*dds'*S̄λλ*dds

        Lλ = c_set_1s
        Lϕ = c_set_1u
        Luϕ = cu_set_1'
        Lxλ = cx_set_1'

        L̄xx = Lxx + Lux'*K[k] + K[k]'*Luu*K[k] + K[k]'*Luϕ*KKu + KKu'*Luϕ'*K[k] + K[k]'*Lux
        L̄xλ = Lxλ
        L̄x = Lux'*d[k] + K[k]'*Luu*d[k] + K[k]'*Luϕ*ddu + KKu'*Luϕ'*d[k] + Lx + K[k]'*Lu + KKu'*Lϕ
        L̄λ = Lλ

        Sxx = L̄xx + Ŝxx
        Sxλ = L̄xλ
        Sλλ = zeros(ps,ps)

        Sx = L̄x + Ŝx
        Sλ = L̄λ

        S[k][1:n,1:n] = Sxx
        S[k][1:n,n+1:n+ps] = Sxλ
        S[k][n+1:n+ps,1:n] = Sxλ'
        S[k][n+1:n+ps,n+1:n+ps] = Sλλ

        s[k][1:n] = Sx
        s[k][n+1:n+ps] = Sλ

        Δv[1] += Lu'*d[k] + Lϕ'*ddu
        Δv[2] += 0.5*d[k]'*Luu*d[k] + 0.5*d[k]'*Luϕ*ddu + 0.5*ddu'*Luϕ'*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

function rollout_active_set!(res::SolverVectorResults,solver::Solver,alpha::Float64)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[1] = solver.obj.x0;

    if solver.control_integration == :foh
        b = res.b
        du = alpha*d[1]
        U_[1] = U[1] + du
        dv = zero(du)
    end

    if solver.opts.active_set_flag
        λ = res.λ
        Kλ = res.Kλ
        Mλ = res.Mλ
        dλ = res.dλ

        δλ = alpha*dλ[1]
        λ[1][pu_idx] += δλ[pu_idx]
        λ[2][ps_idx] += δλ[ps_idx]
    end

    for k = 2:N
        δx = X_[k-1] - X[k-1]

        if solver.control_integration == :foh
            dv = K[k]*δx + b[k]*du + alpha*d[k]
            U_[k] = U[k] + dv
            solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
            solver.fd(X_[k], X_[k-1], U_[k-1][1:m], U_[k][1:m], dt)
            du = dv
        else
            U_[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]
            solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
            solver.fd(X_[k], X_[k-1], U_[k-1][1:m], dt)
        end

        # if solver.opts.active_set_flag
        #     δλ = Kλ[k-1]*δx + alpha*dλ[k-1]
        #     if k != N
        #         λ[k-1][pu_idx] += δλ[pu_idx]
        #         λ[k][ps_idx] += δλ[ps_idx]
        #     else
        #         λ[k-1][pu_idx] += δλ[pu_idx]
        #         res.λN[1:pN] += δλ[Array(pu_idx[end]+1:pu_idx[end]+pN)]
        #     end
        # end
        solver.opts.infeasible ? X_[k] += U_[k-1][m̄.+(1:n)] : nothing

        # Check that rollout has not diverged
        if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    # Calculate state derivatives and midpoints
    if solver.control_integration == :foh
        calculate_derivatives!(res, solver, X_, U_)
        calculate_midpoints!(res, solver, X_, U_)
    end

    # Update constraints
    update_constraints!(res,solver,X_,U_)

    return true
end


a = 1

# function _backwardpass_active_set!(res::SolverVectorResults,solver::Solver)
#     n,m,N = get_sizes(solver)
#     m̄,mm = get_num_controls(solver)
#
#     W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf
#
#     solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
#     solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing
#
#     dt = solver.dt
#
#     # pull out values from results
#     X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s
#     L = res.L; l = res.l
#
#     # Useful linear indices
#     idx = Array(1:2*(n+mm))
#     Idx = reshape(Array(1:(n+mm)^2),n+mm,n+mm)
#
#     x_idx = idx[1:n]
#     u_idx = idx[n+1:n+m]
#     ū_idx = idx[n+1:n+mm]
#
#     # Boundary Conditions
#     S[N] = Wf
#     s[N] = Wf*(X[N] - xf)
#
#     # Initialize expected change in cost-to-go
#     Δv = zeros(2)
#
#     # Terminal constraints
#     if res isa ConstrainedIterResults
#         C = res.C; Iμ = res.Iμ; λ = res.λ
#         CxN = res.Cx_N
#         S[N] += CxN'*res.IμN*CxN
#         s[N] += CxN'*(res.IμN*res.CN + res.λN)
#     end
#
#     # Backward pass
#     k = N-1
#     while k >= 1
#         solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing
#
#         x = X[k]
#         u = U[k][1:m]
#
#         ℓ1 = ℓ(x,u,W,R,xf)
#         ℓx = W*(x - xf)
#         ℓu = R*u
#         ℓxx = W
#         ℓuu = R
#         ℓux = zeros(m,n)
#
#         # Assemble expansion
#
#         Lx = dt*ℓx
#         Lu = dt*ℓu
#         Lxx = dt*ℓxx
#         Luu = dt*ℓuu
#         Lux = dt*ℓux
#
#         # Minimum time expansion components
#         if solver.opts.minimum_time
#             h = U[k][m̄]
#
#             l[n+m̄] = 2*h*(ℓ1 + R_minimum_time)
#
#             tmp = 2*h*ℓu
#             L[u_idx,m̄] = tmp
#             L[m̄,u_idx] = tmp'
#             L[m̄,m̄] = 2*(ℓ1 + R_minimum_time)
#
#             L[m̄,x_idx] = 2*h*ℓx'
#         end
#
#         # Infeasible expansion components
#         if solver.opts.infeasible
#             l[n+m̄+1:n+mm] = R_infeasible*U[k][m̄+1:m̄+n]
#             L[n+m̄+1:n+mm,n+m̄+1:n+mm] = R_infeasible
#         end
#
#         # Final expansion terms
#         if solver.opts.minimum_time || solver.opts.infeasible
#             l[u_idx] = Lu
#
#             L[u_idx,u_idx] = Luu
#             L[u_idx,x_idx] = Lux
#
#             Lu = view(l,n+1:n+mm)
#             Luu = view(L,n+1:n+mm,n+1:n+mm)
#             Lux = view(L,n+1:n+mm,1:n)
#         end
#
#         # Compute gradients of the dynamics
#         fdx, fdu = res.fdx[k], res.fdu[k]
#
#         # Gradients and Hessians of Taylor Series Expansion of Q
#         Qx = Lx + fdx'*s[k+1]
#         Qu = Lu + fdu'*s[k+1]
#         Qxx = Lxx + fdx'*S[k+1]*fdx
#         Quu = Luu + fdu'*S[k+1]*fdu
#         Qux = Lux + fdu'*S[k+1]*fdx
#
#         # Constraints
#         if res isa ConstrainedIterResults
#             Cx, Cu = res.Cx[k], res.Cu[k]
#             Qx .+= Cx'*(Iμ[k]*C[k] + λ[k])
#             Qu .+= Cu'*(Iμ[k]*C[k] + λ[k])
#             Qxx .+= Cx'*Iμ[k]*Cx
#             Quu .+= Cu'*Iμ[k]*Cu
#             Qux .+= Cu'*Iμ[k]*Cx
#         end
#
#         if solver.opts.regularization_type == :state
#             Quu_reg = Quu + res.ρ[1]*fdu'*fdu
#             Qux_reg = Qux + res.ρ[1]*fdu'*fdx
#         elseif solver.opts.regularization_type == :control
#             Quu_reg = Quu + res.ρ[1]*I
#             Qux_reg = Qux
#         end
#
#         # Regularization
#         if !isposdef(Hermitian(Array(Quu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
#             # increase regularization
#             regularization_update!(res,solver,:increase)
#
#             # reset backward pass
#             k = N-1
#             Δv[1] = 0.
#             Δv[2] = 0.
#             continue
#         end
#
#         if k == N-1
#             g = res.Cx_N*fdu
#             h = res.Cx_N*fdx
#
#             tmp = g*(Quu\g')
#             res.Kλ[N] = -tmp\(g*(Quu\Qux) - h)
#             res.Mλ[N] = -tmp\(g*(Quu\res.Cu[k]'))
#             res.dλ[N] = -tmp\(g*(Quu\Qu) - res.CN)
#
#             cx_active = res.Cx[k][res.active_set[k],:]
#             cu_active = res.Cu[k][res.active_set[k],:]
#             c_active = res.C[k][res.active_set[k]]
#
#             tmp = cu_active*(Quu\(cu_active' + (g'*res.Mλ[N])[:,res.active_set[k]]))
#             res.Kλ[k][res.active_set[k],:] = -tmp\(cu_active*(Quu\(Qux + g'*res.Kλ[N])) - cx_active)
#             res.dλ[k][res.active_set[k]] = -tmp\(cu_active*(Quu\(Qu + g'*res.dλ[N])) - c_active)
#
#             res.K[k] = -Quu\(Qux + g'*res.Kλ[N] + (res.Cu[k]' + g'*res.Mλ[N])*res.Kλ[k])
#             res.d[k] = -Quu\(Qu + g'*res.dλ[N] + (res.Cu[k]' + g'*res.Mλ[N])*res.dλ[k])
#         else
#             cx_active = res.Cx[k][res.active_set[k],:]
#             cu_active = res.Cu[k][res.active_set[k],:]
#             c_active = res.C[k][res.active_set[k]]
#
#             tmp = cu_active*(Quu\cu_active')
#             res.Kλ[k][res.active_set[k],:] = -tmp\(cu_active*(Quu\Qux) - cx_active)
#             res.dλ[k][res.active_set[k]] = -tmp\(cu_active*(Quu\Qu) - c_active)
#
#             res.K[k] = -Quu\(Qux + res.Cu[k]'*res.Kλ[k])
#             res.d[k] = -Quu\(Qu + res.Cu[k]'*res.dλ[k])
#         end
#
#         # Compute gains
#         # K[k] = -Quu_reg\Qux_reg
#         # d[k] = -Quu_reg\Qu
#
#         # Calculate cost-to-go (using unregularized Quu and Qux)
#         s[k] = Qx + K[k]'*Quu*d[k] + K[k]'*Qu + Qux'*d[k]
#         S[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'*K[k]
#         S[k] = 0.5*(S[k] + S[k]')
#
#         # calculated change is cost-to-go over entire trajectory
#         Δv[1] += d[k]'*Qu
#         Δv[2] += 0.5*d[k]'*Quu*d[k]
#
#         k = k - 1;
#     end
#
#     # decrease regularization after backward pass
#     regularization_update!(res,solver,:decrease)
#
#     return Δv
# end

a = 1
# function _backwardpass_active_set!(res::SolverVectorResults,solver::Solver)
#     n,m,N = get_sizes(solver)
#     m̄,mm = get_num_controls(solver)
#     p,pI,pE = get_num_constraints(solver)
#     pN = n
#
#     W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf
#
#     solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
#     solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing
#
#     dt = solver.dt
#
#     # pull out values from results
#     X = res.X; U = res.U; K = res.K; d = res.d;
#     L = res.L; l = res.l
#
#     # Useful linear indices
#     idx = Array(1:2*(n+mm))
#     Idx = reshape(Array(1:(n+mm)^2),n+mm,n+mm)
#
#     x_idx = idx[1:n]
#     u_idx = idx[n+1:n+m]
#     ū_idx = idx[n+1:n+mm]
#
#
#     S = [k != N ? zeros(n+p,n+p) : zeros(n+pN,n+pN) for k = 1:N]
#     s = [k != N ? zeros(n+p) : zeros(n+pN) for k = 1:N]
#
#     # Boundary Conditions
#     C = res.C; Iμ = res.Iμ; λ = res.λ; CxN = res.Cx_N
#
#     Sxx = Wf + CxN'*res.IμN*CxN
#     Sxλ = CxN'
#     Sλλ = zeros(pN,pN)
#
#     Sx = Wf*(X[N] - xf) + CxN'*(res.IμN*res.CN + res.λN)
#     Sλ = res.CN
#
#     S[N][1:n,1:n] = Sxx
#     S[N][1:n,n+1:n+pN] = Sxλ
#     S[N][n+1:n+pN,1:n] = Sxλ'
#     S[N][n+1:n+pN,n+1:n+pN] = Sλλ
#
#     s[N][1:n] = Sx
#     s[N][n+1:n+pN] = Sλ
#
#     # Initialize expected change in cost-to-go
#     Δv = zeros(2)
#
#     # Backward pass
#     k = N-1
#     while k >= 1
#         println("k: $k")
#         solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing
#
#         x = X[k]
#         u = U[k][1:m]
#
#         ℓ1 = ℓ(x,u,W,R,xf)
#         ℓx = W*(x - xf)
#         ℓu = R*u
#         ℓxx = W
#         ℓuu = R
#         ℓux = zeros(m,n)
#
#         # Assemble expansion
#
#         Lx = dt*ℓx
#         Lu = dt*ℓu
#         Lxx = dt*ℓxx
#         Luu = dt*ℓuu
#         Lux = dt*ℓux
#
#         # Minimum time expansion components
#         if solver.opts.minimum_time
#             h = U[k][m̄]
#
#             l[n+m̄] = 2*h*(ℓ1 + R_minimum_time)
#
#             tmp = 2*h*ℓu
#             L[u_idx,m̄] = tmp
#             L[m̄,u_idx] = tmp'
#             L[m̄,m̄] = 2*(ℓ1 + R_minimum_time)
#
#             L[m̄,x_idx] = 2*h*ℓx'
#         end
#
#         # Infeasible expansion components
#         if solver.opts.infeasible
#             l[n+m̄+1:n+mm] = R_infeasible*U[k][m̄+1:m̄+n]
#             L[n+m̄+1:n+mm,n+m̄+1:n+mm] = R_infeasible
#         end
#
#         # Final expansion terms
#         if solver.opts.minimum_time || solver.opts.infeasible
#             l[u_idx] = Lu
#
#             L[u_idx,u_idx] = Luu
#             L[u_idx,x_idx] = Lux
#
#             Lu = l[n+1:n+mm]
#             Luu = L[n+1:n+mm,n+1:n+mm]
#             Lux = L[n+1:n+mm,1:n]
#         end
#
#         # Constraints
#         Cx, Cu = res.Cx[k], res.Cu[k]
#         Lx += Cx'*(Iμ[k]*C[k] + λ[k])
#         Lu += Cu'*(Iμ[k]*C[k] + λ[k])
#         Lxx += Cx'*Iμ[k]*Cx
#         Luu += Cu'*Iμ[k]*Cu
#         Lux += Cu'*Iμ[k]*Cx
#
#         # multiplier expansion terms
#         Lλ = res.C[k]
#         Lxλ = res.Cx[k]'
#         Lλλ = zeros(p,p)
#         Lλu = res.Cu[k]
#
#         # Compute gradients of the dynamics
#         fdx, fdu = res.fdx[k], res.fdu[k]
#
#         # Gradients and Hessians of Taylor Series Expansion of Q
#         Qx = Lx + fdx'*s[k+1][1:n]
#         Qu = Lu + fdu'*s[k+1][1:n]
#         Qxx = Lxx + fdx'*S[k+1][1:n,1:n]*fdx
#         Quu = Luu + fdu'*S[k+1][1:n,1:n]*fdu
#         Qux = Lux + fdu'*S[k+1][1:n,1:n]*fdx
#
#         # regularization
#         Quu_reg = Quu + res.ρ[1]*I
#         Qux_reg = Qux
#         Luu_reg = Luu + res.ρ[1]*I
#         Lux_reg = Lux
#
#         # Regularization
#         if !isposdef(Hermitian(Array(Quu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
#             # increase regularization
#             regularization_update!(res,solver,:increase)
#
#             Sxx = S[N][1:n,1:n]
#             Sxλ = S[N][1:n,n+1:n+pN]
#             Sλλ = S[N][n+1:n+pN,n+1:n+pN]
#
#             Sx = s[N][1:n]
#             Sλ = s[N][n+1:n+pN]
#
#             # reset backward pass
#             k = N-1
#             Δv[1] = 0.
#             Δv[2] = 0.
#             continue
#         end
#
#         # linearized dynamics for cost-to-go
#         S̄xx = fdx'*Sxx*fdx
#         S̄xλ = fdx'*Sxλ
#         S̄xu = fdx'*Sxx*fdu
#         S̄λλ = Sλλ
#         S̄λu = Sxλ'*fdu
#         S̄uu = fdu'*Sxx*fdu
#
#         S̄x = fdx'*Sx
#         S̄λ = Sλ
#         S̄u = fdu'*Sx
#
#         # get active set constraints and Jacobians
#         active_set_idx_1 = res.active_set[k] # active set indices at index k
#         k != N-1 ? active_set_idx_2 = res.active_set[k+1] : active_set_idx_2 = ones(Bool,pN) # active set indices at index k+1
#
#         c_active_set_1 = res.C[k][active_set_idx_1] # active set constraints at index k
#         c_active_set_2 = S̄λ[active_set_idx_2] # active set constraints at index k+1
#         c_set_1 = res.C[k]
#         c_set_2 = S̄λ
#
#         println("S̄λ: $S̄λ")
#         k != N-1 ? println("C: $(res.C[k+1])") : println("C: $(res.CN)")
#         println("active set: $(active_set_idx_2)")
#
#
#         cx_active_set_1 = res.Cx[k][active_set_idx_1,:] # active set Jacobians at index k
#         cx_active_set_2 = (S̄xλ[:,active_set_idx_2])' # active set Jacobians at index k+1
#
#         cx_set_1 = res.Cx[k]
#         cx_set_2 = S̄xλ'
#
#         cu_active_set_1 = res.Cu[k][active_set_idx_1,:] # active set Jacobians at index k
#         cu_active_set_2 = S̄λu[active_set_idx_2,:] # active set Jacobians at index k+1
#         cu_set_1 = res.Cu[k]
#         cu_set_2 = S̄λu
#
#         if sum(active_set_idx_2) != 0
#             # calculate multiplier gains
#             tmp = cu_active_set_2*(Quu_reg\cu_active_set_2')
#
#             res.Kλ[k+1][active_set_idx_2,:] = -tmp\(cu_active_set_2*(Quu_reg\Qux_reg) - cx_active_set_2)
#             res.Mλ[k+1][active_set_idx_2,active_set_idx_1] = -tmp\(cu_active_set_2*(Quu_reg\cu_active_set_1'))
#             res.dλ[k+1][active_set_idx_2] = -tmp\(cu_active_set_2*(Quu_reg\Qu) - c_active_set_2)
#
#             # reset inactive gains to zero (this may be unnecessary)
#             res.Kλ[k+1][.!active_set_idx_2,:] .= 0.
#             res.Mλ[k+1][.!active_set_idx_2,.!active_set_idx_1] .= 0.
#             res.dλ[k+1][.!active_set_idx_2] .= 0.
#         end
#
#         if k != 1
#             # calculate control gains
#             res.K[k] = -Quu_reg\(Qux_reg + cu_set_2'*res.Kλ[k+1])
#             res.M[k] = -Quu_reg\(cu_set_1' + cu_set_2'*res.Mλ[k+1])
#             res.d[k] = -Quu_reg\(Qu + cu_set_2'*res.dλ[k+1])
#         elseif k == 1
#             if sum(res.active_set[1]) != 0
#                 # calculate final multiplier gains
#                 tmp = cu_active_set_1*(Quu_reg\(cu_active_set_1' + cu_active_set_2'*res.Mλ[k+1][active_set_idx_2,active_set_idx_1]))
#                 res.Kλ[1][active_set_idx_1,:] = -tmp\(cu_active_set_1*(Quu_reg\(Qux_reg + cu_active_set_2'*res.Kλ[k+1][active_set_idx_2,:])) - cx_active_set_1)
#                 res.dλ[1][active_set_idx_1] = -tmp\(cu_active_set_1*(Quu_reg\(Qu + cu_active_set_2'*res.dλ[k+1][active_set_idx_2])) - c_active_set_1)
#
#                 # reset inactive gains to zero (this may be unnecessary)
#                 res.Kλ[1][.!active_set_idx_1,:] .= 0.
#                 res.dλ[1][.!active_set_idx_1] .= 0.
#             end
#
#             # calculate final control gains
#             res.K[1] = -Quu_reg\(Qux_reg + cu_set_1'*res.Kλ[1] + cu_set_2'*res.Kλ[k+1] + cu_set_2'*res.Mλ[k+1]*res.Kλ[1])
#             res.d[1] = -Quu_reg\(Qu + cu_set_1'*res.dλ[1] + cu_set_2'*res.dλ[k+1] + cu_set_2'*res.Mλ[k+1]*res.dλ[1])
#         end
#
#         # plug in multiplier equation
#         Ŝxx = S̄xx + S̄xλ*res.Kλ[k+1] + res.Kλ[k+1]'*S̄xλ' + res.Kλ[k+1]'*S̄λλ*res.Kλ[k+1]
#         Ŝxλ = S̄xλ*res.Mλ[k+1] + res.Kλ[k+1]'*S̄λλ*res.Mλ[k+1]
#         Ŝxu = res.Kλ[k+1]'*S̄λu + S̄xu
#         Ŝλλ = res.Mλ[k+1]'*S̄λλ*res.Mλ[k+1]
#         Ŝλu = res.Mλ[k+1]'*S̄λu
#         Ŝuu = S̄uu
#
#         Ŝx = S̄xλ*res.dλ[k+1] + res.Kλ[k+1]'*S̄λλ*res.dλ[k+1] + S̄x + res.Kλ[k+1]'*S̄λ
#         Ŝλ = res.Mλ[k+1]'*S̄λλ*res.dλ[k+1] + res.Mλ[k+1]'*S̄λ
#         Ŝu = S̄λu'*res.dλ[k+1] + S̄u
#
#         Δv[1] += res.dλ[k+1]'*S̄λ
#         Δv[2] += 0.5*res.dλ[k+1]'*S̄λλ*res.dλ[k+1]
#         println("Δv: $Δv")
#
#         # calculate cost-to-go
#         Q̂xx = Lxx + Ŝxx
#         Q̂xλ = Lxλ + Ŝxλ
#         Q̂ux = Lux_reg + Ŝxu'
#         Q̂uu = Luu_reg + Ŝuu
#         Q̂λu = Lλu + Ŝλu
#         Q̂λλ = Lλλ + Ŝλλ
#         Q̂x = Lx + Ŝx
#         Q̂u = Lu + Ŝu
#         Q̂λ = Lλ + Ŝλ
#
#         Sxx = Q̂xx + Q̂ux'*res.K[k] + res.K[k]'*Q̂ux + res.K[k]'*Q̂uu*res.K[k]
#         Sxλ = Q̂xλ + Q̂ux'*res.M[k] + res.K[k]'*Q̂λu' + res.K[k]'*Q̂uu*res.M[k]
#         Sλλ = Q̂λλ + res.M[k]'*Q̂λu' + res.M[k]'*Q̂uu*res.M[k]
#
#         Sx = Q̂x + Q̂ux'*res.d[k] + res.K[k]'*Q̂uu*res.d[k] + res.K[k]'*Q̂u
#         Sλ = Q̂λ + Q̂λu*res.d[k] + res.M[k]'*Q̂uu*res.d[k] + res.M[k]'*Q̂u
#
#         S[k][1:n,1:n] = Sxx
#         S[k][1:n,n+1:n+p] = Sxλ
#         S[k][n+1:n+p,1:n] = Sxλ'
#         S[k][n+1:n+p,n+1:n+p] = Sλλ
#
#         s[k][1:n] = Sx
#         s[k][n+1:n+p] = Sλ
#
#         Δv[1] += res.d[k]'*Q̂u
#         Δv[2] += 0.5*res.d[k]'*Q̂uu*res.d[k]
#
#         if k == 1
#             Δv[1] += res.dλ[1]'*Sλ
#             Δv[2] += 0.5*res.dλ[1]'*Sλλ*res.dλ[1]
#         end
#         println("Δv: $Δv")
#         k = k - 1;
#     end
#
#     # decrease regularization after backward pass
#     regularization_update!(res,solver,:decrease)
#
#     return Δv
# end
#
# function rollout!(res::SolverVectorResults,solver::Solver,alpha::Float64)
#     n,m,N = get_sizes(solver)
#     m̄,mm = get_num_controls(solver)
#     dt = solver.dt
#
#     X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_
#
#     X_[1] = solver.obj.x0;
#
#     if solver.control_integration == :foh
#         b = res.b
#         du = alpha*d[1]
#         U_[1] = U[1] + du
#         dv = zero(du)
#     end
#
#     if solver.opts.active_set_flag
#         λ = res.λ
#         Kλ = res.Kλ
#         Mλ = res.Mλ
#         dλ = res.dλ
#
#         M = res.M
#
#         δλ = alpha*dλ[1]
#         λ[1] += δλ
#     end
#
#     for k = 2:N
#         δx = X_[k-1] - X[k-1]
#
#         if solver.control_integration == :foh
#             dv = K[k]*δx + b[k]*du + alpha*d[k]
#             U_[k] = U[k] + dv
#             solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
#             solver.fd(X_[k], X_[k-1], U_[k-1][1:m], U_[k][1:m], dt)
#             du = dv
#         elseif solver.opts.active_set_flag
#             U_[k-1] = U[k-1] + K[k-1]*δx + M[k-1]*δλ + alpha*d[k-1]
#             δλ = Kλ[k]*δx + Mλ[k]*δλ + alpha*dλ[k]
#             k != N ? λ[k] += δλ : res.λN += δλ
#             solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
#             solver.fd(X_[k], X_[k-1], U_[k-1][1:m], dt)
#         else
#             U_[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]
#             solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
#             solver.fd(X_[k], X_[k-1], U_[k-1][1:m], dt)
#         end
#
#         solver.opts.infeasible ? X_[k] += U_[k-1][m̄.+(1:n)] : nothing
#
#         # Check that rollout has not diverged
#         if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
#             return false
#         end
#     end
#
#     # Calculate state derivatives and midpoints
#     if solver.control_integration == :foh
#         calculate_derivatives!(res, solver, X_, U_)
#         calculate_midpoints!(res, solver, X_, U_)
#     end
#
#     # Update constraints
#     update_constraints!(res,solver,X_,U_)
#
#     return true
# end

# function rollout_active_set!(res::SolverVectorResults,solver::Solver,alpha::Float64)
#     n,m,N = get_sizes(solver)
#     m̄,mm = get_num_controls(solver)
#     dt = solver.dt
#
#     X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_
#
#     X_[1] = solver.obj.x0;
#
#     if solver.control_integration == :foh
#         b = res.b
#         du = alpha*d[1]
#         U_[1] = U[1] + du
#         dv = zero(du)
#     end
#
#     if solver.opts.active_set_flag
#         λ = res.λ
#         Kλ = res.Kλ
#         Mλ = res.Mλ
#         dλ = res.dλ
#
#         δλ = alpha*dλ[1]
#         λ[1] += δλ
#     end
#
#     for k = 2:N
#         δx = X_[k-1] - X[k-1]
#
#         if solver.control_integration == :foh
#             dv = K[k]*δx + b[k]*du + alpha*d[k]
#             U_[k] = U[k] + dv
#             solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
#             solver.fd(X_[k], X_[k-1], U_[k-1][1:m], U_[k][1:m], dt)
#             du = dv
#         else
#             U_[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]
#             solver.opts.minimum_time ? dt = U_[k-1][m̄]^2 : nothing
#             solver.fd(X_[k], X_[k-1], U_[k-1][1:m], dt)
#         end
#
#         if solver.opts.active_set_flag
#             k != N ? δλ = Kλ[k]*δx + alpha*dλ[k] : nothing
#             k != N ? λ[k] += δλ : res.λN .+= Kλ[k]*δx + Mλ[k]*δλ + alpha*dλ[k]
#         end
#         solver.opts.infeasible ? X_[k] += U_[k-1][m̄.+(1:n)] : nothing
#
#         # Check that rollout has not diverged
#         if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
#             return false
#         end
#     end
#
#     # Calculate state derivatives and midpoints
#     if solver.control_integration == :foh
#         calculate_derivatives!(res, solver, X_, U_)
#         calculate_midpoints!(res, solver, X_, U_)
#     end
#
#     # Update constraints
#     update_constraints!(res,solver,X_,U_)
#
#     return true
# end
