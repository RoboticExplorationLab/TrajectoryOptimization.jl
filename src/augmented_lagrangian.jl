# """
# $(SIGNATURES)
# BFGS multiplier update
# """
# function BFGS(Hinv,ρ,y,s)
#     p = size(y)
#     ρ = inv(y'*s)
#     Hinv = (1.0*Matrix(I,p,p) - ρ*y*s')*Hinv*(1.0*Matrix(I,p,p) - ρ*s*y') + ρ*s*s'
#     return Hinv
# end

"""
$(SIGNATURES)
    Lagrange multiplier updates
        -see Bertsekas 'Constrained Optimization' chapter 2 (p.135)
        -see Toussaint 'A Novel Augmented Lagrangian Approach for Inequalities and Convergent Any-Time Non-Central Updates'
"""
function λ_update_default!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    for k = 1:N
        k != N ? idx_pI = pI : idx_pI = pI_N

        results.λ[k] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, results.λ[k] + results.μ[k].*results.C[k]))
        results.λ[k][1:idx_pI] = max.(0.0,results.λ[k][1:idx_pI])
    end
end

function λ_update_default!(results::ConstrainedIterResults,solver::Solver,i::Int,k::Int)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    k != N ? idx_pI = pI : idx_pI = pI_N

    results.λ[k][i] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, results.λ[k][i] + results.μ[k][i].*results.C[k][i]))
    i <= idx_pI ? results.λ[k][i] = max.(0.0,results.λ[k][i]) : nothing
end

# function λ_update_accel!(results::ConstrainedIterResults,solver::Solver)
#     n,m,N = get_sizes(solver)
#     p,pI,pE = get_num_constraints(solver)
#     p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
#
#     λ,μ,C = results.λ, results.μ, results.C
#     t,λ_tilde_prev = results.t_prev, results.λ_prev
#     for k = 1:N-1
#         t_prime = @. (1+sqrt(1+4t[k]^2))/2
#         λ_tilde = λ[k] + μ[k].*C[k]
#         λ_prime = @. λ_tilde + ((t[k]-1)/t_prime)*(λ_tilde - λ_tilde_prev[k])
#
#         results.λ[k] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ_prime))
#         results.λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])
#
#         λ_tilde_prev[k] = λ_tilde
#         t[k] = t_prime
#     end
#     t_prime = @. (1+sqrt(1+4t[N]^2))
#     λ_tilde = λ[N] + μ[N].*C[N]
#     λ_prime = @. λ_tilde + ((t[N]-1)/t_prime)*(λ_tilde - λ_tilde_prev[N])
#
#     results.λ[N] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ_prime))
#     results.λ[N][1:pI_N] = max.(0.0,results.λ[N][1:pI_N])
#
#     λ_tilde_prev[N] = λ_tilde
#     t[N] = t_prime
# end

function λ_update_momentum!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    λ,μ,C = results.λ, results.μ, results.C
    λ_prev = results.λ_prev

    for k = 1:N-1
        λ_next = λ[k] + 1.9*μ[k].*C[k] + 0.99*λ_prev[k]
        λ_next = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ_next))
        λ_next[1:pI] = max.(0.0,λ_next[1:pI])
        λ_prev[k] = λ[k]
        λ[k] = λ_next
    end
    λ_next = λ[N] + 1.9μ[N].*C[N] + 0.99*λ_prev[N]
    λ_next = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ_next))
    λ_next[1:pI_N] = max.(0.0,λ_next[1:pI_N])
    λ_prev[N] = λ[N]
    λ[N] = λ_next
end

function λ_update_nesterov!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    λ,μ,C = results.λ, results.μ, results.C
    y = results.λ_prev
    α_prev = results.nesterov[1]
    α = results.nesterov[2]
    α_next = (1+sqrt(1+4α^2))/2
    γ_k = -(1-α)/α_next

    for k = 1:N-1
        y_next = λ[k] + μ[k].*C[k]
        y_next = max.(solver.opts.dual_min, min.(solver.opts.dual_max, y_next))
        y_next[1:pI] = max.(0.0,y_next[1:pI])
        λ[k] = (1-γ_k)*y_next + γ_k*y[k]
        y[k] = y_next

        λ[k] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ[k]))
        λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])
    end
    y_next = λ[N] + μ[N].*C[N]
    y_next = max.(solver.opts.dual_min, min.(solver.opts.dual_max, y_next))
    y_next[1:pI_N] = max.(0.0,y_next[1:pI_N])
    λ[N] = (1-γ_k)*y_next + γ_k*y[N]
    y[N] = y_next

    λ[N] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ[N]))
    λ[N][1:pI_N] = max.(0.0,results.λ[N][1:pI_N])

    results.nesterov[1] = α
    results.nesterov[2] = α_next
end


# """
# $(SIGNATURES)
#     Second order dual update - Buys Update
#     -UNDER DEVELOPMENT -
# """
# function Buys_λ_second_order_update!(results::SolverIterResults,solver::Solver,update::Bool=true)
#     bp = results.bp
#     n,m,N = get_sizes(solver)
#     n̄,nn = get_num_states(solver)
#     m̄,mm = get_num_controls(solver)
#     p,pI,pE = get_num_constraints(solver)
#     p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
#
#     nm = nn + mm
#     Nz = nn*N + mm*(N-1)
#     Np = p*(N-1) + p_N
#     ∇²L = zeros(Nz,Nz)
#     ∇c = zeros(Np,Nz)
#
#     for k = 1:N
#         if k < N
#             idx = ((k-1)*nm + 1):k*nm
#             solver.opts.square_root ? Q = [bp.Qxx[k]'*bp.Qxx[k] bp.Qux[k]'; bp.Qux[k] bp.Quu[k]'*bp.Quu[k]] : Q = [bp.Qxx[k] bp.Qux[k]'; bp.Qux[k] bp.Quu[k]]
#             ∇²L[idx,idx] = Q
#
#             idx2 = ((k-1)*p + 1):k*p
#             ∇c[idx2,idx] = [results.Cx[k] results.Cu[k]]
#         else
#             idx = ((k-1)*nm + 1):Nz
#             solver.opts.square_root ? Q = results.S[N]'*results.S[N] : Q = results.S[N]
#             ∇²L[idx,idx] = Q
#
#             idx2 = ((k-1)*p + 1):Np
#             ∇c[idx2,idx] = results.Cx[N]
#         end
#     end
#
#     C = vcat(results.C...)
#     λ = vcat(results.λ...)
#     active_set = vcat(results.active_set...)
#
#     ∇c̄ = ∇c[active_set,:]
#
#     tmp = (∇c̄*(∇²L\∇c̄'))
#     λ[active_set] += tmp\C[active_set]
#
#     if update
#         # update the results
#         for k = 1:N
#             if k != N
#                 idx_pI = pI
#                 idx = (k-1)*p+1:k*p
#             else
#                 idx_pI = pI_N
#                 idx = (k-1)*p+1:Np
#             end
#             results.λ[k] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ[idx]))
#             results.λ[k][1:idx_pI] = max.(0.0,results.λ[k][1:idx_pI])
#         end
#         return nothing
#     else
#         return λ
#     end
# end
using Test

function Buys_λ_second_order_update!(results::SolverIterResults,solver::Solver,update::Bool=true)
    n,m,N = get_sizes(solver)
    n̄,nn = get_num_states(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    update_constraints!(results,solver)

    X = results.X
    U = results.U
    Iμ = results.Iμ
    Cx = results.Cx
    Cu = results.Cu

    x0 = solver.obj.x0
    nm = nn + mm
    Nz = nn*N + mm*(N-1)
    Np = p*(N-1) + p_N
    Nu = mm*(N-1) # number of control decision variables u
    Ā = zeros(Nz,nn)
    B̄ = zeros(Nz,Nu)
    Q̄ = zeros(Nz,Nz)
    C̄ = zeros(Np,Nz)

    for k = 1:N
        # Calculate Ā
        k == 1 ? Ā[((k-1)*nm + 1):((k-1)*nm + nn),1:nn] = 1.0*Matrix(I,nn,nn) : Ā[((k-1)*nm + 1):((k-1)*nm + nn),1:nn] = prod(results.fdx[1:k-1])

        if k < N
            x = X[k][1:n]
            u = U[k][1:m]
            solver.state.minimum_time ? dt = U[k][m̄]^2 : dt = solver.dt
            expansion = taylor_expansion(solver.obj.cost,x,u)
            Q,R,H,q,r = expansion .* dt

            idx = ((k-1)*nm + 1):k*nm
            Q̄[idx,idx] = [Q H'; H R] + [Cx[k]'*Iμ[k]*Cx[k] Cx[k]'*Iμ[k]*Cu[k]; Cu[k]'*Iμ[k]*Cx[k] Cu[k]'*Iμ[k]*Cu[k]]

            idx2 = ((k-1)*p + 1):k*p
            C̄[idx2,idx] = [results.Cx[k] results.Cu[k]]
        else
            x = X[N][1:n]
            expansion = taylor_expansion(solver.obj.cost,x)
            Qf,qf = expansion

            idx = ((k-1)*nm + 1):Nz
            Q̄[idx,idx] = Qf + Cx[N]'*Iμ[N]*Cx[N]

            idx2 = ((k-1)*p + 1):Np
            C̄[idx2,idx] = results.Cx[N]
        end
    end

    for k = 1:N
        # Indices
        idx3 = ((k-1)*nm + 1):((k-1)*nm + n)
        idx4 = ((k-1)*nm + n + 1):k*nm
        idx5 = ((k-1)*m + 1):k*m

        # Calculate B̄
        if k > 1
            for j = 1:k-1
                idx7 = ((j-1)*m + 1):j*m
                j == k-1 ? B̄[idx3,idx7] = results.fdu[j][1:n,1:m] : B̄[idx3,idx7] = prod(results.fdx[j+1:(k-1)])*results.fdu[j][1:n,1:m]
            end
        end

        if k != N
            B̄[idx4,idx5] = 1.0*Matrix(I,m,m)
        end
    end

    # ū = vcat(results.U...)
    # z = B̄*ū + Ā*x0
    #
    # x̄ = [zeros(n) for i = 1:solver.N]
    # x̄[1] = x0
    # for k = 1:N-1
    #     x̄[k+1] = results.fdx[k]*x̄[k] + results.fdu[k]*results.U[k]
    # end
    # x̂ = [zeros(n) for i = 1:solver.N]
    # for k = 1:N
    #     k != N ? idx = ((k-1)*nm+1:(k-1)*nm+n) : idx = ((k-1)*nm+1:Nz)
    #     x̂[k] = z[idx]
    # end
    #
    # @test isapprox(to_array(x̄),to_array(x̂))
    # @test B̄[1:n,1:n] == zeros(n,n)
    # @test B̄[n+1:nm,1:m] == 1.0*Matrix(I,m,m)
    # @test B̄[nm+1:nm+n,1:m] == results.fdu[1][1:n,1:m]
    # @test B̄[nm+n+1:2*nm,m+1:2*m] == 1.0*Matrix(I,m,m)
    # @test B̄[(N-1)*nm+1:Nz,1:m] == prod(results.fdx[2:N-1])*results.fdu[1]
    # @test B̄[(N-1)*nm+1:Nz,m+1:2*m] == prod(results.fdx[3:N-1])*results.fdu[2]
    # @test B̄[(N-1)*nm+1:Nz,(N-2)*m+1:(N-1)*m] == results.fdu[N-1]

    ∇²L = B̄'*Q̄*B̄
    println("cond(∇²L): $(cond(∇²L))")
    ∇g = C̄*B̄

    C = vcat(results.C...)
    λ = vcat(results.λ...)
    active_set = vcat(results.active_set...)

    ∇ḡ = ∇g[active_set,:]
    #
    tmp = (∇ḡ*(∇²L\∇ḡ'))
    println("cond tmp: $(cond(tmp))")
    λ[active_set] += tmp\C[active_set]
    # λ += tmp\C

    if update
        # update the results
        for k = 1:N
            if k != N
                idx_pI = pI
                idx = (k-1)*p+1:k*p
            else
                idx_pI = pI_N
                idx = (k-1)*p+1:Np
            end
            results.λ[k] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ[idx]))
            results.λ[k][1:idx_pI] = max.(0.0,results.λ[k][1:idx_pI])
        end
        return nothing
    else
        return λ
    end
end



# """
# $(SIGNATURES)
#     Second order dual update - Batch Dual QP solve
#     -UNDER DEVELOPMENT - currently not improving convergence rate
# """
function qp_λ_second_order_update!(results::SolverIterResults,solver::Solver)
    nothing
end
#     n = solver.model.n
#     m = solver.model.m
#     nm = n+m
#     N = solver.N
#     p,pI,pE = get_num_constraints(solver)
#     p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
#
#     Nz = n*N + m*(N-1) # number of decision variables x, u
#     Nu = m*(N-1) # number of control decision variables u
#
#     Np = p*(N-1) + p_N # number of constraints
#
#     Mz = N + N-1 # number of decision vectors
#     Mu = N-1 # number of control decision vectors
#
#     B̄ = zeros(Nz,Nu)
#     Ā = zeros(Nz,n)
#
#     Q̄ = zeros(Nz,Nz)
#     q̄ = zeros(Nz)
#
#     C̄ = zeros(Np,Nz)
#     c̄ = zeros(Np)
#
#     λ_tmp = zeros(Np)
#     idx_inequality = zeros(Bool,Np)
#     active_set = zeros(Bool,Np)
#
#     x0 = solver.obj.x0
#     costfun = solver.obj.cost
#
#     for k = 1:N
#         x = results.X[k]
#
#         if k != N
#             u = results.U[k]
#             Q,R,H,q,r = taylor_expansion(costfun,x,u) .* solver.dt
#         else
#             Qf,qf = taylor_expansion(costfun,x) .* solver.dt
#         end
#
#         # Indices
#         idx = ((k-1)*nm + 1):k*nm
#         idx2 = ((k-1)*p + 1):k*p
#         idx3 = ((k-1)*nm + 1):((k-1)*nm + n)
#         idx4 = ((k-1)*nm + n + 1):k*nm
#         idx5 = ((k-1)*m + 1):k*m
#         idx6 = ((k-1)*p + 1):((k-1)*p + pI)
#
#         # Calculate Ā
#         k == 1 ? Ā[idx3,1:n] = 1.0*Matrix(I,n,n) : Ā[idx3,1:n] = prod(results.fdx[1:k-1])
#
#         # Calculate B̄
#         if k > 1
#             for j = 1:k-1
#                 idx7 = ((j-1)*m + 1):j*m
#                 j == k-1 ? B̄[idx3,idx7] = results.fdu[j][1:n,1:m] : B̄[idx3,idx7] = prod(results.fdx[j+1:(k-1)])*results.fdu[j][1:n,1:m]
#             end
#         end
#
#         # Calculate Q̄, q̄, C̄, c̄
#         if k != N
#             Q̄[idx,idx] = [Q H';H R]
#             q̄[idx] = [q;r]
#
#             G = results.Cx[k]
#             H = results.Cu[k]
#             C̄[idx2,idx] = [G H]
#             c̄[idx2] = results.C[k]
#
#             λ_tmp[idx2] = results.λ[k]
#             idx_inequality[idx6] .= true
#             active_set[idx2] = results.active_set[k]
#
#             B̄[idx4,idx5] = 1.0*Matrix(I,m,m)
#         else
#             idx = ((k-1)*nm + 1):Nz
#             Q̄[idx,idx] = Qf
#             q̄[idx] = qf
#
#             idx2 = ((k-1)*p + 1):Np
#             C̄[idx2,idx] = results.Cx[N]
#             c̄[idx2] = results.C[N]
#
#             idx6 = ((k-1)*p + 1):((k-1)*p + pI_N)
#
#             λ_tmp[idx2] = results.λ[N]
#             idx_inequality[idx6] .= true
#             active_set[idx2] .= results.active_set[N]
#         end
#
#     end
#     N_active_set = sum(active_set)
#
#     # if verbose
#     #     # Tests
#     #     @test Ā[1:n,:] == Matrix(I,n,n)
#     #     k = 3
#     #     @test Ā[(k-1)*nm+1:(k-1)*nm+n,:] == prod(results.fdx[1:2])
#     #     @test results.fdx[1]*results.fdx[2] == prod(results.fdx[1:2])
#     #
#     #     k = 7
#     #     @test Ā[(k-1)*nm+1:(k-1)*nm+n,:] == prod(results.fdx[1:k-1])
#     #     @test Ā[(N-1)*nm+1:Nz,:] == prod(results.fdx[1:N-1])
#     #
#     #     @test B̄[1:n,1:n] == zeros(n,n)
#     #     @test B̄[n+1:nm,1:m] == 1.0*Matrix(I,m,m)
#     #     @test B̄[nm+1:nm+n,1:m] == results.fdu[1][1:n,1:m]
#     #     @test B̄[nm+n+1:2*nm,m+1:2*m] == 1.0*Matrix(I,m,m)
#     #     @test B̄[(N-1)*nm+1:Nz,1:m] == prod(results.fdx[2:N-1])*results.fdu[1]
#     #     @test B̄[(N-1)*nm+1:Nz,m+1:2*m] == prod(results.fdx[3:N-1])*results.fdu[2]
#     #     @test B̄[(N-1)*nm+1:Nz,(N-2)*m+1:(N-1)*m] == results.fdu[N-1]
#     #
#     #     k = 1
#     #     Q,R,H,q,r = taylor_expansion(costfun,results.X[k],results.U[k])
#     #     @test Q̄[1:nm,1:nm] == [Q H'; H R]
#     #     @test q̄[1:nm] == [q;r]
#     #
#     #     k = 13
#     #     Q,R,H,q,r = taylor_expansion(costfun,results.X[k],results.U[k])
#     #     @test Q̄[(k-1)*nm+1:k*nm,(k-1)*nm+1:k*nm] == [Q H'; H R]
#     #     @test q̄[(k-1)*nm+1:k*nm] == [q;r]
#     #
#     #     k = N
#     #     Qf,qf = taylor_expansion(costfun,results.X[k])
#     #     @test Q̄[(k-1)*nm+1:Nz,(k-1)*nm+1:Nz] == Qf
#     #     @test q̄[(k-1)*nm+1:Nz] == qf
#     #
#     #     @test C̄[1:p,1:nm] == [results.Cx[1] results.Cu[1]]
#     #
#     #     k = 9
#     #     @test C̄[(k-1)*p+1:k*p,(k-1)*nm+1:k*nm] == [results.Cx[k] results.Cu[k]]
#     #     @test C̄[(N-1)*p+1:Np,(N-1)*nm+1:Nz] == results.Cx[N]
#     #
#     #     @test c̄[1:p] == results.C[1]
#     #
#     #     k = 17
#     #     @test c̄[(k-1)*p+1:k*p] == results.C[k]
#     #     @test c̄[(N-1)*p+1:Np] == results.C[N]
#     #
#     #     @test all(idx_inequality[1:pI] .== true)
#     #     @test all(idx_inequality[pI+1:p] .== false)
#     #     k = 12
#     #
#     #     @test all(idx_inequality[(k-1)*p+1:(k-1)*p+pI] .== true)
#     #     idx_inequality[(k-1)*p+1:(k-1)*p+pI]
#     #     @test all(idx_inequality[(N-1)*p+1:(N-1)*pI_N] .== true)
#     # end
#     ū = -(B̄'*Q̄*B̄)\(B̄'*(q̄ + Q̄*Ā*x0) + B̄'*C̄'*λ_tmp)
#
#     # println(any(isnan.(q̄)))
#     # println(any(isnan.(Ā)))
#     # println(any(isnan.(λ_tmp)))
#     # println(any(isnan.(ū)))
#     # println(any(isnan.(Ā)))
#     # println(any(isnan.(B̄)))
#     # println(any(isnan.(C̄)))
#     # println(any(isnan.(c̄)))
#     P = B̄'*Q̄*B̄
#     M = q̄ + Q̄*Ā*x0
#     # println(rank(P))
#     # println(size(P))
#     # println(λ_tmp)
#     L = 0.5*ū'*P*ū + M'*B̄*ū + λ_tmp'*(C̄*B̄*ū + C̄*Ā*x0 + c̄)
#
#     D = 0.5*M'*B̄*inv(P')*B̄'*M + 0.5*M'*B̄*inv(P')*B̄'*C̄'*λ_tmp + 0.5*λ_tmp'*C̄*B̄*inv(P')*B̄'*M + 0.5*λ_tmp'*C̄*B̄*inv(P')*B̄'*C̄'*λ_tmp - M'*B̄*inv(P)*B̄'*M - M'*B̄*inv(P)*B̄'*C̄'*λ_tmp - λ_tmp'*C̄*B̄*inv(P)*B̄'*M - λ_tmp'*C̄*B̄*inv(P)*B̄'*C̄'*λ_tmp + λ_tmp'*C̄*Ā*x0 + λ_tmp'*c̄
#     Q_dual = C̄*B̄*inv(P')*B̄'*C̄' - 2*C̄*B̄*inv(P)*B̄'*C̄'
#     q_dual = M'*B̄*inv(P')*B̄'*C̄' - 2*M'*B̄*inv(P)*B̄'*C̄' + x0'*Ā'*C̄' + c̄'
#     qq_dual = 0.5*M'*B̄*inv(P')*B̄'*M - M'*B̄*inv(P)*B̄'*M
#     DD = 0.5*λ_tmp'*Q_dual*λ_tmp + q_dual*λ_tmp + qq_dual
#
#     # @test isapprox(D,L)
#     # @test isapprox(DD,L)
#
#     # solve QP
#     m = JuMP.Model(solver=IpoptSolver(print_level=0))
#     # m = JuMP.Model(solver=ClpSolver())
#
#     # @variable(m, u[1:Nu])
#     #
#     # @objective(m, Min, 0.5*u'*P*u + M'*B̄*u)
#     # @constraint(m, con, C̄*B̄*u + C̄*Ā*x0 + c̄ .== 0.)
#
#     # @variable(m, λ[1:Np])
#     N_active_set = sum(active_set)
#
#     @variable(m, λ[1:N_active_set])
#
#     # @objective(m, Min, λ'*λ)
#     # @constraint(m, con, Q_dual[active_set,active_set]*λ + q_dual[active_set] .== 0.)
#     #
#     @objective(m, Max, 0.5*λ'*Q_dual[active_set,active_set]*λ + q_dual[active_set]'*λ)
#     @constraint(m, con2, λ[idx_inequality[active_set]] .>= 0)
#     #
#     # # print(m)
#     #
#     status = JuMP.solve(m)
#     #
#     if status != :Optimal
#         error("QP failed")
#     end
#
#     # Solution
#     # println("Objective value: ", JuMP.getobjectivevalue(m))
#     # println("λ = ", JuMP.getvalue(λ))
#
#     λ_tmp[active_set] = JuMP.getvalue(λ)
#
#     for k = 1:N
#         if k != N
#             idx = ((k-1)*p + 1):k*p
#             results.λ[k] = λ_tmp[idx]#JuMP.getvalue(λ[idx])
#             # results.λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])
#         else
#             idx = ((N-1)*p + 1):Np
#             results.λ[k] = λ_tmp[idx]#JuMP.getvalue(λ[idx])
#             # results.λ[k][1:pI_N] = max.(0.0,results.λ[k][1:pI_N])
#         end
#
#         # tolerance check
#         results.λ[k][abs.(results.λ[k]) .< 1e-8] .= 0.
#     end
# end

""" $(SIGNATURES) Penalty update """
function μ_update!(results::ConstrainedIterResults,solver::Solver)
    if solver.opts.outer_loop_update_type ∈ [:default,:accelerated, :momentum]
        μ_update_default!(results,solver)
    elseif solver.opts.outer_loop_update_type == :individual
        μ_update_individual!(results,solver)
    end
    return nothing
end

""" $(SIGNATURES) Penalty update scheme ('default') - all penalty terms are updated"""
function μ_update_default!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    for k = 1:N
        results.μ[k] = min.(solver.opts.penalty_max, solver.opts.penalty_scaling*results.μ[k])
    end
    return nothing
end

""" @(SIGNATURES) Penalty update scheme ('individual')- all penalty terms are updated uniquely according to indiviual improvement compared to previous iteration"""
function μ_update_individual!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    constraint_decrease_ratio = solver.opts.constraint_decrease_ratio
    penalty_max = solver.opts.penalty_max
    penalty_scaling_no  = solver.opts.penalty_scaling_no
    penalty_scaling = solver.opts.penalty_scaling

    # Stage constraints
    for k = 1:N
        if k != N
            _p = p
            _pI = pI
        else
            _p = p_N
            _pI = pI_N
        end
        for i = 1:_p
            if i <= _pI
                if max(0.0,results.C[k][i]) <= constraint_decrease_ratio*max(0.0,results.C_prev[k][i])
                    results.μ[k][i] = min(penalty_max, penalty_scaling_no*results.μ[k][i])
                else
                    results.μ[k][i] = min(penalty_max, penalty_scaling*results.μ[k][i])
                end
            else
                if abs(results.C[k][i]) <= constraint_decrease_ratio*abs(results.C_prev[k][i])
                    results.μ[k][i] = min(penalty_max, penalty_scaling_no*results.μ[k][i])
                else
                    results.μ[k][i] = min(penalty_max, penalty_scaling*results.μ[k][i])
                end
            end
        end
    end
    return nothing
end

function feedback_outer_loop_update!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    constraint_decrease_ratio = solver.opts.constraint_decrease_ratio
    penalty_max = solver.opts.penalty_max
    penalty_scaling_no  = solver.opts.penalty_scaling_no
    penalty_scaling = solver.opts.penalty_scaling
    use_nesterov = solver.opts.use_nesterov

    if use_nesterov
        λ,μ,C = results.λ, results.μ, results.C
        y = results.λ_prev
        α_prev = results.nesterov[1]
        α = results.nesterov[2]
        α_next = (1+sqrt(1+4α^2))/2
        γ_k = -(1-α)/α_next
    end

    if solver.state.second_order_dual_update
        λ = Buys_λ_second_order_update!(results,solver,false)
    end

    # Stage constraints
    for k = 1:N
        if k != N
            _p = p
            _pI = pI
        else
            _p = p_N
            _pI = pI_N
        end
        for i = 1:_p
            if i <= _pI
                if max(0.0,results.C[k][i]) <= constraint_decrease_ratio*max(0.0,results.C_prev[k][i])
                    if use_nesterov
                        y_next = λ[k][i] + μ[k][i].*C[k][i]
                        y_next = max.(solver.opts.dual_min, min.(solver.opts.dual_max, y_next))
                        y_next = max.(0.0,y_next)

                        λ[k][i] = (1-γ_k)*y_next + γ_k*y[k][i]
                        y[k][i] = y_next
                    # else
                    #     results.λ[k][i] += results.μ[k][i]*results.C[k][i]
                    end
                    if solver.state.second_order_dual_update
                        # λ_second_order_update!(results,solver,i,k)


                        idx = (k-1)*p+i

                        results.λ[k][i] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ[idx]))
                        results.λ[k][i] = max.(0.0,results.λ[k][i])
                    else
                        λ_update_default!(results,solver,i,k)
                    end

                    # results.λ[k][i] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, results.λ[k][i]))
                    # results.λ[k][i] = max.(0.0,results.λ[k][i])
                    !solver.state.second_order_dual_update ? results.μ[k][i] = min(penalty_max, penalty_scaling_no*results.μ[k][i]) : nothing
                else
                    !solver.state.second_order_dual_update ? results.μ[k][i] = min(penalty_max, penalty_scaling*results.μ[k][i]) : nothing
                end
            else
                if abs(results.C[k][i]) <= constraint_decrease_ratio*abs(results.C_prev[k][i])
                    if use_nesterov
                        y_next = λ[k][i] + μ[k][i].*C[k][i]
                        y_next = max.(solver.opts.dual_min, min.(solver.opts.dual_max, y_next))

                        λ[k][i] = (1-γ_k)*y_next + γ_k*y[k][i]
                        y[k][i] = y_next
                    # else
                    #     results.λ[k][i] += results.μ[k][i]*results.C[k][i]
                    end

                    # results.λ[k][i] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, results.λ[k][i]))

                    if solver.state.second_order_dual_update
                        idx = (k-1)*p+i
                        results.λ[k][i] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ[idx]))
                    else
                        λ_update_default!(results,solver,i,k)
                    end

                    !solver.state.second_order_dual_update ? results.μ[k][i] = min(penalty_max, penalty_scaling_no*results.μ[k][i]) : nothing
                else
                    !solver.state.second_order_dual_update ? results.μ[k][i] = min(penalty_max, penalty_scaling*results.μ[k][i]) : nothing
                end
            end
        end
    end
    if use_nesterov
        results.nesterov[1] = α
        results.nesterov[2] = α_next
    end
    return nothing
end

λ_second_order_update! = qp_λ_second_order_update!#Buys_λ_second_order_update!

"""
$(SIGNATURES)
    Updates penalty (μ) and Lagrange multiplier (λ) parameters for Augmented Lagrangian method
"""
function outer_loop_update(results::ConstrainedIterResults,solver::Solver,k::Int=0)::Nothing

    if solver.opts.outer_loop_update_type == :default
        if solver.opts.use_nesterov
            λ_update_nesterov!(results,solver)
        else
            solver.state.second_order_dual_update ? λ_second_order_update!(results,solver) : λ_update_default!(results,solver)
        end
        if !solver.state.second_order_dual_update
            k % solver.opts.penalty_update_frequency == 0 ? μ_update_default!(results,solver) : nothing
        end
    elseif solver.opts.outer_loop_update_type == :individual
        λ_update_default!(results,solver)
        k % solver.opts.penalty_update_frequency == 0 ? μ_update_individual!(results,solver) : nothing

    # elseif solver.opts.outer_loop_update_type == :accelerated
    #     λ_update_accel!(results,solver)
    #     k % solver.opts.penalty_update_frequency == 0 ? μ_update_default!(results,solver) : nothing

    elseif solver.opts.outer_loop_update_type == :momentum
        λ_update_momentum!(results,solver)
        k % solver.opts.penalty_update_frequency == 0 ? μ_update_default!(results,solver) : nothing

    elseif solver.opts.outer_loop_update_type == :feedback
        if solver.state.penalty_only
            μ_update_default!(results,solver)
        else
            feedback_outer_loop_update!(results,solver)
        end
    end

    ## Store current constraints evaluations for next outer loop update
    results.C_prev .= deepcopy(results.C)

    # reset regularization
    results.ρ[1] = 0.
    return nothing
end

function outer_loop_update(results::UnconstrainedIterResults,solver::Solver, k::Int=0)::Nothing
    return nothing
end
