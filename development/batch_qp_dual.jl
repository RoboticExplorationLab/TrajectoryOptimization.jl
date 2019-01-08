using Test, Plots, JuMP, Ipopt, Clp, LinearAlgebra

u_bound = 1.5
model, obj = TrajectoryOptimization.Dynamics.pendulum!
obj = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound)

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = true
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
U = zeros(model.m,solver.N-1)
results_c, = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c)

plot(to_array(results_c.U)')

# costfun = solver.obj.cost
# x = results_c.X[1]
# u = results_c.U[1]
# expansion = taylor_expansion(costfun,x,u)
# Q,R,H,q,r = taylor_expansion(costfun,x,u)
# H

plot(to_array(results_c.λ)[:,1:solver.N-1]')

function λ_second_order_update!(results::SolverIterResults,solver::Solver,verbose::Bool=false)
    n = solver.model.n
    m = solver.model.m
    nm = n+m
    N = solver.N
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    Nz = n*N + m*(N-1) # number of decision variables x, u
    Nu = m*(N-1) # number of control decision variables u

    Np = p*(N-1) + p_N # number of constraints

    Mz = N + N-1 # number of decision vectors
    Mu = N-1 # number of control decision vectors

    B̄ = zeros(Nz,Nu)
    Ā = zeros(Nz,n)

    Q̄ = zeros(Nz,Nz)
    q̄ = zeros(Nz)

    C̄ = zeros(Np,Nz)
    c̄ = zeros(Np)

    λ_tmp = zeros(Np)
    idx_inequality = zeros(Bool,Np)
    active_set = zeros(Bool,Np)

    x0 = solver.obj.x0
    costfun = solver.obj.cost

    for k = 1:N
        x = results.X[k]

        if k != N
            u = results.U[k]
            Q,R,H,q,r = taylor_expansion(costfun,x,u)
        else
            Qf,qf = taylor_expansion(costfun,x)
        end

        # Indices
        idx = ((k-1)*nm + 1):k*nm
        idx2 = ((k-1)*p + 1):k*p
        idx3 = ((k-1)*nm + 1):((k-1)*nm + n)
        idx4 = ((k-1)*nm + n + 1):k*nm
        idx5 = ((k-1)*m + 1):k*m
        idx6 = ((k-1)*p + 1):((k-1)*p + pI)

        # Calculate Ā
        k == 1 ? Ā[idx3,1:n] = 1.0*Matrix(I,n,n) : Ā[idx3,1:n] = prod(results.fdx[1:k-1])

        # Calculate B̄
        if k > 1
            for j = 1:k-1
                idx7 = ((j-1)*m + 1):j*m
                j == k-1 ? B̄[idx3,idx7] = results.fdu[j][1:n,1:m] : B̄[idx3,idx7] = prod(results.fdx[j+1:(k-1)])*results.fdu[j][1:n,1:m]
            end
        end

        # Calculate Q̄, q̄, C̄, c̄
        if k != N
            Q̄[idx,idx] = [Q H';H R]
            q̄[idx] = [q;r]

            G = results.Cx[k]
            H = results.Cu[k]
            C̄[idx2,idx] = [G H]
            c̄[idx2] = results.C[k]

            λ_tmp[idx2] = results.λ[k]
            idx_inequality[idx6] .= true
            active_set[idx2] = results.active_set[k]

            B̄[idx4,idx5] = 1.0*Matrix(I,m,m)
        else
            idx = ((k-1)*nm + 1):Nz
            Q̄[idx,idx] = Qf
            q̄[idx] = qf

            idx2 = ((k-1)*p + 1):Np
            C̄[idx2,idx] = results.Cx[N]
            c̄[idx2] = results.C[N]

            idx6 = ((k-1)*p + 1):((k-1)*p + pI_N)

            λ_tmp[idx2] = results.λ[N]
            idx_inequality[idx6] .= true
            active_set[idx2] .= results.active_set[N]
        end

    end
    N_active_set = sum(active_set)

    # if verbose
    #     # Tests
    #     @test Ā[1:n,:] == Matrix(I,n,n)
    #     k = 3
    #     @test Ā[(k-1)*nm+1:(k-1)*nm+n,:] == prod(results.fdx[1:2])
    #     @test results.fdx[1]*results.fdx[2] == prod(results.fdx[1:2])
    #
    #     k = 7
    #     @test Ā[(k-1)*nm+1:(k-1)*nm+n,:] == prod(results.fdx[1:k-1])
    #     @test Ā[(N-1)*nm+1:Nz,:] == prod(results.fdx[1:N-1])
    #
    #     @test B̄[1:n,1:n] == zeros(n,n)
    #     @test B̄[n+1:nm,1:m] == 1.0*Matrix(I,m,m)
    #     @test B̄[nm+1:nm+n,1:m] == results.fdu[1][1:n,1:m]
    #     @test B̄[nm+n+1:2*nm,m+1:2*m] == 1.0*Matrix(I,m,m)
    #     @test B̄[(N-1)*nm+1:Nz,1:m] == prod(results.fdx[2:N-1])*results.fdu[1]
    #     @test B̄[(N-1)*nm+1:Nz,m+1:2*m] == prod(results.fdx[3:N-1])*results.fdu[2]
    #     @test B̄[(N-1)*nm+1:Nz,(N-2)*m+1:(N-1)*m] == results.fdu[N-1]
    #
    #     k = 1
    #     Q,R,H,q,r = taylor_expansion(costfun,results.X[k],results.U[k])
    #     @test Q̄[1:nm,1:nm] == [Q H'; H R]
    #     @test q̄[1:nm] == [q;r]
    #
    #     k = 13
    #     Q,R,H,q,r = taylor_expansion(costfun,results.X[k],results.U[k])
    #     @test Q̄[(k-1)*nm+1:k*nm,(k-1)*nm+1:k*nm] == [Q H'; H R]
    #     @test q̄[(k-1)*nm+1:k*nm] == [q;r]
    #
    #     k = N
    #     Qf,qf = taylor_expansion(costfun,results.X[k])
    #     @test Q̄[(k-1)*nm+1:Nz,(k-1)*nm+1:Nz] == Qf
    #     @test q̄[(k-1)*nm+1:Nz] == qf
    #
    #     @test C̄[1:p,1:nm] == [results.Cx[1] results.Cu[1]]
    #
    #     k = 9
    #     @test C̄[(k-1)*p+1:k*p,(k-1)*nm+1:k*nm] == [results.Cx[k] results.Cu[k]]
    #     @test C̄[(N-1)*p+1:Np,(N-1)*nm+1:Nz] == results.Cx[N]
    #
    #     @test c̄[1:p] == results.C[1]
    #
    #     k = 17
    #     @test c̄[(k-1)*p+1:k*p] == results.C[k]
    #     @test c̄[(N-1)*p+1:Np] == results.C[N]
    #
    #     @test all(idx_inequality[1:pI] .== true)
    #     @test all(idx_inequality[pI+1:p] .== false)
    #     k = 12
    #
    #     @test all(idx_inequality[(k-1)*p+1:(k-1)*p+pI] .== true)
    #     idx_inequality[(k-1)*p+1:(k-1)*p+pI]
    #     @test all(idx_inequality[(N-1)*p+1:(N-1)*pI_N] .== true)
    # end
    ū = -(B̄'*Q̄*B̄)\(B̄'*(q̄ + Q̄*Ā*x0) + B̄'*C̄'*λ_tmp)

    # println(any(isnan.(q̄)))
    # println(any(isnan.(Ā)))
    # println(any(isnan.(λ_tmp)))
    # println(any(isnan.(ū)))
    # println(any(isnan.(Ā)))
    # println(any(isnan.(B̄)))
    # println(any(isnan.(C̄)))
    # println(any(isnan.(c̄)))
    P = B̄'*Q̄*B̄
    M = q̄ + Q̄*Ā*x0

    # println(rank(P))
    # println(size(P))
    # println(λ_tmp)

    L = 0.5*ū'*P*ū + M'*B̄*ū + λ_tmp'*(C̄*B̄*ū + C̄*Ā*x0 + c̄)

    D = 0.5*M'*B̄*inv(P')*B̄'*M + 0.5*M'*B̄*inv(P')*B̄'*C̄'*λ_tmp + 0.5*λ_tmp'*C̄*B̄*inv(P')*B̄'*M + 0.5*λ_tmp'*C̄*B̄*inv(P')*B̄'*C̄'*λ_tmp - M'*B̄*inv(P)*B̄'*M - M'*B̄*inv(P)*B̄'*C̄'*λ_tmp - λ_tmp'*C̄*B̄*inv(P)*B̄'*M - λ_tmp'*C̄*B̄*inv(P)*B̄'*C̄'*λ_tmp + λ_tmp'*C̄*Ā*x0 + λ_tmp'*c̄
    Q_dual = C̄*B̄*inv(P')*B̄'*C̄' - 2*C̄*B̄*inv(P)*B̄'*C̄'
    q_dual = M'*B̄*inv(P')*B̄'*C̄' - 2*M'*B̄*inv(P)*B̄'*C̄' + x0'*Ā'*C̄' + c̄'
    qq_dual = 0.5*M'*B̄*inv(P')*B̄'*M - M'*B̄*inv(P)*B̄'*M

    DD = 0.5*λ_tmp'*Q_dual*λ_tmp + q_dual*λ_tmp + qq_dual

    @test isapprox(D,L)
    @test isapprox(DD,L)

    # solve QP
    m = JuMP.Model(solver=IpoptSolver(print_level=0))
    # m = JuMP.Model(solver=ClpSolver())

    # @variable(m, u[1:Nu])
    #
    # @objective(m, Min, 0.5*u'*P*u + M'*B̄*u)
    # @constraint(m, con, C̄*B̄*u + C̄*Ā*x0 + c̄ .== 0.)

    # @variable(m, λ[1:Np])
    N_active_set = sum(active_set)

    @variable(m, λ[1:N_active_set])

    # @objective(m, Min, λ'*λ)
    # @constraint(m, con, Q_dual[active_set,active_set]*λ + q_dual[active_set] .== 0.)
    #
    @objective(m, Max, 0.5*λ'*Q_dual[active_set,active_set]*λ + q_dual[active_set]'*λ)
    @constraint(m, con2, λ[idx_inequality[active_set]] .>= 0)
    #
    # # print(m)
    #
    status = JuMP.solve(m)
    #
    if status != :Optimal
        error("QP failed")
    end

    # Solution
    # println("Objective value: ", JuMP.getobjectivevalue(m))
    # println("λ = ", JuMP.getvalue(λ))

    λ_tmp[active_set] = JuMP.getvalue(λ)

    for k = 1:N
        if k != N
            idx = ((k-1)*p + 1):k*p
            results.λ[k] = λ_tmp[idx]#JuMP.getvalue(λ[idx])
            # results.λ[k][1:pI] = max.(0.0,results.λ[k][1:pI])
        else
            idx = ((N-1)*p + 1):Np
            results.λ[k] = λ_tmp[idx]#JuMP.getvalue(λ[idx])
            # results.λ[k][1:pI_N] = max.(0.0,results.λ[k][1:pI_N])
        end

        # tolerance check
        results.λ[k][abs.(results.λ[k]) .< 1e-8] .= 0.
    end
end

λ_second_order_update!(results_c,solver,true)

plot!(to_array(results_c.λ)[:,1:solver.N-1]')

solver = TrajectoryOptimization.Solver(model,obj,dt=0.01,opts=opts)
U = zeros(model.m,solver.N-1)
solver.opts.verbose = true
solver.opts.cost_tolerance = 1e-6
solver.opts.cost_tolerance_intermediate = 1e-6
solver.opts.gradient_tolerance = 1e-6
solver.opts.gradient_tolerance_intermediate = 1e-6
solver.opts.constraint_tolerance = 1e-6
solver.opts.use_second_order_dual_update = true
@time results_c,stats = TrajectoryOptimization.solve(solver, U)

plot(to_array(results_c.λ)[:,1:solver.N-1]')

plot(to_array(results_c.U)[:,1:solver.N-1]')

plot!(log.(stats["cost"] .+ 1000))
