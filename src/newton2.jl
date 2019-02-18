struct NewtonResults
    z::Vector # z = [x1;u1;...;xN]
    λ::Vector # constraint multipliers
    ν::Vector # dynamics constraints multipliers
    s::Vector # slack variables

    ∇²J::SparseMatrixCSC # Hessian of cost function (including penalty terms)
    ∇J::Vector # gradient of cost function (including penalty terms)

    ∇c::SparseMatrixCSC # Jacobian of constraints
    c::Vector # constraints

    ∇d::SparseMatrixCSC # Jacobian of dynamics constraints
    d::Vector # dynamics constraints

    A::SparseMatrixCSC # KKT Hessian matrix
    b::Vector # KKT gradient matrix

    active_set::Vector
    active_set_ineq::Vector

    x_eval::Vector

    δ::Vector

    Nz::Int
    Np::Int
    Nx::Int
end

function NewtonResults(Nz::Int,Np::Int,Nx::Int)
    z = zeros(Nz)
    λ = zeros(Np)
    ν = zeros(Nx)
    s = zeros(Np)

    ∇²J = spzeros(Nz,Nz)
    ∇J = zeros(Nz)

    ∇c = spzeros(Np,Nz)
    c = zeros(Np)

    ∇d = spzeros(Nx,Nz)
    d = zeros(Nx)

    A = spzeros(Nz+Np+Nx+Np,Nz+Np+Nx+Np)
    b = spzeros(Nz+Np+Nx+Np)

    active_set = zeros(Bool,Np)
    active_set_ineq = zeros(Bool,Np)
    x_eval = zeros(Nx)

    δ = zeros(Nz+Np+Nx+Np)

    Nz = Nz
    Np = Np
    Nx = Nx

    NewtonResults(z,λ,ν,s,∇²J,∇J,∇c,c,∇d,d,A,b,active_set,active_set_ineq,x_eval,δ,Nz,Np,Nx)
end

function copy(r::NewtonResults)
    NewtonResults(copy(r.z),copy(r.λ),copy(r.ν),copy(r.s),copy(r.∇²J),copy(r.∇J),copy(r.∇c),copy(r.c),copy(r.∇d),copy(r.d),copy(r.A),copy(r.b),copy(r.active_set),copy(r.active_set_ineq),copy(r.x_eval),copy(r.δ),copy(r.Nz),copy(r.Np),copy(r.Nx))
end

function NewtonResults(solver::Solver)
    n,m,N = get_sizes(solver)
    n̄,nn = get_num_states(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    # batch problem dimensions
    nm = nn + mm
    Nz = nn*N + mm*(N-1)
    Np = p*(N-1) + p_N
    Nx = N*nn

    NewtonResults(Nz,Np,Nx)
end

function get_batch_sizes(solver::Solver)
    n,m,N = get_sizes(solver)
    n̄,nn = get_num_states(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    # batch problem dimensions
    nm = nn + mm
    Nz = nn*N + mm*(N-1)
    Np = p*(N-1) + p_N
    Nx = N*n
    Nu = mm*(N-1) # number of control decision variables u

    return Nz,Np,Nx,Nu,nm
end

function newton_active_set!(newton_results::NewtonResults,results::SolverIterResults,solver::Solver,tolerance::Float64=1.0e3,slacks::Bool=true)
    # get problem dimensions
    n,m,N = get_sizes(solver)
    n̄,nn = get_num_states(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    Nz,Np,Nx,Nu,nm = get_batch_sizes(solver)

    active_set = newton_results.active_set
    active_set_ineq = newton_results.active_set_ineq

    s = newton_results.s

    active_set .= copy(vcat(results.active_set...))
    active_set_ineq .= 0 # reset

    for k = 1:N
        k != N ? p_idx = p : p_idx = p_N
        k != N ? pI_idx = pI : pI_idx = pI_N

        for i = 1:p_idx
            idx = (k-1)*p + i

            if i <= pI_idx
                if results.C[k][i] > -tolerance || results.λ[k][i] > 0.0
                    active_set[idx] = 1
                    active_set_ineq[idx] = 1
                    s[idx] == 0.0 ? s[idx] = sqrt(2.0*max(0.0,-results.C[k][i])) : nothing
                end
            else
                active_set[idx] = 1
            end
        end
    end
end

function update_newton_results!(newton_results::NewtonResults,results::SolverIterResults,solver::Solver)
    # get problem dimensions
    n,m,N = get_sizes(solver)
    n̄,nn = get_num_states(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    Nz,Np,Nx,Nu,nm = get_batch_sizes(solver)

    # pull out results for convenience
    X = results.X
    U = results.U
    Iμ = results.Iμ
    C = results.C
    Cx = results.Cx
    Cu = results.Cu
    fdx = results.fdx
    fdu = results.fdu
    x0 = solver.obj.x0

    # pull out newton results for convenience
    z = newton_results.z
    λ = newton_results.λ
    ν = newton_results.ν
    s = newton_results.s

    ∇²J = newton_results.∇²J
    ∇J = newton_results.∇J

    ∇c = newton_results.∇c
    c = newton_results.c

    ∇d = newton_results.∇d
    d = newton_results.d

    active_set = newton_results.active_set
    active_set_ineq = newton_results.active_set_ineq

    x_eval = newton_results.x_eval

    Inn = sparse(Matrix(I,nn,nn))

    # update batch matrices
    for k = 1:N
        x = results.X[k]

        if k != N
            u = results.U[k]
            Q,R,H,q,r = taylor_expansion(solver.obj.cost,x,u) .* solver.dt
        else
            Qf,qf = taylor_expansion(solver.obj.cost,x)
        end

        # Indices
        k != N ? idx = (((k-1)*nm + 1):k*nm) : idx = (((k-1)*nm + 1):Nz) # index over x and u
        k != N ? idx2 = (((k-1)*p + 1):k*p) : idx2 = (((k-1)*p + 1):Np) # index over p
        idx3 = ((k-1)*nm + 1):((k-1)*nm + nn) # index over x only
        idx4 = ((k-1)*nm + nn + 1):k*nm # index over u only
        idx5 = ((k-1)*mm + 1):k*mm # index through stacked u vector
        k != N ? idx6 = (((k-1)*p + 1):((k-1)*p + pI)) : idx6 = (((k-1)*p + 1):((k-1)*p + pI_N))# index over p only inequality indices
        idx7 = ((k-1)*nn + 1):(k*nn)
        idx8 = ((k-2)*nm + 1):((k-1)*nm + nn)
        idx9 = ((k-2)*nm + 1):((k-2)*nm + nn)
        idx10 = ((k-2)*nm + nn + 1):((k-1)*nm)

        k != N ? pI_idx = pI : pI_idx = pI_N

        # Assemble ∇²J, ∇J, ∇c, c, z
        if k != N
            ∇²J[idx3,idx3] = Q + Cx[k]'*Iμ[k]*Cx[k]
            ∇²J[idx3,idx4] = H' + Cx[k]'*Iμ[k]*Cu[k]
            ∇²J[idx4,idx3] = H + Cu[k]'*Iμ[k]*Cx[k]
            ∇²J[idx4,idx4] = R + Cu[k]'*Iμ[k]*Cu[k]

            ∇J[idx3] = q + Cx[k]'*Iμ[k]*C[k]
            ∇J[idx4] = r + Cu[k]'*Iμ[k]*C[k]

            ∇c[idx2,idx3] = Cx[k]
            ∇c[idx2,idx4] = Cu[k]

            c[idx2] = C[k]
            # c[idx6] += 0.5*s[idx6].^2

            z[idx] = [x;u]
        else
            ∇²J[idx,idx] = Qf + Cx[N]'*Iμ[N]*Cx[N]
            ∇J[idx] = qf + Cx[N]'*Iμ[N]*C[N]

            ∇c[idx2,idx] = results.Cx[N]
            c[idx2] = results.C[N]
            # c[idx6] += 0.5*s[idx6] .^2

            z[idx] = x
        end

        # assemble ∇d, d
        if k == 1
            ∇d[1:nn,1:nn] = Inn
            d[1:nn] = X[1] - solver.obj.x0
        else
            ∇d[idx7,idx9] = -fdx[k-1]
            ∇d[idx7,idx10] = -fdu[k-1]
            ∇d[idx7,idx3] = Inn

            solver.fd(view(x_eval,idx7),X[k-1][1:n],U[k-1][1:m])
            d[idx7] = X[k] - x_eval[idx7]
        end

        # assemble λ, ν
        λ[idx2] = results.λ[k]
        ν[idx7] = results.s[k]

        # assembly active set indices
        idx_as_pI = idx6[active_set_ineq[idx6]] # indices of active set for inequality constraints

    end

    c[active_set_ineq] += 0.5*s[active_set_ineq].^2

    return nothing
end

function update_results_from_newton_results!(results::SolverIterResults,newton_results::NewtonResults,solver::Solver)
    n,m,N = get_sizes(solver)
    n̄,nn = get_num_states(solver)
    m̄,mm = get_num_controls(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    # batch problem dimensions
    nm = nn + mm
    Nz = nn*N + mm*(N-1)
    Np = p*(N-1) + p_N
    Nu = mm*(N-1) # number of control decision variables u

    z = newton_results.z
    λ = newton_results.λ

    # update results with stack vector
    for k = 1:N
        idx = ((k-1)*nm + 1):k*nm # index over x and u
        k != N ? idx2 = (((k-1)*p + 1):k*p) : idx2 = (((k-1)*p + 1):Np) # index over p
        k != N ? idx3 = (((k-1)*nm + 1):((k-1)*nm + nn)) : idx3 = (((k-1)*nm + 1):Nz)# index over x only
        idx4 = ((k-1)*nm + nn + 1):k*nm # index over u only
        idx5 = ((k-1)*mm + 1):k*mm # index through stacked u vector
        idx6 = ((k-1)*p + 1):((k-1)*p + pI) # index over p only inequality indices

        results.X[k] = z[idx3]
        k != N ? results.U[k] = z[idx4] : nothing
        results.λ[k] = λ[idx2]
    end

    update_constraints!(results,solver)
    update_jacobians!(results,solver)

    return nothing
end

function solve_kkt!(newton_results::NewtonResults)
    z = newton_results.z
    λ = newton_results.λ
    ν = newton_results.ν
    s = newton_results.s

    ∇²J = newton_results.∇²J
    ∇J = newton_results.∇J

    ∇c = newton_results.∇c
    c = newton_results.c

    ∇d = newton_results.∇d
    d = newton_results.d

    A = newton_results.A
    b = newton_results.b

    active_set = newton_results.active_set
    active_set_ineq = newton_results.active_set_ineq

    δ = newton_results.δ

    # get batch problem sizes
    Nz = newton_results.Nz
    Np = newton_results.Np
    Nx = newton_results.Nx

    Np_as = sum(active_set)

    # assemble KKT matrix,vector
    _idx1 = Array(1:Nz)
    _idx2 = Array((1:Np) .+ Nz)[active_set]
    _idx3 = Array((1:Nx) .+ (Nz + Np))
    _idx4 = Array((1:Np) .+ (Nz + Np + Nx))[active_set_ineq]

    _idx5 = [_idx1;_idx2;_idx3;_idx4]

    A[_idx1,_idx1] = ∇²J
    A[_idx1,_idx2] = ∇c[active_set,:]'
    A[_idx1,_idx3] = ∇d'
    A[_idx2,_idx1] = ∇c[active_set,:]
    A[_idx3,_idx1] = ∇d

    A[_idx2,_idx4] = Diagonal(s)[active_set_ineq,active_set]'
    A[_idx4,_idx2] = Diagonal(s)[active_set_ineq,active_set]
    A[_idx4,_idx4] = Diagonal(λ)[active_set_ineq,active_set_ineq]

    b[_idx1] = -(∇J + ∇c[active_set,:]'*λ[active_set] + ∇d'*ν)
    b[_idx2] = -c[active_set]
    b[_idx3] = -d
    b[_idx4] = -(λ .* s)[active_set_ineq]

    ## indexing
    # println("A:")
    # println(rank(Array(A)))
    # println(size(A))
    println("Active set A:")
    println("rank: $(rank(Array(A[_idx5,_idx5])))")
    println("size: $(size(A[_idx5,_idx5]))")
    println("Nz: $Nz")
    println("Np: $Np")
    println("Nx: $Nx")
    println("active set: $(sum(active_set))")
    println("active set (ineq.): $(sum(active_set_ineq))")
    # println(rank(Array(A[[_idx1;_idx2;_idx3],[_idx1;_idx2;_idx3]])))
    # println(length([_idx1;_idx2;_idx3]))
    # println("No slack A:")
    # println(rank(Array(A[_idx_tmp,_idx_tmp])))
    # println(size(A[_idx_tmp,_idx_tmp]))
    # println("slacks")
    # println(sum(active_set))
    # println("reg.")
    # println(rank(Array(A)+Matrix(I,Nz+Np+Nx+Np,Nz+Np+Nx+Np)))
    # println("KKT size: $(length(_idx5))")

    δ[_idx5] = A[_idx5,_idx5]\b[_idx5]

    # z .+= alpha*δ[_idx1]
    # λ[active_set] += alpha*δ[_idx2]
    # ν .+= alpha*δ[_idx3]
    # s[active_set_ineq] += alpha*δ[_idx4]

    return nothing
end

function newton_step!(results::SolverIterResults,newton_results::NewtonResults,solver::Solver,alpha::Float64=1.0)
    update_newton_results!(newton_results,results,solver)
    solve_kkt!(newton_results)

    ###
    _idx1 = Array(1:newton_results.Nz)
    _idx2 = Array((1:newton_results.Np) .+ newton_results.Nz)[newton_results.active_set]
    _idx3 = Array((1:newton_results.Nx) .+ (newton_results.Nz + newton_results.Np))
    _idx4 = Array((1:newton_results.Np) .+ (newton_results.Nz + newton_results.Np + newton_results.Nx))[newton_results.active_set_ineq]

    newton_results.z .+= alpha*newton_results.δ[_idx1]
    newton_results.λ[newton_results.active_set] += alpha*newton_results.δ[_idx2]
    newton_results.λ[newton_results.active_set_ineq] = max.(0,newton_results.λ[newton_results.active_set_ineq])
    newton_results.ν .+= alpha*newton_results.δ[_idx3]
    newton_results.s[newton_results.active_set_ineq] += alpha*newton_results.δ[_idx4]
    ###

    update_results_from_newton_results!(results,newton_results,solver)
    newton_active_set!(newton_results,results,solver)
    return nothing
end

function newton_solve!(results::SolverIterResults,solver::Solver)
    # copy current results
    results = copy(results)
    results_new = copy(results)

    # instantiate Newton Results
    newton_results = NewtonResults(solver)

    # get initial cost and max constraint violation
    newton_active_set!(newton_results,results_new,solver)
    update_newton_results!(newton_results,results_new,solver)
    J_prev = cost(solver,results_new)
    c_max = max_violation(results_new)

    println("Newton initialization")
    println("Cost: $J_prev")
    println("c_max: $c_max")
    println("\n")

    α = 1.0
    max_iter = 100
    max_c = 1e-8
    iter = 1
    ls_param = 0.01
    J = Inf

    while (c_max > max_c) && iter <= max_iter
        newton_step!(results_new,newton_results,solver,α)
        backwardpass!(results_new,solver)
        rollout!(results_new,solver,1.0)
        results_new.X .= deepcopy(results_new.X_)
        results_new.U .= deepcopy(results_new.U_)
        update_constraints!(results_new,solver)
        J = cost(solver,results_new)
        c_max = max_violation(results_new)

        println("*Newton step: $iter")
        println("α = $α")
        println("Cost: $J \n    prev. Cost: $J_prev")
        println("ΔJ = $(J-J_prev)")
        println("c_max: $c_max")

        iter == 1 ? J_prev = J : nothing

        if J <= J_prev + ls_param*newton_results.b'*α*newton_results.δ
            results = copy(results_new)
            J_prev = copy(J)
            println("improved")
            α = 1.0
        else
            println("α = $α increased cost")
            results_new = copy(results)
            newton_results = NewtonResults(solver)
            newton_active_set!(newton_results,results_new,solver)
            update_newton_results!(newton_results,results_new,solver)
            α = 0.5*α
        end
        println("-----\n")
        iter += 1
    end
    println("Newton solve complete")
    println("J: $J")
    println("c_max: $c_max")
end

# """
# $(SIGNATURES)
#     Time Varying Discrete Linear Quadratic Regulator (TVLQR)
# """
# function lqr(A::Array, B::Array, Q::AbstractMatrix, R::AbstractMatrix, Qf::AbstractMatrix)::Array
#     n,m = size(B[1])
#     N_ = length(B)
#     K = [zeros(m,n) for k = 1:N_]
#     S = zeros(n,n)
#
#     # Boundary condition
#     S .= Qf
#
#     # Riccati
#     for k = 1:N_
#         # Compute control gains
#         K[k] = -(R + B[k]'*S*B[k])\(B[k]'*S*A[k])
#
#         # Calculate cost-to-go for backward propagation
#         S .= Q + A[k]'*S*A[k] - A[k]'*S*B[k]*K[k]
#     end
#     return K
# end
#
# function lqr(results::SolverResults, solver::Solver)::Array
#     Q, R, Qf = get_cost_matrices(solver)
#     A, B = results.fdx, results.fdu
#     return lqr(A,B,Q,R,Qf)
# end
