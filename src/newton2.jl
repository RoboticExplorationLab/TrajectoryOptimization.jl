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

    r::Vector # λs constraint

    A::SparseMatrixCSC # KKT Hessian matrix
    b::Vector # KKT gradient matrix

    active_set::Vector
    active_set_ineq::Vector

    x_eval::Vector

    δ::Vector
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

    r = zeros(Np)

    A = spzeros(Nz+Np+Nx+Np,Nz+Np+Nx+Np)
    b = spzeros(Nz+Np+Nx+Np)

    active_set = zeros(Bool,Np)
    active_set_ineq = zeros(Bool,Np)
    x_eval = zeros(Nx)

    δ = zeros(Nz+Np+Nx+Np)

    NewtonResults(z,λ,ν,s,∇²J,∇J,∇c,c,∇d,d,r,A,b,active_set,active_set_ineq,x_eval,δ)
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

function newton_active_set!(newton_results::NewtonResults,results::SolverIterResults,solver::Solver,tolerance::Float64=1e-8,slacks::Bool=true)
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

    for k = 1:N
        k != N ? p_idx = p : p_idx = p_N
        k != N ? pI_idx = pI : pI_idx = pI_N

        for i = 1:p_idx
            idx = (k-1)*p + i
            if i <= pI_idx && results.C[k][i] > -tolerance
                active_set[idx] = 1
                active_set_ineq[idx] = 1

                if slacks
                    s[idx] = sqrt(2.0*max(0.0,-results.C[k][i]))
                end
            elseif i > pI_idx
                active_set[idx] = 1
            else
                active_set[idx] = 0
                active_set_ineq[idx] = 0
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

    r = newton_results.r

    active_set = newton_results.active_set
    active_set_ineq = newton_results.active_set_ineq

    x_eval = newton_results.x_eval

    Inn = sparse(Matrix(I,nn,nn))

    # update batch matrices
    for k = 1:N
        x = results.X[k]

        if k != N
            u = results.U[k]
            Q,R,H,q,r̄ = taylor_expansion(solver.obj.cost,x,u) .* solver.dt
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
            ∇J[idx4] = r̄ + Cu[k]'*Iμ[k]*C[k]

            ∇c[idx2,idx3] = Cx[k]
            ∇c[idx2,idx4] = Cu[k]

            c[idx2] = C[k]
            c[idx6] += 0.5*s[idx6].^2

            z[idx] = [x;u]
        else
            ∇²J[idx,idx] = Qf + Cx[N]'*Iμ[N]*Cx[N]
            ∇J[idx] = qf + Cx[N]'*Iμ[N]*C[N]

            ∇c[idx2,idx] = results.Cx[N]
            c[idx2] = results.C[N]
            c[idx6] += 0.5*s[idx6] .^2

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

        # assemble λ, ν, ϕ
        λ[idx2] = results.λ[k]
        ν[idx7] = results.s[k] #TODO confirm that penalty isn't breaking this

        # assembly active set indices
        idx_as_pI = idx6[active_set_ineq[idx6]] #indices of active set for inequality constraints

        # assemble r
        r[idx_as_pI] = (s[idx6] .* results.λ[k][1:pI_idx])[active_set_ineq[idx6]]
    end

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

function solve_kkt!(newton_results::NewtonResults,alpha::Float64=1.0)
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

    r = newton_results.r

    A = newton_results.A
    b = newton_results.b

    active_set = newton_results.active_set
    active_set_ineq = newton_results.active_set_ineq

    δ = newton_results.δ

    # get batch problem sizes
    Nz = size(∇²J,1)
    Np = size(∇c,1)
    Nx = size(∇d,1)

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
    b[_idx4] = -r[active_set_ineq]

    ## indexing
    println("A:")
    println(rank(Array(A)))
    println(size(A))
    println("Active set A:")
    println(rank(Array(A[_idx5,_idx5])))
    println(size(A[_idx5,_idx5]))
    println("Nz: $Nz")
    println("Np: $Np")
    println("Nx: $Nx")
    println("active set: $(sum(active_set))")
    println("active set (ineq.): $(sum(active_set_ineq))")
    # println("No slack A:")
    # println(rank(Array(A[_idx_tmp,_idx_tmp])))
    # println(size(A[_idx_tmp,_idx_tmp]))
    # println("slacks")
    # println(sum(active_set))
    # println("reg.")
    # println(rank(Array(A)+Matrix(I,Nz+Np+Nx+Np,Nz+Np+Nx+Np)))
    println("KKT size: $(length(_idx5))")

    δ[_idx5] = A[_idx5,_idx5]\b[_idx5]

    z .+= alpha*δ[(1:Nz)]
    λ .+= alpha*δ[(1:Np) .+ Nz]
    ν .+= alpha*δ[(1:Nx) .+ (Nz + Np)]
    s .+= alpha*δ[(1:Np) .+ (Nz + Np + Nx)]

    return nothing
end

function cost_newton(results::SolverIterResults,newton_results::NewtonResults,solver::Solver)
    # get problem dimensions
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    s = newton_results.s
    λ = newton_results.λ
    # results_new = copy(results)
    # for k = 1:N
    #     k != N ? idx = (((k-1)*p + 1):k*p) : idx = (((k-1)*p+1):(k-1)*p+p_N)
    #     results_new.C[k] += 0.5*(s[idx].^2)
    # end
    J = cost(solver,results)#_new)
    J += sum(0.5*(λ .* s.^2)[newton_results.active_set_ineq])

    # add dynamics constraint costs
    X = results.X
    U = results.U
    ν = newton_results.ν
    x_eval = newton_results.x_eval

    for k = 1:N
        if k == 1
            J += ν[1:n]'*(X[1] - solver.obj.x0)
        else
            idx = ((k-1)*n+1):k*n
            solver.fd(view(x_eval,idx),X[k-1][1:n],U[k-1][1:m])
            J += ν[idx]'*(X[k] - x_eval[idx])
        end
    end

    return J
end

function newton_step!(results::SolverIterResults,newton_results::NewtonResults,solver::Solver,alpha::Float64=1.0)
    update_newton_results!(newton_results,results,solver)
    solve_kkt!(newton_results,alpha)
    update_results_from_newton_results!(results,newton_results,solver)
    return nothing
end

function newton_solve!(results::SolverIterResults,solver::Solver)
    # instantiate Newton Results
    newton_results = NewtonResults(solver)

    # copy current results
    results_new = copy(results)

    # get initial cost and max constraint violation
    newton_active_set!(newton_results,results,solver)
    update_newton_results!(newton_results,results_new,solver)
    J_prev = cost_newton(results_new,newton_results,solver)
    c_max = max_violation(results_new)

    println("Newton initialization")
    println("Cost: $J_prev")
    println("c_max: $c_max")

    α = 1.0
    max_iter = 10
    max_c = 1e-8
    iter = 1
    ls_param = 0.01

    while c_max > max_c && iter <= max_iter
        println("$α")
        newton_step!(results_new,newton_results,solver,α)
        J = cost_newton(results_new,newton_results,solver)
        c_max = max_violation(results_new)
        println("Newton step: $iter")
        println("Cost: $J \n    prev. Cost: $J_prev")
        println("c_max: $c_max")

        if J <= J_prev + ls_param*newton_results.b'*α*newton_results.δ
            results = copy(results_new)
            J_prev = copy(J)
            println("improved")
            α = 1.0
        else
            println("what??")
            results_new = copy(results)
            α = 0.5*α
        end
        iter += 1
    end
    println("Newton solver end")
    println("c_max: $c_max")
end
