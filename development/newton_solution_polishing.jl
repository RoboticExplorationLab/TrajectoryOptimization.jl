using Test
using Plots

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-6
opts.constraint_tolerance = 1e-2
opts.square_root = true
opts.outer_loop_update_type = :default
opts.live_plotting = false
###
model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -0.2
u_max = 0.2
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=5.0,use_xf_equality_constraint=true, u_min=u_min, u_max=u_max)#, x_min=x_min, x_max=x_max)
###

###
model, obj = TrajectoryOptimization.Dynamics.pendulum
obj_con = Dynamics.pendulum_constrained[2]
obj_con.u_min[1] = -1
###

###
model, obj_con = TrajectoryOptimization.Dynamics.dubinscar_parallelpark
###

solver = TrajectoryOptimization.Solver(model,obj_con,integration=:rk4,N=30,opts=opts)
U = rand(solver.model.m, solver.N)


results, stats = TrajectoryOptimization.solve(solver,U)
@assert max_violation(results) < opts.constraint_tolerance
# plot(results.X,title="Block push to origin",xlabel="time step",ylabel="state",label=["pos.";"vel."])
# plot(results.U,title="Block push to origin",xlabel="time step",ylabel="control")
plot(to_array(results.X)[1,:],to_array(results.X)[2,:])
stats["cost"][end]
stats["c_max"][end]

struct NewtonResults
    z̄::Vector
    λ_::Vector
    ν_::Vector
    Q̄::Matrix
    q̄::Vector
    C̄::Matrix
    c̄::Vector
    D̄::Matrix
    d̄::Vector
    active_set::Vector
end

function NewtonResults(Nz::Int,Np::Int,Nx::Int)
    z̄ = zeros(Nz)
    λ_ = zeros(Np)
    ν_ = zeros(Nx)

    Q̄ = zeros(Nz,Nz)
    q̄ = zeros(Nz)

    C̄ = zeros(Np,Nz)
    c̄ = zeros(Np)

    D̄ = zeros(Nx,Nz)
    d̄ = zeros(Nx)

    z̄ = zeros(Nz)

    active_set = zeros(Bool,Np)

    NewtonResults(z̄,λ_,ν_,Q̄,q̄,C̄,c̄,D̄,d̄,active_set)
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
    Nx = N*n

    NewtonResults(Nz,Np,Nx)
end

## Newton solve
function update_newton_results!(newton_results::NewtonResults,results::SolverIterResults,solver::Solver)
    # get problem dimensions
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

    newton_results.λ_ .= vcat(results.λ...)
    newton_results.ν_ .= vcat(results.s...)
    newton_results.active_set .= vcat(results.active_set...)

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
    Q̄ = newton_results.Q̄
    q̄ = newton_results.q̄

    C̄ = newton_results.C̄
    c̄ = newton_results.c̄

    D̄ = newton_results.D̄
    d̄ = newton_results.d̄

    z̄ = newton_results.z̄

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
        idx = ((k-1)*nm + 1):k*nm # index over x and u
        idx2 = ((k-1)*p + 1):k*p # index over p
        idx3 = ((k-1)*nm + 1):((k-1)*nm + nn) # index over x only
        idx4 = ((k-1)*nm + nn + 1):k*nm # index over u only
        idx5 = ((k-1)*mm + 1):k*mm # index through stacked u vector
        idx6 = ((k-1)*p + 1):((k-1)*p + pI) # index over p only inequality indices

        # Assemble Q̄, q̄, C̄, c̄
        if k != N
            Q̄[idx,idx] = [Q H';H R] + [Cx[k]'*Iμ[k]*Cx[k] Cx[k]'*Iμ[k]*Cu[k]; Cu[k]'*Iμ[k]*Cx[k] Cu[k]'*Iμ[k]*Cu[k]]
            q̄[idx] = [q;r] + [Cx[k]'*Iμ[k]*C[k]; Cu[k]'*Iμ[k]*C[k]]

            C̄[idx2,idx] = [Cx[k] Cu[k]]
            c̄[idx2] = C[k]

            z̄[idx] = [x;u]
        else
            idx = ((k-1)*nm + 1):Nz
            Q̄[idx,idx] = Qf + Cx[N]'*Iμ[N]*Cx[N]
            q̄[idx] = qf + Cx[N]'*Iμ[N]*C[N]

            idx2 = ((k-1)*p + 1):Np
            C̄[idx2,idx] = results.Cx[N]
            c̄[idx2] = results.C[N]

            z̄[idx] = x
        end

        if k == 1
            D̄[1:nn,1:nn] = Matrix(I,nn,nn)
            d̄[1:nn] = X[1] - solver.obj.x0
        else
            idx7 = ((k-1)*nn + 1):(k*nn)
            idx8 = ((k-2)*nm + 1):((k-1)*nm + nn)
            D̄[idx7,idx8] = [-fdx[k-1] -fdu[k-1] Matrix(I,nn,nn)]
            tmp = zeros(nn)
            solver.fd(tmp,X[k-1][1:n],U[k-1][1:m])
            d̄[idx7] = X[k] - tmp
        end
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

    z̄ = newton_results.z̄
    λ_ = newton_results.λ_
    active_set = newton_results.active_set

    # update results with stack vector
    for k = 1:N
        idx = ((k-1)*nm + 1):k*nm # index over x and u
        k != N ? idx2 = (((k-1)*p + 1):k*p) : idx2 = (((k-1)*p + 1):Np) # index over p
        k != N ? idx3 = (((k-1)*nm + 1):((k-1)*nm + nn)) : idx3 = (((k-1)*nm + 1):Nz)# index over x only
        idx4 = ((k-1)*nm + nn + 1):k*nm # index over u only
        idx5 = ((k-1)*mm + 1):k*mm # index through stacked u vector
        idx6 = ((k-1)*p + 1):((k-1)*p + pI) # index over p only inequality indices

        results.X[k] = z̄[idx3]
        k != N ? results.U[k] = z̄[idx4] : nothing
        results.λ[k] = λ_[idx2]
        results.active_set[k] = active_set[idx2]
    end
    update_constraints!(results,solver)
    update_jacobians!(results,solver)

    return nothing
end

function solve_KKT!(newton_results::NewtonResults,alpha::Float64=1.0)
    Q̄ = newton_results.Q̄
    q̄ = newton_results.q̄

    C̄ = newton_results.C̄
    c̄ = newton_results.c̄

    D̄ = newton_results.D̄
    d̄ = newton_results.d̄

    z̄ = newton_results.z̄
    λ_ = newton_results.λ_
    ν_ = newton_results.ν_

    # get batch problem sizes
    Nz = size(Q̄,1)
    Np = size(C̄,1)
    Nx = size(D̄,1)

    active_set = newton_results.active_set

    Np_as = sum(active_set)

    # initialize KKT matrix/vector
    A = zeros(Nz+Np_as+Nx,Nz+Np_as+Nx)
    b = zeros(Nz+Np_as+Nx)

    # assemble KKT matrix/vector
    A[1:Nz,1:Nz] = Q̄
    A[1:Nz,Nz+1:Nz+Np_as] = C̄[active_set,:]'
    A[1:Nz,Nz+Np_as+1:Nz+Np_as+Nx] = D̄'
    A[Nz+1:Nz+Np_as,1:Nz] = C̄[active_set,:]
    A[Nz+Np_as+1:Nz+Np_as+Nx,1:Nz] = D̄

    b[1:Nz] = -q̄
    b[Nz+1:Nz+Np_as] = -c̄[active_set]
    b[Nz+Np_as+1:Nz+Np_as+Nx] = -d̄

    δ = A\b

    tmp = [z̄;λ_[active_set];ν_]

    tmp_new = tmp + alpha*δ

    z̄ .= tmp_new[1:Nz]
    λ_[active_set] = tmp_new[Nz+1:Nz+Np_as]
    ν_ .= tmp_new[Nz+Np_as+1:Nz+Np_as+Nx]

    return nothing
end

function cost_newton(results::SolverIterResults,newton_results::NewtonResults,solver::Solver)
    results = copy(results)

    # get problem dimensions
    n,m,N = get_sizes(solver)

    J = cost(solver,results)

    # add dynamics constraint costs
    tmp = zeros(n)
    X = results.X
    U = results.U
    ν_ = newton_results.ν_
    for k = 1:N
        if k == 1
            J += ν_[1:n]'*(X[1] - solver.obj.x0)
        else
            idx = ((k-1)*n+1):k*n
            solver.fd(tmp,X[k-1][1:n],U[k-1][1:m])
            J += ν_[idx]'*(X[k] - tmp)
        end
    end
    return J
end

function newton_solve(results::SolverIterResults,solver::Solver,alpha::Float64=1.0)
    results_new = copy(results)
    newton_results = NewtonResults(solver)
    update_newton_results!(newton_results,results_new,solver)
    solve_KKT!(newton_results,alpha)
    update_results_from_newton_results!(results_new,newton_results,solver)
    J = cost_newton(results_new,newton_results,solver)
    c_max = max_violation(results_new)
    return results_new, J, c_max
end

results_newton, J_newton, c_max_newton = newton_solve(results,solver,1.0)

# plot(results_newton.U,title="Pendulum",xlabel="time step",ylabel="control",label="Newton",legend=:bottomright)
# plot!(results.U,label="AuLa")

x_min = obj_con.x_min
x_max = obj_con.x_max
plt = plot(title="Parallel Park")#,aspect_ratio=:equal)
plot!(x_min[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(x_max[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_min[2]*ones(1000),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_max[2]*ones(1000),color=:red,width=2,label="")
plot_trajectory!(to_array(results_newton.X),width=2,color=:blue,label="Newton",legend=:bottomright)
plot_trajectory!(to_array(results.X),width=2,color=:green,label="AuLa",legend=:bottomright)


# Cost
cost(solver,results)
J_newton
# Final max constraint tolerance
max_violation(results)
max_violation(results_newton)
