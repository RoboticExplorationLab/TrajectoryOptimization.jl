using Test
using Plots

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-6
opts.constraint_tolerance = 1e-3
opts.square_root = true
opts.outer_loop_update_type = :default
opts.live_plotting = false
model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -10
u_max = 10
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=5.0,use_xf_equality_constraint=true)#, u_min=u_min, u_max=u_max)#, x_min=x_min, x_max=x_max)
solver = TrajectoryOptimization.Solver(model,obj_con,integration=:rk4,N=10,opts=opts)
U = rand(solver.model.m, solver.N)

results, stats = TrajectoryOptimization.solve(solver,U)
@assert max_violation(results) < opts.constraint_tolerance
plot(results.X,title="Block push to origin",xlabel="time step",ylabel="state",label=["pos.";"vel."])
plot(results.U,title="Block push to origin",xlabel="time step",ylabel="control")
stats["cost"][end]
stats["c_max"][end]


## Newton solve
function update_batch_problem(results::SolverIterResults,solver::Solver,z̄::Vector=Vector{Float64}(undef,0),λ_::Vector=Vector{Float64}(undef,0),ν_::Vector=Vector{Float64}(undef,0))
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
    Nu = mm*(N-1) # number of control decision variables u

    # update results with stack vector
    if !isempty(z̄) || !isempty(λ_)
        update_results_from_batch!(results,solver,z̄,λ_,ν_)
    else
        @warn "using previous results"
    end

    # update constraints and Jacobians
    update_constraints!(results,solver)
    update_jacobians!(results,solver)

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

    # initialize batch matrices
    Q̄ = zeros(Nz,Nz)
    q̄ = zeros(Nz)

    C̄ = zeros(Np,Nz)
    c̄ = zeros(Np)

    D̄ = zeros(N*n,Nz)
    d̄ = zeros(N*n)

    z̄ = zeros(Nz)

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
    λ_ = vcat(results.λ...)
    ν_ = vcat(results.s...)

    return z̄, λ_, ν_, Q̄, q̄, C̄, c̄, D̄, d̄
end

function update_results_from_batch!(results::SolverIterResults,solver::Solver,z̄::Vector,λ_::Vector,ν_::Vector)
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
    end
    update_constraints!(results,solver)
    update_jacobians!(results,solver)

    return nothing
end

function solve_KKT(z̄::Vector, λ_::Vector, ν_::Vector, Q̄::Matrix, q̄::Vector, C̄::Matrix, c̄::Vector, D̄::Matrix, d̄::Vector)
    # get batch problem sizes
    Nz = size(Q̄,1)
    Np = size(C̄,1)
    Nx = size(D̄,1)

    # initialize KKT matrix/vector
    A = zeros(Nz+Np+Nx,Nz+Np+Nx)
    b = zeros(Nz+Np+Nx)

    # assemble KKT matrix/vector
    A[1:Nz,1:Nz] = Q̄
    A[1:Nz,Nz+1:Nz+Np] = C̄'
    A[1:Nz,Nz+Np+1:Nz+Np+Nx] = D̄'
    A[Nz+1:Nz+Np,1:Nz] = C̄
    A[Nz+Np+1:Nz+Np+Nx,1:Nz] = D̄

    b[1:Nz] = -q̄
    b[Nz+1:Nz+Np] = -c̄
    b[Nz+Np+1:Nz+Np+Nx] = -d̄

    δ = A\b

    tmp = [z̄;λ_;ν_]

    α = 1.0
    tmp_new = tmp + α*δ

    println(norm(tmp_new - tmp))

    return tmp_new[1:Nz], tmp_new[Nz+1:Nz+Np], tmp_new[Nz+Np+1:Nz+Np+Nx] # z,λ,ν
end

function cost_newton(results::SolverIterResults,solver::Solver,ν_::Vector)
    results = copy(results)

    # get problem dimensions
    n,m,N = get_sizes(solver)

    J = cost(solver,results)

    # add dynamics constraint costs
    tmp = zeros(n)
    for k = 1:N
        if k == 1
            J += ν_[1:n]'*(X[1] - solver.obj.x0)
        else
            idx = ((k-1)*n+1):k*n
            solver.fd(tmp,results.X[k-1][1:n],results.U[k-1][1:m])
            J += ν_[idx]'*(X[k] - tmp)
        end
    end
    return J
end

function newton_solve(results::SolverIterResults,solver::Solver)
    results = copy(results)
    z̄, λ_, ν_, Q̄, q̄, C̄, c̄, D̄, d̄ = update_batch_problem(results,solver)
    z̄_new,λ_new,ν_new = solve_KKT(z̄, λ_, ν_, Q̄, q̄, C̄, c̄, D̄, d̄)
    update_results_from_batch!(results,solver,z̄_new,λ_new,ν_new)

    J = cost_newton(results,solver,ν_new)
    c_max = max_violation(results)

    return results, J, c_max
end

results_newton, J_newton, c_max_newton = newton_solve(results,solver)




# results_newton = copy(results)
# norm(vcat(results.X...)-vcat(results_newton.X...))
#
# z̄, λ_, ν_, Q̄, q̄, C̄, c̄, D̄, d̄ = update_batch_problem(results_newton,solver)
# cost_newton(results_newton,solver,ν_)
#
# z̄a,λ_a,ν_a = solve_KKT(z̄, λ_, ν_, Q̄, q̄, C̄, c̄, D̄, d̄)
# norm(z̄-z̄a)
# norm(λ_-λ_a)
# norm(ν_ -ν_a)
# results_newton_update = copy(results_newton)
# update_results_from_batch!(results_newton_update,solver,z̄a,λ_a,ν_a)
# norm(vcat(results_newton_update.C...)-vcat(results_newton.C...))
# cost_newton(results_newton_update,solver,ν_a) - cost_newton(results_newton,solver,ν_)
# max_violation(results_newton_update)
# max_violation(results_newton)
#
# results_newton_update.C
# plot(results_newton_update.U)
# plot!(results_newton.U)
# results_newton_update.X[1]
#
# z̄, λ_, ν_, Q̄, q̄, C̄, c̄, D̄, d̄ = update_batch_problem(results_newton,solver)
#
#
# J = cost_newton(z̄,λ_,ν_,results_newton,solver)
# max_violation(results_newton)
#
#
# norm(vcat(results.X...) - vcat(results_newton.X...))
#
# vcat(results.C...)
# vcat(results_newton)
