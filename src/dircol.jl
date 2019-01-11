"""
$(SIGNATURES)
Solve a trajectory optimization problem with direct collocation

# Arguments
* model: TrajectoryOptimization.Model for the dynamics
* obj: TrajectoryOptimization.ConstrainedObjective to describe the cost function
* dt: time step. Used to determine the number of knot points. May be modified
    by the solver in order to achieve an integer number of knot points.
* method: Collocation method.
    :midpoint - Zero order interpolation on states and controls.
    :trapezoidal - First order interpolation on states and zero order on control.
    :hermite_simpson_separated - Hermite Simpson collocation with the midpoints
        included as additional decision variables with constraints
    :hermite_simpson - condensed verision of Hermite Simpson. Currently only
        supports grads=:auto or grads=:none (recommended)
* grads: Specifies the gradient information provided to SNOPT.
    :none - returns no gradient information to SNOPT
    :auto - uses ForwardDiff to calculate gradients
    :quadratic - uses functions exploiting quadratic cost functions
"""
function solve_dircol(solver::Solver,X0::Matrix,U0::Matrix;
        nlp::Symbol=:ipopt, method::Symbol=:auto, grads::Symbol=:quadratic, start=:cold, options::Dict{String,T}=Dict{String,Any}()) where T

    if solver.obj isa UnconstrainedObjective
        solver = Solver(solver,obj=ConstrainedObjective(solver.obj))
        # solver.state.constrained = false
    end

    obj = solver.obj
    model = solver.model

    X0 = copy(X0)
    U0 = copy(U0)

    if method == :auto
        if solver.integration == :rk3_foh
            method = :hermite_simpson
        elseif solver.integration == :midpoint
            method = :midpoint
        else
            method = :hermite_simpson
        end
    end

    # Constants
    n,m,N = get_sizes(solver)
    m̄, = get_num_controls(solver)
    dt = solver.dt
    N = convert_N(N,method)

    if N != size(X0,2)
        solver.opts.verbose ? println("Interpolating initial guess") : nothing
        X0,U0 = interp_traj(N,obj.tf,X0,U0)
    end

    if size(U0,1) < m̄
        U0 = [U0; ones(1,size(U0,2))*get_initial_dt(solver)]
    end

    if nlp == :ipopt
        return solve_ipopt(solver,X0,U0,method,options=options)
    elseif nlp == :snopt
        return solve_snopt(solver,X0,U0, method=method, grads=grads, start=start)
    else
        ArgumentError("$nlp is not a known NLP solver. Options are (:ipopt, :snopt)")
    end
    # return sol, stats, prob
end

"""
$(SIGNATURES)
Automatically generate an initial guess by linearly interpolating the state
between initial and final state and settings the controls to zero.
"""
function solve_dircol(solver::Solver;
        method::Symbol=:auto, grads::Symbol=:quadratic, start=:cold)
    # Constants
    N = solver.N
    N = convert_N(N,method)

    X0, U0 = get_initial_state(solver.obj,N)
    solve_dircol(solver, X0, U0, method=method, grads=grads, start=start)
end


function convertInf!(A::VecOrMat{Float64},infbnd=1.1e20)
    infs = isinf.(A)
    A[infs] = sign.(A[infs])*infbnd
    return nothing
end


function convert_N(N::Int,method::Symbol)::Int
    nSeg = N-1
    if method == :hermite_simpson_separated
        N = 2nSeg + 1
    else
        N = nSeg + 1
    end
    return N
end

function get_weights(method::Symbol,N::Int)
    if method == :trapezoid
        weights = ones(N)
        weights[[1,end]] .= 0.5
    elseif method == :hermite_simpson_separated ||
            method == :hermite_simpson
        weights = ones(N)*2/6
        weights[2:2:end] .= 4/6
        weights[[1,end]] .= 1/6
    elseif method == :midpoint
        weights = ones(N)
        weights[end] = 0
    end
    return weights
end

function gen_custom_constraint_fun(solver::Solver,method)
    N, = get_N(solver,method)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    NN = (n+m̄)N
    obj = solver.obj
    pI_obj, pE_obj = count_constraints(solver.obj)
    pI_c,   pE_c   = pI_obj[2], pE_obj[2]  # Number of custom stage constraints
    pI_N_c, pE_N_c = pI_obj[4], pE_obj[4]  # Number of custom terminal constraints
    pC = pE_c + pI_c        # Number of custom stage constraints (total)
    pC_N = pE_N_c + pI_N_c      # Number of custom terminal constraints (total)
    P = (N-1)pC + pC_N  # Total number of custom constraints
    PI = (N-1)pI_c + pI_N_c  # Total number of inequality constraints
    PE = (N-1)pE_c + pE_N_c  # Total number of equality constraints
    P = PI+PE


    function c_fun!(C_vec,X,U)
        CE = view(C_vec,1:PE)
        CI = view(C_vec,PE.+(1:PI))
        cE = reshape(view(CE,1:(N-1)pE_c),pE_c,N-1)
        cI = reshape(view(CI,1:(N-1)pI_c),pI_c,N-1)

        # Equality Constraints
        if pE_c > 0
            for k = 1:N-1
                obj.cE(view(cE,1:pE_c,k),X[:,k],U[:,k])
            end
        end
        if pE_N_c > 0
            obj.cE(view(CE,(N-1)pE_c+1:PE),X[:,N])
        end

        # Inequality Constraints
        if pI_c > 0
            for k = 1:N-1
                obj.cI(view(cI,1:pI_c,k),X[:,k],U[:,k])
            end
        end
        if pI_N_c > 0
            obj.cI(view(CI,(N-1)pI_c+1:PI),X[:,N])
        end
        return nothing
    end

    # Jacobian
    jac_cI = generate_general_constraint_jacobian(obj.cI, pI_c, n, m)
    jac_cE = generate_general_constraint_jacobian(obj.cE, pE_c, n, m)

    function jacobian_block!(vals, jac::Function, x, u, p::Int)
        blk = reshape(vals,p,n+m̄)
        Ac = view(blk,1:p,1:n)
        Bc = view(blk,1:p,n.+(1:m))
        jac(Ac,Bc,x,u)
    end

    function jac_c!(vals,X::Matrix{Float64},U::Matrix{Float64})
        # Equality Constraints
        n_blk = (n+m̄)pE_c
        nG_PE = (N-1)n_blk  # number of nonzero entries in custom equality constraint jacobian
        for k = 1:N-1
            off = (k-1)*n_blk
            block = view(vals,off.+(1:n_blk))
            jacobian_block!(block, jac_cE, X[:,k], U[1:m,k], pE_c)
        end
        if pE_N_c > 0
            n_blk = n*pE_N_c
            block = view(vals,nG_PE.+(1:n_blk))
            nG_PE += n_blk
            Ac = reshape(block,pE_N_c,n)
            jac_cE(Ac,X[:,N])
        end

        # Inequality Constraints
        n_blk = (n+m̄)pI_c
        nG_PI = (N-1)n_blk  # number of nonzero entries in custom equality constraint jacobian
        for k = 1:N-1
            off = (k-1)*n_blk + nG_PE
            block = view(vals,off.+(1:n_blk))
            jacobian_block!(block, jac_cI, X[:,k], U[1:m,k], pI_c)
        end
        if pI_N_c > 0
            n_blk = n*pI_N_c
            block = view(vals,(nG_PE+nG_PI).+(1:n_blk))
            Ac = reshape(block,pE_N_c,n)
            jac_cE(Ac,X[:,N])
        end
    end

    J = spzeros(P,NN)
    JE = view(J,1:PE,:)
    JI = view(J,PE+1:P,:)

    function jacobian_sparsity(start::Int=0)
        i = start
        # Equality Constraints
        n_blk = (n+m)pE_c
        num_block = reshape(1:n_blk,pE_c,n+m̄)
        if pE_c > 0
            for k = 1:N-1
                off_1 = (k-1)pE_c
                off_2 = (k-1)*(n+m̄)
                JE[off_1.+(1:pE_c),off_2.+(1:n+m)] = num_block .+ i
                i += n_blk
            end
        end
        if pE_N_c > 0
            n_blk = n*pE_N_c
            num_block = reshape(1:n_blk,pE_N_c,n)
            off = (N-1)*(n+m̄)
            JE[(N-1)pE_c.+1:PE,off.+(1:n)] = num_block .+ i
            i += n_blk
        end

        # Inequality Constraints
        n_blk = (n+m)pI_c
        num_block = reshape(1:n_blk,pI_c,n+m̄)
        if pI_c > 0
            for k = 1:N-1
                off_1 = (k-1)pI_c
                off_2 = (k-1)*(n+m̄)
                JI[off_1.+(1:pI_c),off_2.+(1:n+m)] = num_block .+ i
                i += n_blk
            end
        end
        if pE_N_c > 0
            n_blk = n*pI_N_c
            num_block = reshape(1:n_blk,pI_N_c,n)
            off = (N-1)*(n+m̄)
            JI[(N-1)pI_c.+1:PI,off.+(1:n)] = num_block .+ i
            i += n_blk
        end
        return J
    end

    return c_fun!, jac_c!, jacobian_sparsity
end

function get_bounds(solver::Solver,method::Symbol)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    N,N_ = get_N(solver,method)
    obj = solver.obj
    x_L = zeros((n+m̄),N)
    x_U = zeros((n+m̄),N)

    ## STATE CONSTRAINTS ##
    x_L[1:n,:] .= obj.x_min
    x_U[1:n,:] .= obj.x_max
    x_L[n.+(1:m),:] .= obj.u_min
    x_U[n.+(1:m),:] .= obj.u_max

    # Minimum time bounds
    if solver.state.minimum_time
        x_L[n+m̄,:] .= solver.opts.min_dt
        x_U[n+m̄,:] .= solver.opts.max_dt
        p_dt = N-2
    else
        p_dt = 0
    end

    # Initial Condition
    x_L[1:n,1] .= obj.x0
    x_U[1:n,1] .= obj.x0

    # Terminal Constraint
    x_L[1:n,N] .= obj.xf
    x_U[1:n,N] .= obj.xf

    ## CONSTRAINTS ##
    p_colloc = (N-1)n
    pI_obj, pE_obj = count_constraints(solver.obj)
    pI_c,   pE_c   = pI_obj[2], pE_obj[2]  # Number of custom stage constraints
    pI_N_c, pE_N_c = pI_obj[4], pE_obj[4]  # Number of custom terminal constraints
    PE_c = (N-1)pE_c + pE_N_c  # Total number of custom equality constraints
    PI_c = (N-1)pI_c + pI_N_c  # Total number of custom inequality constraints
    PE = PE_c + p_colloc + p_dt  # Total equality constraints
    PI = PI_c                    # Total inequality constraints
    P = PI+PE                    # Total constraints

    # Get bounds
    g_L = zeros(P)
    g_U = zeros(P)
    g_L[1:PE] .= 0
    g_U[1:PE] .= 0
    g_L[PE+1:end] .= -Inf
    g_U[PE+1:end] .= 0

    # Convert Infinite bounds
    convertInf!(x_L)
    convertInf!(x_U)
    convertInf!(g_L)
    convertInf!(g_U)
    return vec(x_L), vec(x_U), g_L, g_U
end


"""
Stack state and controls for all time steps into a single vector of variables
Z = [X1,U1,X2,U2,..,] (all transposed)
"""
function packZ(X,U)
    n, N = size(X)
    m = size(U,1)
    if size(U,2) == N-1
        U = [U U[:,end]]
    end
    Z = zeros(n+m,N)
    Z[1:n,:] .= X
    Z[n+1:end,1:end] .= U
    Z = vec(Z)
end

function unpackZ(Z, sze)
    n,m,N = sze
    Z = reshape(Z,n+m,N)
    X = Z[1:n,:]
    U = Z[n+1:end,:]
    return X,U
end

function get_initial_state(obj::Objective, N::Int)
    n,m = get_sizes(obj)
    X0 = line_trajectory(obj.x0, obj.xf, N)
    U0 = zeros(m,N)
    return X0, U0
end




"""
Evaluate Objective Value
"""
function cost(solver::Solver,res::DircolResults)
    cost(solver,res.X_,res.U_,res.weights)
end

function cost(solver::Solver,X::AbstractMatrix,U::AbstractMatrix,method::Symbol)
    # pull out solver/objective
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    obj = solver.obj
    Q = obj.cost.Q; R = obj.cost.R; xf::Vector{Float64} = obj.xf; Qf::Matrix{Float64} = obj.cost.Qf
    dt = solver.dt

    J = 0.0
    if method == :hermite_simpson || method == :hermite_simpson_separated
        nSeg = (N_-1)÷2
        Xk = view(X,:,1:2:N_)
        Uk = view(U,:,1:2:N_)
        Xm = view(X,:,2:2:N_-1)
        Um = view(U,:,2:2:N_-1)
        solver.state.minimum_time ? dt = view(U,m̄,1:2:N_) : dt = ones(N)*solver.dt

        for k = 1:nSeg
             # Simpson quadrature (integral approximation) for foh stage cost
            J += dt[k]/6*(stage_cost(obj.cost,Xk[:,k],Uk[1:m,k]) + 4stage_cost(obj.cost,Xm[:,k],Um[1:m,k]) + stage_cost(obj.cost,Xk[:,k+1],Uk[1:m,k+1]))
            solver.state.minimum_time ? J += solver.opts.R_minimum_time*dt[k] : nothing
        end
    elseif method == :midpoint
        solver.state.minimum_time ? dt = view(U,m̄,1:2:N_) : dt = ones(N)*solver.dt
        for k = 1:N-1
            J += dt[k]*stage_cost(obj.cost,X[:,k],U[1:m,k])
        end
    elseif method == :trapezoid
        solver.state.minimum_time ? dt = view(U,m̄,1:2:N_) : dt = ones(N)*solver.dt
        for k = 1:N-1
            J += dt[k]*(stage_cost(obj.cost,X[:,k],U[1:m,k]) + stage_cost(obj.cost,X[:,k+1],U[1:m,k+1]))/2
        end
    end


    J += stage_cost(obj.cost,X[:,N_])

    return J
end

function cost(solver::Solver,X::AbstractArray,U::AbstractArray,weights::Vector{Float64})
    n,m = get_sizes(solver)
    N_ = size(X,2)
    m̄, = get_num_controls(solver)
    Qf = solver.obj.cost.Qf; Q = solver.obj.cost.Q;
    xf = solver.obj.xf; R = solver.obj.cost.R;
    solver.state.minimum_time ? dt = U[m̄,:] : dt = ones(N_)*solver.dt

    J = zeros(eltype(X),N_)
    for k = 1:N_
        J[k] = stage_cost(solver.obj.cost,X[:,k],U[1:m,k])*dt[k]
    end

    J = weights'J
    J += 0.5*(X[:,N_] - xf)'*Qf*(X[:,N_] - xf)
end


"""
Gradient of Objective
"""
function cost_gradient!(solver::Solver, X, U, fVal, A, B, weights, vals, method::Symbol)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    N,N_ = get_N(solver,method)
    dt = solver.dt

    obj = solver.obj
    xf = obj.xf
    Q = obj.cost.Q; R = obj.cost.R; Qf = obj.cost.Qf;
    # X,U = res.X_, res.U_
    grad_f = reshape(vals, n+m̄, N)

    if method == :hermite_simpson
        I_n = Matrix(I,n,n)

        Xk = view(X,:,1:2:N_)
        Uk = view(U,:,1:2:N_)
        Xm = view(X,:,2:2:N_-1)
        Um = view(U,:,2:2:N_-1)
        Ak = view(A,:,:,1:2:N_)
        Bk = view(B,:,:,1:2:N_)
        fk = fVal

        if solver.state.minimum_time
            dt = Uk[m̄,:]
            c_dt = solver.opts.R_minimum_time
        else
            dt = ones(N)*solver.dt
        end

        grad_f[1:n,1] =     (Q*(Xk[:,1]-xf) + 4*(I_n/2 + dt[1]/8*Ak[:,:,1])'Q*(Xm[:,1] - xf))*dt[1]/6
        grad_f[n.+(1:m),1] = (R*Uk[1:m,1] + 4(dt[1]/8*Bk[:,:,1]'Q*(Xm[:,1] - xf) + R*Um[1:m,1]/2))*dt[1]/6
        solver.state.minimum_time ? grad_f[n+m̄,1] = (stage_cost(obj.cost,Xk[:,1],Uk[1:m,1]) + 4stage_cost(obj.cost,Xm[:,1],Um[1:m,1]) + stage_cost(obj.cost,Xk[:,2],Uk[1:m,2]))/6 +
                                                 (fk[:,1] - fk[:,2])'*Q*(Xm[:,1] - xf)*dt[1]/12 + c_dt : nothing
        for k = 2:N-1
            grad_f[1:n,k] = (Q*(Xk[:,k]-xf) + 4(I_n/2 + dt[k]/8*Ak[:,:,k])'Q*(Xm[:,k] - xf))*dt[k]/6 +
                            (Q*(Xk[:,k]-xf) + 4(I_n/2 - dt[k]/8*Ak[:,:,k])'Q*(Xm[:,k-1] - xf))*dt[k-1]/6
            grad_f[n.+(1:m),k] = (R*Uk[1:m,k] + 4(dt[k]/8*Bk[:,:,k]'Q*(Xm[:,k] - xf)   + R*Um[1:m,k]/2))*dt[k]/6 +
                                 (R*Uk[1:m,k] - 4(dt[k]/8*Bk[:,:,k]'Q*(Xm[:,k-1] - xf) - R*Um[1:m,k-1]/2))*dt[k-1]/6
            solver.state.minimum_time ? grad_f[n+m̄,k] = (stage_cost(obj.cost,Xk[:,k],Uk[1:m,k]) + 4stage_cost(obj.cost,Xm[:,k],Um[1:m,k]) + stage_cost(obj.cost,Xk[:,k+1],Uk[1:m,k+1]))/6 +
                                                     (fk[:,k] - fk[:,k+1])'*Q*(Xm[:,k] - xf)*dt[k]/12 + c_dt : nothing
        end
        grad_f[1:n,N] = (Q*(Xk[:,N]-xf) + 4(I_n/2 - dt[N-1]/8*Ak[:,:,N])'Q*(Xm[:,N-1] - xf))*dt[N-1]/6
        grad_f[n.+(1:m),N] = (R*Uk[1:m,N] - 4(dt[N-1]/8*Bk[:,:,N]'Q*(Xm[:,N-1] - xf) - R*Um[1:m,N-1]/2))*dt[N-1]/6
        solver.state.minimum_time ? grad_f[n+m̄,N] = 0 : nothing
        grad_f[1:n,N] += Qf*(Xk[:,N] - xf)
    elseif method == :midpoint
        I_n = Matrix(I,n,n)
        Xm = X

        # Get dt
        solver.state.minimum_time ? dt = U[m̄,:] : dt = ones(N)*solver.dt

        grad_f[1:n,1] = Q*(Xm[:,1] - xf)*dt[1]/2
        grad_f[n.+(1:m),1] = R*U[1:m,1]*dt[1]
        solver.state.minimum_time ? grad_f[n+m̄,1] = stage_cost(obj.cost,Xm[:,1],U[1:m,1]) : nothing
        for k = 2:N-1
            grad_f[1:n,k] = Q*(Xm[:,k] - xf)*dt[k]/2 + Q*(Xm[:,k-1] - xf)*dt[k-1]/2
            grad_f[n.+(1:m),k] = R*U[1:m,k]*dt[k]
            solver.state.minimum_time ? grad_f[n+m̄,k] = stage_cost(obj.cost,Xm[:,k],U[1:m,k]) : nothing
        end
        grad_f[1:n,N] = Q*(Xm[:,N-1] - xf)*dt[N-1]/2
        grad_f[n+1:end,N] = zeros(m̄)
        grad_f[1:n,N] += Qf*(X[:,N] - xf)
    else
        for k = 1:N
            grad_f[1:n,k] = weights[k]*Q*(X[:,k] - xf)*dt
            grad_f[n+1:end,k] = weights[k]*R*U[:,k]*dt
        end
        grad_f[1:n,N] += Qf*(X[:,N] - xf)
    end
    return nothing
end

"""
$(SIGNATURES)
Return placeholders for trajectories
"""
function init_traj_points(solver::Solver,fVal::Matrix,method::Symbol)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    X_,U_,fVal_ = zeros(n,N_),zeros(m̄,N_),zeros(n,N_)
    if method == :trapezoid || method == :hermite_simpson_separated
        fVal_ = fVal
    end
    return X_,U_,fVal_
end


"""
$(SIGNATURES)
Return all the trajectory points used to evaluate integrals
"""
function get_traj_points!(solver::Solver,X,U,X_,U_,method::Symbol)
    fVal = zeros(X)
    fVal_ = zeros(X_)
    update_derivatives!(solver,fVal,X,U)
    get_traj_points!(solver,X,U,X_,U_,fVal,fVal_)
end

function get_traj_points!(solver::Solver,X,U,X_,U_,fVal,method::Symbol)
    n,m,N = get_sizes(solver)
    m̄,N_ = size(U_)
    dt = solver.dt
    if method == :hermite_simpson
        X_[:,1:2:end] = X
        U_[:,1:2:end] = U
        # fVal_[:,1:2:end] = fVal

        # Midpoints
        Xm = view(X_,:,2:2:N_-1)
        Um = view(U_,1:m,2:2:N_-1)
        Um .= (U[1:m,1:end-1] + U[1:m,2:end])/2
        # fValm = view(fVal_,:,2:2:N_-1)
        for k = 1:N-1
            solver.state.minimum_time ? dt = U[m̄,k] : nothing
            x1,x2 = X[:,k], X[:,k+1]
            Xm[:,k] = (x1+x2)/2 + dt/8*(fVal[:,k]-fVal[:,k+1])
            # solver.fc(view(fValm,:,k),Xm[:,k],Um[:,k])
        end
        # if solver.state.minimum_time
        #     U_[m̄,2:2:N_-1] = U[m̄,1:N-1]
        #     U_[m̄,3:2:N_-2] = (U[m̄,1:N-2] + U[m̄,2:N-1])/2
        #     U_[m̄,N_] = U[m̄,N-1]
        # end
    elseif method == :midpoint
        Xm = (X[:,1:end-1] + X[:,2:end])/2
        X_[:,1:end-1] = Xm
        X_[:,end] = X[:,end]
        # res.U_ is set in the results constructor
        # for k = 1:N-1
        #     solver.fc(view(fVal_,:,k),res.X_[:,k],U_[:,k])
        # end
    # else
    #     X_ .= X
    #     U_ .= U
    end
    return nothing
end

function get_traj_points(solver,X,U,gfVal,gX_,gU_,method::Symbol,cost_only::Bool=false)
    if !cost_only
        update_derivatives!(solver,X,U,gfVal)
    end
    if method == :trapezoid || method == :hermite_simpson_separated
        X_,U_ = X,U
    else
        if cost_only && method == :hermite_simpson
            update_derivatives!(solver,X,U,gfVal)
        end
        get_traj_points!(solver,X,U,gX_,gU_,gfVal,method)
        X_,U_ = gX_,gU_
        if method == :midpoint
            U_ = U
        end
    end
    return X_, U_
end

function get_traj_points_derivatives!(solver::Solver,X_,U_,fVal_,fVal,method::Symbol)
    if method == :hermite_simpson
        N_ = size(X_,2)
        fVal_[:,1:2:end] = fVal
        Xm,Um,fm = X_[:,2:2:end-1],U_[:,2:2:end-1],view(fVal_,:,2:2:N_-1)
        update_derivatives!(solver,Xm,Um,fm)
    elseif method == :midpoint
        update_derivatives!(solver,X_,U_,fVal_)
    else
        fVal_ .= fVal
    end
end

function update_derivatives!(solver::Solver,X::AbstractArray,U::AbstractArray,fVal::AbstractArray)
    n,m = get_sizes(solver)
    N = size(X,2)
    for k = 1:N
        solver.model.f(view(fVal,:,k),X[:,k],U[1:m,k])
    end
end

function update_jacobians!(solver::Solver,X,U,A,B,method::Symbol,cost_only::Bool=false)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver,method)
    if ~cost_only || method == :hermite_simpson
        if method == :hermite_simpson || method == :midpoint
            inds = cost_only ? (1:2:N_) : (1:N_)  # HS only needs jacobians at knot points for cost gradient
            for k = inds
                A[:,:,k], B[:,:,k] = solver.Fc(X[:,k],U[1:m,k])
            end
        else
            # Z = packZ(X,U)
            # z = reshape(Z,n+m,N)
            for k = 1:N_
                tA,tB = solver.Fc(X[:,k],U[:,k])
                A[:,:,k] = [tA tB]
            end
        end
    end
end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                              #
#                           CONSTRAINTS                                        #
#                                                                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

"""
Evaluate constraint values
[ dt*(f(x1)+f(x2))/2 - x2 + x1     ] Collocation Constraints
[ dt*(f(x2)+f(x3))/2 - x3 + x2     ]
[           ...                 ]
[ dt*(f(xN-1)+f(xN))/2 - xN + xN-1 ]
"""
function collocation_constraints!(solver::Solver, X, U, fVal, g_colloc, method::Symbol)
    # X,U need to be the "trajectory points", or X_,U_
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    g = reshape(g_colloc,n,N-1)
    dt = solver.dt

    if method == :trapezoid
        # solver.state.minimum_time ? dt = U[m̄,:] : dt = ones(N)*solver.dt
        for k = 1:N-1
            # Collocation Constraints
            g[:,k] = dt*( fVal[:,k+1] + fVal[:,k] )/2 - X[:,k+1] + X[:,k]
        end
    elseif method == :hermite_simpson_separated || method == :hermite_simpson
        iLow = 1:2:N_-1
        iMid = iLow .+ 1
        iUpp = iMid .+ 1
        solver.state.minimum_time ? dt = U[m̄:m̄,:] : dt = ones(1,N_)*solver.dt

        collocation = - X[:,iUpp] + X[:,iLow] + dt[:,iLow].*(fVal[:,iLow] + 4*fVal[:,iMid] + fVal[:,iUpp])/6

        if method == :hermite_simpson
            g .= collocation
        else
            midpoints =  - X[:,iMid] + (X[:,iLow]+X[:,iUpp])/2 + dt[:,iLow].*(fVal[:,iLow] - fVal[:,iUpp])/8
            g[:,iLow] = collocation
            g[:,iMid] = midpoints
        end
        # Xk = view(X,:,1:2:N_)
        # Uk = view(U,:,1:2:N_)
        # Xm = view(X,:,2:2:N_-1)
        # Um = view(U,:,2:2:N_-1)
        # Fk = view(fVal,:,1:2:N_)
        # Fm = view(fVal,:,2:2:N_-1)
        #
        # for k = 1:N-1
        #     x1,x2 = Xk[:,k],Xk[:,k+1]
        #     u1,u2 = Uk[1:m,k],Uk[1:m,k+1]
        #     solver.state.minimum_time ? dt = Uk[m̄,k] : dt = solver.dt
        #     f1 = Fk[:,k]
        #     f2 = Fk[:,k+1]
        #     fm = Fm[:,k]
        #     g[:,k] = dt/6*(f1+4fm+f2) - x2 + x1
        # end

    elseif method == :midpoint
        Xm = X
        # Calculate the knot points from the midpoints (and the terminal point)
        Xk = zero(Xm)
        Xk[:,end] = Xm[:,end]
        solver.state.minimum_time ? dt = U[m̄,:] : dt = ones(N_)*solver.dt
        for k = N-1:-1:1
            Xk[:,k] = 2Xm[:,k] - Xk[:,k+1]
        end
        for k = 1:N-1
            g[:,k] = dt[k]*fVal[:,k] - Xk[:,k+1] + Xk[:,k]
        end
    end
end

function dt_constraints!(solver, g_dt, dt)
    N = length(dt)
    for k = 1:N-2
        g_dt[k] = dt[k] - dt[k+1]
    end
end


"""
Constraint Jacobian
"""
function collocation_constraint_jacobian!(solver::Solver, X, U, fVal, A::Array{Float64,3}, B::Array{Float64,3}, vals, method::Symbol)
    # X and U are X_, U_ trajectory points
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    dt = solver.dt
    Inm = Matrix(I,n,n+m)

    if method == :trapezoid
        Z = packZ(X,U)
        z = reshape(Z,n+m̄,N)
        n_blk = 2(n+m̄)n
        solver.state.minimum_time ? dt = U[m̄,:] : dt = ones(N_)*solver.dt

        function calc_block_trap(k,blk::SubArray{Float64})
            blk = reshape(blk,n,2(n+m̄))
            blk[:,1:n+m] =         dt[k]*A[:,:,k  ]/2+Inm
            blk[:,n.+m̄.+(1:n+m)] = dt[k]*A[:,:,k+1]/2-Inm
            if solver.state.minimum_time
                blk[:,n+m̄] = (fVal[:,k] + fVal[:,k+1])/2
            end
            return nothing
        end

        for k = 1:N-1
            off = (k-1)*n_blk
            block = view(vals,off.+(1:n_blk))
            calc_block_trap(k,block)
        end

    elseif method == :hermite_simpson_separated
        nSeg = Int((N-1)/2)
        Z = packZ(X,U)
        z = reshape(Z,n+m̄,N)

        n_blk = 6(n+m)n
        function calc_block_sep!(k,val_block)
            blk = reshape(val_block,2n,3(n+m))
            fz = A[:,:,k]  # F(z[:,k])
            blk[   (1:n),(1:n+m)] = dt*fz/6 + Inm
            blk[n.+(1:n),(1:n+m)] = dt*fz/8 + Inm/2
            fm = A[:,:,k+1]  # F(z[:,k+1])
            blk[   (1:n),n.+m.+(1:n+m)] = 2*dt*fm/3
            blk[n.+(1:n),n.+m.+(1:n+m)] = -Inm
            fz1 = A[:,:,k+2]  # F(z[:,k+2])
            blk[   (1:n),2(n+m).+(1:n+m)] =  dt*fz1/6 - Inm
            blk[n.+(1:n),2(n+m).+(1:n+m)] = -dt*fz1/8 + Inm/2
            return nothing
        end

        for i = 1:nSeg
            k = 2i-1
            off = (i-1)*n_blk
            block = view(vals,off.+(1:n_blk))
            calc_block_sep!(k,block)
        end
    elseif method == :hermite_simpson
        nSeg = N-1
        n_blk = 2(n+m̄)n

        Xk = view(X,:,1:2:N_)
        Uk = view(U,:,1:2:N_)
        Xm = view(X,:,2:2:N_-1)
        Um = view(U,:,2:2:N_-1)
        Ak = view(A,:,:,1:2:N_)
        Bk = view(B,:,:,1:2:N_)
        AM = view(A,:,:,2:2:N_-1)
        BM = view(B,:,:,2:2:N_-1)
        fk = view(fVal,:,1:2:N_)
        fm = view(fVal,:,2:2:N_-1)

        solver.state.minimum_time ? dt = Uk[m̄,:] : dt = ones(N_)*solver.dt

        function calc_block_hs!(k::Int,vals::SubArray)
            x1,u1 = Xk[:,k],Uk[:,k]
            x2,u2 = Xk[:,k+1],Uk[:,k+1]
            A1,A2 = Ak[:,:,k],Ak[:,:,k+1]
            B1,B2 = Bk[:,:,k],Bk[:,:,k+1]
            xm = Xm[:,k] #(x1+x2)/2 + dt/8*(fVal[:,k]-fVal[:,k+1])
            um = Um[:,k] # (u1+u2)/2
            Am,Bm = AM[:,:,k],BM[:,:,k]
            In = Matrix(I,n,n)
            Im = Matrix(I,m,m)

            vals = reshape(vals,n,2(n+m̄))
            vals[:,1:n] =          dt[k]/6*(A1 + 4Am*( dt[k]/8*A1 + In/2)) + In    # ∇x1
            vals[:,n.+(1:m)] =     dt[k]/6*(B1 + 4Am*( dt[k]/8*B1) + 4Bm*(Im/2))   # ∇u1
            vals[:,n.+m̄.+(1:n)] =  dt[k]/6*(A2 + 4Am*(-dt[k]/8*A2 + In/2)) - In    # ∇x2
            vals[:,2n.+m̄.+(1:m)] = dt[k]/6*(B2 + 4Am*(-dt[k]/8*B2) + 4Bm*(Im/2))   # ∇u2
            if solver.state.minimum_time
                vals[:,n+m̄] = (fk[:,k] + 4fm[:,k] + fk[:,k+1])/6 + dt[k]/12*Am*(fk[:,k] - fk[:,k+1])
            end
            return nothing
        end

        for k = 1:nSeg
            off = (k-1)*n_blk
            block = view(vals,off.+(1:n_blk))
            calc_block_hs!(k,block)
        end

    elseif method == :midpoint
        nSeg = N-1
        n_blk = (2n+m)n
        In = Matrix(I,n,n)

        function calc_jacob_block_midpoint!(k,blk)
            blk = reshape(blk,n,2n+m)
            blk[:,1:n] =        In + dt*A[:,:,k]/2    # ∇x1
            blk[:,n.+(1:m)] =    dt*B[:,:,k]           # ∇u1
            blk[:,n.+m.+(1:n)] = -In + dt*A[:,:,k]/2    # ∇x2
            return nothing
        end

        for k = 1:nSeg
            off = (k-1)*n_blk
            block = view(vals,off.+(1:n_blk))
            calc_jacob_block_midpoint!(k,block)
        end
    end
    return nothing
end

function time_step_constraint_jacobian!(vals, solver::Solver)
    if solver.state.minimum_time
        vals[1:2:2(N-2)] .= 1
        vals[2:2:2(N-2)+1] .= -1
    end
end


function collocation_constraint_jacobian_sparsity(solver::Solver, method::Symbol, start::Int=0)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    jacob_g = spzeros(Int,(N-1)*n,N*(n+m̄))
    dt = solver.dt
    i = start

    if method == :trapezoid || method == :hermite_simpson

        n_blk = 2(n+m̄)n
        num_block = reshape(1:n_blk,n,2(n+m̄))

        for k = 1:N-1
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m̄)
            jacob_g[off_1.+(1:n), off_2.+(1:2(n+m̄))] = num_block .+ i
            i += n_blk
        end

    elseif method == :hermite_simpson_separated
        nSeg = Int((N-1)/2)
        n_blk = 6(n+m̄)n
        num_block = reshape(1:n_blk,2n,3(n+m̄))

        for k = 1:nSeg
            off_1 = 2(k-1)*(n)
            off_2 = 2(k-1)*(n+m̄)

            jacob_g[off_1.+(1:2n), off_2.+(1:3(n+m̄))] = num_block .+ i
            i += n_blk
        end

    elseif method == :midpoint
        nSeg = N-1
        n_blk = (2n+m̄)n
        num_block = reshape(1:n_blk,n,2n+m̄)

        for k = 1:nSeg
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m̄)
            jacob_g[off_1.+(1:n), off_2.+(1:2n+m̄)] = num_block .+ i
            i += n_blk
        end
    end


    # rows,cols,inds = findnz(jacob_g)
    # v = sortperm(inds)
    # rows = rows[v]
    # cols = cols[v]

    return jacob_g
end

function time_step_constraint_jacobian_sparsity(solver::Solver, start::Int=0)
    n,m,N = get_sizes(solver)
    m̄, = get_num_controls(solver)
    NN = (n+m̄)N
    if solver.state.minimum_time
        jacob_dt = spzeros(Int,N-2,NN)
        dts = n+m̄:n+m̄:NN-2(n+m̄)
        jacob_dt[CartesianIndex.(1:N-2,dts)] = (1:2:2(N-2)) .+ start
        jacob_dt[CartesianIndex.(1:N-2,dts.+n.+m̄)] = (2:2:2(N-2)+1) .+ start
        return jacob_dt
    else
        return sparse([],[],Float64[],0,NN)
    end
end



# JUNK FUNCTIONS

function rollout_midpoint(solver::Solver,U::Matrix)
    N = solver.N
    N = convert_N(N,method)
    n,m = solver.model.n,solver.model.m
    nSeg = N-1
    N_ = 2*nSeg + 1
    X_ = zeros(n,N_)
    U_ = zeros(size(U,1),N_)
    X_[:,1] = solver.obj.x0
    U_[:,1] = U[:,1]

    for k = 1:N_-1
        if isodd(k)
            j = (k+1)÷2
            U_[:,k+2] = U[:,j+1]
            solver.fd(view(X_,:,k+2), X_[:,k], U_[1:m,k], U_[1:m,k+2])
        else
            Ac1, Bc1 = solver.Fc(X_[:,k-1],U_[:,k-1])
            Ac2, Bc2 = solver.Fc(X_[:,k+1],U_[:,k+1])
            M = [(0.5*eye(n) + dt/8*Ac1) (dt/8*Bc1) (0.5*eye(n) - dt/8*Ac2) (-dt/8*Bc2)]

            Xm = M*[X_[:,k-1]; U_[:,k-1]; X_[:,k+1]; U_[:,k+1]]
            Um = (U_[:,k-1] + U_[:,k+1])/2

            X_[:,k] = Xm
            U_[:,k] = Um
        end
    end
    return X_,U_
end

function calc_midpoints(X::Matrix, U::Matrix, solver::Solver)
    N = solver.N
    N = convert_N(N,method)
    n,m = solver.model.n,solver.model.m
    nSeg = N-1
    N_ = 2*nSeg + 1
    X_ = zeros(n,N_)
    U_ = zeros(size(U,1),N_)
    X_[:,1:2:end] = X
    U_[:,1:2:end] = U

    f1 = zeros(n)
    f2 = zeros(n)
    for k = 2:2:N_

        f(f1,X_[:,k-1], U_[:,k-1])
        f(f2,X_[:,k+1], U_[:,k+1])
        x1 = X_[:,k-1]
        x2 = X_[:,k+1]
        Xm = (x1+x2)/2 + dt/8*(f1-f2)

        Um = (U_[:,k-1] + U_[:,k+1])/2

        X_[:,k] = Xm
        U_[:,k] = Um
    end
    return X_,U_
end

function interp(t,T,X,U,F)
    k = findlast(t .> T)
    constraint_decrease_ratio = t-T[k]
    if method == :trapezoid
        u = U[:,k] + constraint_decrease_ratio/dt*(F[:,k+1]-F[:,k])
        x = X[:,k] + F[:,k]*constraint_decrease_ratio + constraint_decrease_ratio^2/(2*dt)*(F[:,k+1]-F[:,k])
    elseif method == :hermite_simpson || method == :hermite_simpson_separated
        x1,x2 = X[:,k], X[:,k+1]
        u1,u2 = U[:,k], U[:,k+1]
        f1,f2 = F[:,k], F[:,k+1]
        xm = (x1+x2)/2 + dt/8*(f1-f2)
        um = (U[:,k] + U[:,k+1])/2

        u = (2(constraint_decrease_ratio-dt/2)(constraint_decrease_ratio-dt)U[:,k] - 4constraint_decrease_ratio*(constraint_decrease_ratio-dt)Um + 2constraint_decrease_ratio*(constraint_decrease_ratio-dt/2)*U[:,k+1])/dt^2
        x = X[:,k] + F[:,k]*constraint_decrease_ratio/dt + 1/2*(-3F[:,])
    end
    return x,u
end
