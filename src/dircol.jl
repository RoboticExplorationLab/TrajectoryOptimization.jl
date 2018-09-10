


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
        nlp::Symbol=:ipopt, method::Symbol=:auto, grads::Symbol=:quadratic, start=:cold)

    if solver.obj isa UnconstrainedObjective
        solver = Solver(solver,obj=ConstrainedObjective(solver.obj))
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
    dt = solver.dt
    N = convert_N(N,method)

    if N != size(X0,2)
        solver.opts.verbose ? println("Interpolating initial guess") : nothing
        X0,U0 = interp_traj(N,obj.tf,X0,U0)
    end

    if nlp == :ipopt
        return solve_ipopt(solver,X0,U0,method)
    elseif nlp == :snopt
        return solve_snopt(solver,X0,U0, method=method, grads=grads, start=start)
    else
        ArgumentError("$nlp is not a known NLP solver. Options are (:ipopt, :snopt)")
    end
    # return sol, stats, prob
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
        weights[[1,end]] = 0.5
    elseif method == :hermite_simpson_separated ||
            method == :hermite_simpson
        weights = ones(N)*2/6
        weights[2:2:end] = 4/6
        weights[[1,end]] = 1/6
    elseif method == :midpoint
        weights = ones(N)
        weights[end] = 0
    end
    return weights
end

function gen_custom_constraint_fun(solver::Solver,method)
    N, = get_N(solver,method)
    n,m = get_sizes(solver)
    NN = (n+m)N
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


    function c_fun!(C_vec::Vector,X::Matrix,U::Matrix)
        CE = view(C_vec,1:PE)
        CI = view(C_vec,PE+(1:PI))
        cE = reshape(view(CE,1:(N-1)pE_c),pE_c,N-1)
        cI = reshape(view(CI,1:(N-1)pI_c),pI_c,N-1)

        # Equality Constraints
        for k = 1:N-1
            cE[:,k] = obj.cE(X[:,k],U[:,k])
        end
        if pE_N_c > 0
            CE[(N-1)pE_c+1:end] = obj.cE(X[:,N])
        end

        # Inequality Constraints
        for k = 1:N-1
            cI[:,k] = obj.cI(X[:,k],U[:,k])
        end
        if pI_N > 0
            CI[(N-1)pI_c+1:end] = obj.cI(X[:,N])
        end
        return nothing
    end

    # Jacobian
    jac_cI = generate_general_constraint_jacobian(obj.cI,pI_c,pI_N_c,n,m)
    jac_cE = generate_general_constraint_jacobian(obj.cE,pE_c,pE_N_c,n,m)
    J = spzeros(P,NN)
    JE = view(J,1:PE,:)
    JI = view(J,PE+1:P,:)

    function jac_c(X,U)
        # Equality Constraints
        for k = 1:N-1
            off_1 = (k-1)pE_c
            off_2 = (k-1)*(n+m)
            Ac,Bc = jac_cE(X[:,k],U[:,k])
            JE[off_1+(1:pE_c),off_2 + (1:n)] = Ac
            JE[off_1+(1:pE_c),off_2+n+(1:m)] = Bc
        end
        if pE_N_c > 0
            off = (N-1)*(n+m)
            Ac = jac_cE(X[:,N])
            JE[(N-1)pE_c+1:end,off+(1:n)] = Ac
        end

        # Inequality Constraints
        for k = 1:N-1
            off_1 = (k-1)pI_c
            off_2 = (k-1)*(n+m)
            Ac,Bc = jac_cI(X[:,k],U[:,k])
            JI[off_1+(1:pI_c),off_2 + (1:n)] = Ac
            JI[off_1+(1:pI_c),off_2+n+(1:m)] = Bc
        end
        if pI_N_c > 0
            off = (N-1)*(n+m)
            Ac = jac_cI(X[:,N])
            JI[pI_c+1:end,off+(1:n)] = Ac
        end
        return J
    end

    return c_fun!, jac_c
end

function get_bounds(solver::Solver,method::Symbol)
    n,m,N = get_sizes(solver)
    N,N_ = get_N(N,method)
    obj = solver.obj
    x_L = zeros((n+m),N)
    x_U = zeros((n+m),N)

    ## STATE CONSTRAINTS ##
    x_L[1:n,:] .= obj.x_min
    x_U[1:n,:] .= obj.x_max
    x_L[n+(1:m),:] .= obj.u_min
    x_U[n+(1:m),:] .= obj.u_max

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
    PE = PE_c + p_colloc  # Total equality constraints
    PI = PI_c             # Total inequality constraints
    P = PI+PE             # Total constraints

    # Get bounds
    g_L = zeros(P)
    g_U = zeros(P)
    g_L[1:PE] = 0
    g_U[1:PE] = 0
    g_L[PE+1:end] = Inf
    g_U[PE+1:end] = 0

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
    n = size(obj.Q,1); m = size(obj.R,1)
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

function cost(solver::Solver,X,U,weights)
    n,m,N = get_sizes(X,U)
    Qf = solver.obj.Qf; Q = solver.obj.Q;
    xf = solver.obj.xf; R = solver.obj.R;
    J = zeros(eltype(X),N)
    for k = 1:N
        J[k] = stage_cost(X[:,k],U[:,k],Q,R,xf)
    end

    J = weights'J*solver.dt
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
end

function cost(solver::Solver,X::Matrix,U::Matrix,weights::Vector,method::Symbol)
    obj = solver.obj
    f = solver.fc
    Q = obj.Q; xf = obj.xf; Qf = obj.Qf; R = obj.R;
    n,m,N = get_sizes(X,U)
    dt = solver.dt

    if method == :hermite_simpson
        # pull out solver/objective values
        J = zeros(eltype(X),N-1)
        f1 = zeros(eltype(X),n)
        f2 = zeros(eltype(X),n)
        for k = 1:N-1
            f(f1,X[:,k], U[:,k])
            f(f2,X[:,k+1], U[:,k+1])
            x1 = X[:,k]
            x2 = X[:,k+1]

            Xm = (x1+x2)/2 + dt/8*(f1-f2)
            Um = (U[:,k] + U[:,k+1])/2

            J[k] = dt/6*(stage_cost(X[:,k],U[:,k],Q,R,xf) + 4*stage_cost(Xm,Um,Q,R,xf) + stage_cost(X[:,k+1],U[:,k+1],Q,R,xf)) # rk3 foh stage cost (integral approximation)
        end
        J = sum(J)
        J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
        return J
    elseif method == :midpoint
        Xm = zeros(eltype(X),n,N-1)
        for k = 1:N-1
            Xm[:,k] = (X[:,k] + X[:,k+1])/2
        end
        J = zeros(eltype(Xm),N)
        for k = 1:N-1
            J[k] = stage_cost(Xm[:,k],U[:,k],Q,R,xf)
        end
        J = weights'J
        J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
        return J

    else
        J = zeros(eltype(X),N)
        for k = 1:N
            J[k] = stage_cost(X[:,k],U[:,k],Q,R,xf)
        end
        J = weights'J
        J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
        return J
    end
end

"""
Gradient of Objective
"""
function cost_gradient(solver::Solver, res::DircolResults, method::Symbol)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver,method)
    grad_f = zeros((n+m)N)
    cost_gradient!(solver,res.X_,res.U_,res.fVal_,res.A,res.B,res.weights,grad_f,method)
    return grad_f
end

function cost_gradient!(solver::Solver, vars::DircolVars, weights::Vector{Float64}, vals::Vector{Float64}, method::Symbol)::Void
    # println("simple")
    n,m = get_sizes(X,U)
    N,N_ = get_N(solver,method)
    dt = solver.dt

    obj = solver.obj
    Q = obj.Q; xf = obj.xf; R = obj.R; Qf = obj.Qf;
    # X,U = res.X_, res.U_
    grad_f = reshape(vals, n+m, N)

    for k = 1:N
        grad_f[1:n,k] = weights[k]*Q*(X[:,k] - xf)*dt
        grad_f[n+1:end,k] = weights[k]*R*U[:,k]*dt
    end
    grad_f[1:n,N] += Qf*(X[:,N] - xf)
    return nothing
end

function cost_gradient!(solver::Solver, X, U, fVal, A, B, weights, vals, method::Symbol)
    n,m = get_sizes(X,U)
    N,N_ = get_N(solver,method)
    dt = solver.dt

    obj = solver.obj
    Q = obj.Q; xf = obj.xf; R = obj.R; Qf = obj.Qf;
    # X,U = res.X_, res.U_
    grad_f = reshape(vals, n+m, N)


    if method == :hermite_simpson
        I_n = Matrix(I,n,n)

        Xk = view(X,:,1:2:N_)
        Uk = view(U,:,1:2:N_)
        Xm = view(X,:,2:2:N_-1)
        Um = view(U,:,2:2:N_-1)
        Ak = view(A,:,:,1:2:N_)
        Bk = view(B,:,:,1:2:N_)

        grad_f[1:n,1] =     Q*(Xk[:,1]-xf) + 4*(I_n/2 + dt/8*Ak[:,:,1])'Q*(Xm[:,1] - xf)
        grad_f[n+1:end,1] = R*Uk[:,1] + 4(dt/8*Bk[:,:,1]'Q*(Xm[:,1] - xf) + R*Um[:,1]/2)
        for k = 2:N-1
            grad_f[1:n,k] = Q*(Xk[:,k]-xf) + 4*(I_n/2 + dt/8*Ak[:,:,k])'Q*(Xm[:,k] - xf) +
                            Q*(Xk[:,k]-xf) + 4(I_n/2 - dt/8*Ak[:,:,k])'Q*(Xm[:,k-1] - xf)
            grad_f[n+1:end,k] = R*Uk[:,k] + 4(dt/8*Bk[:,:,k]'Q*(Xm[:,k] - xf)   + R*Um[:,k]/2) +
                                R*Uk[:,k] - 4(dt/8*Bk[:,:,k]'Q*(Xm[:,k-1] - xf) - R*Um[:,k-1]/2)
        end
        grad_f[1:n,N] = Q*(Xk[:,N]-xf) + 4(I_n/2 - dt/8*Ak[:,:,N])'Q*(Xm[:,N-1] - xf)
        grad_f[n+1:end,N] = R*Uk[:,N] - 4(dt/8*Bk[:,:,N]'Q*(Xm[:,N-1] - xf) - R*Um[:,N-1]/2)
        grad_f .*= dt/6
        grad_f[1:n,N] += Qf*(Xk[:,N] - xf)
    elseif method == :midpoint
        I_n = Matrix(I,n,n)
        Xm = X

        grad_f[1:n,1] = Q*(Xm[:,1] - xf)/2
        grad_f[n+1:end,1] = R*U[:,1]
        for k = 2:N-1
            grad_f[1:n,k] = Q*(Xm[:,k] - xf)/2 + Q*(Xm[:,k-1] - xf)/2 # weight includes dt
            grad_f[n+1:end,k] = R*U[:,k]
        end
        grad_f[1:n,N] = Q*(Xm[:,N-1] - xf)/2
        grad_f[n+1:end,N] = zeros(m)
        grad_f .*= dt
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

function init_traj_points(solver::Solver,method::Symbol)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    X_,U_,fVal_ = zeros(n,N_),zeros(m,N_),zeros(n,N_)
    return X_,U_,fVal_
end


"""
$(SIGNATURES)
Return all the trajectory points used to evaluate integrals
"""
function get_traj_points!(solver::Solver,res::DircolResults,method::Symbol)
    get_traj_points!(solver,res.X,res.U,res.X_,res.U_,res.fVal,method)
end

function get_traj_points!(solver::Solver,X,U,X_,U_,method::Symbol)
    fVal = zeros(X)
    fVal_ = zeros(X_)
    update_derivatives!(solver,fVal,X,U)
    get_traj_points!(solver,X,U,X_,U_,fVal,fVal_)
end

function get_traj_points!(solver::Solver,X,U,X_,U_,fVal,method::Symbol)
    n,N = size(X)
    m,N_ = size(U_)
    dt = solver.dt
    if method == :hermite_simpson
        X_[:,1:2:end] = X
        U_[:,1:2:end] = U
        # fVal_[:,1:2:end] = fVal

        # Midpoints
        Xm = view(X_,:,2:2:N_-1)
        Um = view(U_,:,2:2:N_-1)
        Um .= (U[:,1:end-1] + U[:,2:end])/2
        # fValm = view(fVal_,:,2:2:N_-1)
        for k = 1:N-1
            x1,x2 = X[:,k], X[:,k+1]
            Xm[:,k] = (x1+x2)/2 + dt/8*(fVal[:,k]-fVal[:,k+1])
            # solver.fc(view(fValm,:,k),Xm[:,k],Um[:,k])
        end
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
        if cost_only
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

function get_traj_points_derivatives!(solver::Solver,res::DircolResults,method::Symbol)
    get_traj_points_derivatives!(solver::Solver,res.X_,res.U_,res.fVal_,res.fVal,method::Symbol)
end

function get_traj_points_derivatives!(solver::Solver,X_,U_,fVal_,fVal,method::Symbol)
    if method == :hermite_simpson
        N_ = size(X_,2)
        fVal_[:,1:2:end] = fVal
        Xm,Um,fm = X_[:,2:2:end-1],U_[:,2:2:end-1],view(fVal_,:,2:2:N_-1)
        update_derivatives!(solver,Xm,Um,fm)
    elseif method == :midpoint
        update_derivatives!(solver,X_,U_,fVal_)
    # else
    #     fVal_ = fVal
    end
end

function update_derivatives!(solver::Solver,res::DircolResults,method::Symbol)
    # Calculate derivative
    if method != :midpoint
        update_derivatives!(solver,res.X,res.U,res.fVal)
    end
end

function update_derivatives!(solver::Solver,X::AbstractArray,U::AbstractArray,fVal::AbstractArray)
    N = size(X,2)
    for k = 1:N
        solver.fc(view(fVal,:,k),X[:,k],U[:,k])
    end
end

function update_jacobians!(solver::Solver,res::DircolResults,method::Symbol,cost_only::Bool=false)
    update_jacobians!(solver,res.X_,res.U_,res.A,res.B,method,cost_only) # TODO: pass in DircolVar
end

function update_jacobians!(solver::Solver,X,U,A,B,method::Symbol,cost_only::Bool=false)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver,method)
    if ~cost_only || method == :hermite_simpson
        if method == :hermite_simpson || method == :midpoint
            inds = cost_only ? (1:2:N_) : (1:N_)  # HS only needs jacobians at knot points for cost gradient
            for k = inds
                A[:,:,k], B[:,:,k] = solver.Fc(X[:,k],U[:,k])
            end
        else
            Z = packZ(X,U)
            z = reshape(Z,n+m,N)
            for k = 1:N_
                A[:,:,k] = solver.Fc(z[:,k])
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
function collocation_constraints(solver::Solver, res::DircolResults, method::Symbol)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver,method)
    g = zeros(eltype(res.X),(N-1)*n)
    collocation_constraints!(solver,res.X_,res.U_,res.fVal_,g,method)
    return g
end

function collocation_constraints!(solver::Solver, X, U, fVal, g_colloc, method::Symbol)
    # X,U need to be the "trajectory points", or X_,U_
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    g = reshape(g_colloc,n,N-1)
    dt = solver.dt

    if method == :trapezoid
        for k = 1:N-1
            # Collocation Constraints
            g[:,k] = dt*( fVal[:,k+1] + fVal[:,k] )/2 - X[:,k+1] + X[:,k]
        end
    elseif method == :hermite_simpson_separated || method == :hermite_simpson
        iLow = 1:2:N_-1
        iMid = iLow + 1
        iUpp = iMid + 1

        collocation = - X[:,iUpp] + X[:,iLow] + dt*(fVal[:,iLow] + 4*fVal[:,iMid] + fVal[:,iUpp])/6

        if method == :hermite_simpson
            g .= collocation
        else
            midpoints =  - X[:,iMid] + (X[:,iLow]+X[:,iUpp])/2 + dt*(fVal[:,iLow] - fVal[:,iUpp])/8
            g[:,iLow] = collocation
            g[:,iMid] = midpoints
        end

    elseif method == :midpoint
        Xm = X
        # Calculate the knot points from the midpoints (and the terminal point)
        Xk = zeros(Xm)
        Xk[:,end] = Xm[:,end]
        for k = N-1:-1:1
            Xk[:,k] = 2Xm[:,k] - Xk[:,k+1]
        end
        for k = 1:N-1
            g[:,k] = dt*fVal[:,k] - Xk[:,k+1] + Xk[:,k]
        end
    end
end

function collocation_constraints(X,U,method,dt,f::Function)
    n,m,N = get_sizes(X,U)
    g = zeros(eltype(X),(N-1)*n)
    g = reshape(g,n,N-1)

    fVal = zeros(eltype(X),n,N)
    for k = 1:N
        f(view(fVal,:,k),X[:,k],U[:,k])
    end

    if method == :trapezoid
        for k = 1:N-1
            # Collocation Constraints
            g[:,k] = dt*( fVal[:,k+1] + fVal[:,k] )/2 - X[:,k+1] + X[:,k]
        end
    elseif method == :hermite_simpson_separated
        iLow = 1:2:N-1
        iMid = iLow + 1
        iUpp = iMid + 1

        midpoints =  - X[:,iMid] + (X[:,iLow]+X[:,iUpp])/2 + dt*(fVal[:,iLow] - fVal[:,iUpp])/8
        collocation = - X[:,iUpp] + X[:,iLow] + dt*(fVal[:,iLow] + 4*fVal[:,iMid] + fVal[:,iUpp])/6
        g[:,iLow] = collocation
        g[:,iMid] = midpoints

    elseif method == :hermite_simpson
        fm = zeros(eltype(X),n)
        for k = 1:N-1
            x1 = X[:,k]
            x2 = X[:,k+1]
            xm = (x1+x2)/2 + dt/8*(fVal[:,k]-fVal[:,k+1])
            um = (U[:,k] + U[:,k+1])/2
            f(fm, xm, um)
            g[:,k] = -x2 + x1 + dt*(fVal[:,k] + 4*fm + fVal[:,k+1])/6
        end

    elseif method == :midpoint
        fm = zeros(eltype(X),n)
        for k = 1:N-1
            x1 = X[:,k]
            x2 = X[:,k+1]
            xm = (x1+x2)/2
            f(fm,xm,U[:,k])
            g[:,k] = dt*fm - x2 + x1
        end
    end
    return vec(g)
end

"""
Constraint Jacobian
"""
function constraint_jacobian(solver::Solver, res::DircolResults, method::Symbol)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    jacob_g = spzeros((N-1)*n,N*(n+m))
    jacob_g = constraint_jacobian(solver,res.X_,res.U_,res.A,res.B,method)
    return jacob_g
end

function constraint_jacobian(solver::Solver, X, U, A, B, method::Symbol)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    dt = solver.dt
    jacob_g = spzeros((N-1)*n,N*(n+m))
    Inm = Matrix(I,n,n+m)

    if method == :trapezoid
        Z = packZ(X,U)
        z = reshape(Z,n+m,N)

        # First time step
        fz = A[:,:,1]
        jacob_g[1:n,1:n+m] .= dt*fz/2+Inm

        # Loop over time steps
        for k = 2:N-1
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            # Calculate (n,n+m) Jacobian of both states and controls
            fz = A[:,:,k]  #F(z[:,k])
            jacob_g[off_1-n+(1:n),off_2+(1:n+m)] .= dt*fz/2 - Inm
            jacob_g[off_1 + (1:n),off_2+(1:n+m)] .= dt*fz/2 + Inm
        end

        # Last time step
        fz = A[:,:,N]  # F(z[:,N])
        jacob_g[end-n+1:end,end-n-m+1:end] = dt*fz/2-Inm

    elseif method == :hermite_simpson_separated
        nSeg = Int((N-1)/2)
        Z = packZ(X,U)
        z = reshape(Z,n+m,N)

        fz1 = A[:,:,1]  # F(z[:,1])

        function calc_block(k)
            vals = zeros(2n,3(n+m))
            fz = A[:,:,k]  # F(z[:,k])
            vals[0+(1:n),(1:n+m)] .= dt*fz/6 + Inm
            vals[n+(1:n),(1:n+m)] .= dt*fz/8 + Inm/2
            fm = A[:,:,k+1]  # F(z[:,k+1])
            vals[0+(1:n),n+m+(1:n+m)] .= 2*dt*fm/3
            vals[n+(1:n),n+m+(1:n+m)] .= -Inm
            fz1 .= A[:,:,k+2]  # F(z[:,k+2])
            vals[0+(1:n),2(n+m)+(1:n+m)] .=  dt*fz1/6 - Inm
            vals[n+(1:n),2(n+m)+(1:n+m)] .= -dt*fz1/8 + Inm/2
            return vals
        end


        for i = 1:nSeg
            off_1 = 2(i-1)*(n)
            off_2 = 2(i-1)*(n+m)
            k = 2i-1

            jacob_g[off_1+(1:2n), off_2+(1:3(n+m))] = calc_block(k)
        end
    elseif method == :hermite_simpson
        nSeg = N-1

        Xk = view(X,:,1:2:N_)
        Uk = view(U,:,1:2:N_)
        Xm = view(X,:,2:2:N_-1)
        Um = view(U,:,2:2:N_-1)
        Ak = view(A,:,:,1:2:N_)
        Bk = view(B,:,:,1:2:N_)
        AM = view(A,:,:,2:2:N_-1)
        BM = view(B,:,:,2:2:N_-1)

        function calc_jacob_block(k::Int)::Matrix
            x1,u1 = Xk[:,k],Uk[:,k]
            x2,u2 = Xk[:,k+1],Uk[:,k+1]
            A1,A2 = Ak[:,:,k],Ak[:,:,k+1]
            B1,B2 = Bk[:,:,k],Bk[:,:,k+1]
            xm = Xm[:,k] #(x1+x2)/2 + dt/8*(fVal[:,k]-fVal[:,k+1])
            um = Um[:,k] # (u1+u2)/2
            Am,Bm = AM[:,:,k],BM[:,:,k]
            In = Matrix(I,n,n)
            Im = Matrix(I,m,m)

            vals = zeros(n,2(n+m))
            vals[:,1:n] =        dt/6*(A1 + 4Am*( dt/8*A1 + In/2)) + In    # ∇x1
            vals[:,n+(1:m)] =    dt/6*(B1 + 4Am*( dt/8*B1) + 4Bm*(Im/2))   # ∇u1
            vals[:,n+m+(1:n)] =  dt/6*(A2 + 4Am*(-dt/8*A2 + In/2)) - In    # ∇x2
            vals[:,2n+m+(1:m)] = dt/6*(B2 + 4Am*(-dt/8*B2) + 4Bm*(Im/2))   # ∇u2
            return vals
        end

        for k = 1:nSeg
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            jacob_g[off_1+(1:n), off_2+(1:2(n+m))] = calc_jacob_block(k)
        end

    elseif method == :midpoint
        nSeg = N-1
        In = Matrix(I,n,n)

        function calc_jacob_block_midpoint(k)
            vals = zeros(n,2(n+m))
            vals[:,1:n] =        In + dt*A[:,:,k]/2    # ∇x1
            vals[:,n+(1:m)] =    dt*B[:,:,k]           # ∇u1
            vals[:,n+m+(1:n)] = -In + dt*A[:,:,k]/2    # ∇x2
            vals[:,2n+m+(1:m)] = zeros(n,m)            # ∇u2
            return vals
        end

        for k = 1:nSeg
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            jacob_g[off_1+(1:n), off_2+(1:2(n+m))] = calc_jacob_block_midpoint(k)
        end
    end

    return jacob_g
end


function constraint_jacobian!(solver::Solver, X, U, A, B, vals, method::Symbol)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    dt = solver.dt
    Inm = Matrix(I,n,n+m)

    if method == :trapezoid
        Z = packZ(X,U)
        z = reshape(Z,n+m,N)
        n_blk = 2(n+m)n

        function calc_block_trap(k,blk)
            blk = reshape(blk,n,2(n+m))
            blk[:,1:n+m] =       dt*A[:,:,k  ]/2+Inm
            blk[:,n+m+(1:n+m)] = dt*A[:,:,k+1]/2-Inm
            return nothing
        end


        for k = 1:N-1
            off = (k-1)*n_blk
            block = view(vals,off+(1:n_blk))
            calc_block_trap(k,block)
        end

    elseif method == :hermite_simpson_separated
        nSeg = Int((N-1)/2)
        Z = packZ(X,U)
        z = reshape(Z,n+m,N)

        n_blk = 6(n+m)n
        function calc_block_sep!(k,val_block)
            blk = reshape(val_block,2n,3(n+m))
            fz = A[:,:,k]  # F(z[:,k])
            blk[0+(1:n),(1:n+m)] = dt*fz/6 + Inm
            blk[n+(1:n),(1:n+m)] = dt*fz/8 + Inm/2
            fm = A[:,:,k+1]  # F(z[:,k+1])
            blk[0+(1:n),n+m+(1:n+m)] = 2*dt*fm/3
            blk[n+(1:n),n+m+(1:n+m)] = -Inm
            fz1 = A[:,:,k+2]  # F(z[:,k+2])
            blk[0+(1:n),2(n+m)+(1:n+m)] =  dt*fz1/6 - Inm
            blk[n+(1:n),2(n+m)+(1:n+m)] = -dt*fz1/8 + Inm/2
            return nothing
        end

        for i = 1:nSeg
            k = 2i-1
            off = (i-1)*n_blk
            block = view(vals,off+(1:n_blk))
            calc_block_sep!(k,block)
        end
    elseif method == :hermite_simpson
        nSeg = N-1
        n_blk = 2(n+m)n

        Xk = view(X,:,1:2:N_)
        Uk = view(U,:,1:2:N_)
        Xm = view(X,:,2:2:N_-1)
        Um = view(U,:,2:2:N_-1)
        Ak = view(A,:,:,1:2:N_)
        Bk = view(B,:,:,1:2:N_)
        AM = view(A,:,:,2:2:N_-1)
        BM = view(B,:,:,2:2:N_-1)

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

            vals = reshape(vals,n,2(n+m))
            vals[:,1:n] =        dt/6*(A1 + 4Am*( dt/8*A1 + In/2)) + In    # ∇x1
            vals[:,n+(1:m)] =    dt/6*(B1 + 4Am*( dt/8*B1) + 4Bm*(Im/2))   # ∇u1
            vals[:,n+m+(1:n)] =  dt/6*(A2 + 4Am*(-dt/8*A2 + In/2)) - In    # ∇x2
            vals[:,2n+m+(1:m)] = dt/6*(B2 + 4Am*(-dt/8*B2) + 4Bm*(Im/2))   # ∇u2
            return nothing
        end

        for k = 1:nSeg
            off = (k-1)*n_blk
            block = view(vals,off+(1:n_blk))
            calc_block_hs!(k,block)
        end

    elseif method == :midpoint
        nSeg = N-1
        n_blk = (2n+m)n
        In = Matrix(I,n,n)

        function calc_jacob_block_midpoint!(k,blk)
            blk = reshape(blk,n,2n+m)
            blk[:,1:n] =        In + dt*A[:,:,k]/2    # ∇x1
            blk[:,n+(1:m)] =    dt*B[:,:,k]           # ∇u1
            blk[:,n+m+(1:n)] = -In + dt*A[:,:,k]/2    # ∇x2
            return nothing
        end

        for k = 1:nSeg
            off = (k-1)*n_blk
            block = view(vals,off+(1:n_blk))
            calc_jacob_block_midpoint!(k,block)
        end
    end

    return nothing
end


function constraint_jacobian_sparsity(solver::Solver, method::Symbol)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    jacob_g = spzeros(Int,(N-1)*n,N*(n+m))
    dt = solver.dt
    i = 0

    if method == :trapezoid || method == :hermite_simpson

        n_blk = 2(n+m)n
        num_block = reshape(1:n_blk,n,2(n+m))

        for k = 1:N-1
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            jacob_g[off_1+(1:n), off_2+(1:2(n+m))] = num_block + i
            i += n_blk
        end

    elseif method == :hermite_simpson_separated
        nSeg = Int((N-1)/2)
        n_blk = 6(n+m)n
        num_block = reshape(1:n_blk,2n,3(n+m))


        for k = 1:nSeg
            off_1 = 2(k-1)*(n)
            off_2 = 2(k-1)*(n+m)

            jacob_g[off_1+(1:2n), off_2+(1:3(n+m))] = num_block + i
            i += n_blk
        end

    elseif method == :midpoint
        nSeg = N-1
        n_blk = (2n+m)n
        num_block = reshape(1:n_blk,n,2n+m)

        for k = 1:nSeg
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            jacob_g[off_1+(1:n), off_2+(1:2n+m)] = num_block + i
            i += n_blk
        end
    end

    rows,cols,inds = findnz(jacob_g)
    v = sortperm(inds)
    rows = rows[v]
    cols = cols[v]

    # return jacob_g
    return rows,cols
end


"""
$(SIGNATURES)
Interpolate a trajectory using cubic interpolation
"""
function interp_traj(N::Int,tf::Float64,X::Matrix,U::Matrix)::Tuple{Matrix,Matrix}
    X2 = interp_rows(N,tf,X)
    U2 = interp_rows(N,tf,U)
    return X2, U2
end

"""
$(SIGNATURES)
Interpolate the rows of a matrix using cubic interpolation
"""
function interp_rows(N::Int,tf::Float64,X::Matrix)::Matrix
    n,N1 = size(X)
    t1 = linspace(0,tf,N1)
    t2 = linspace(0,tf,N)
    X2 = zeros(n,N)
    for i = 1:n
        interp_cubic = CubicSplineInterpolation(t1, X[i,:])
        X2[i,:] = interp_cubic(t2)
    end
    return X2
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
    τ = t-T[k]
    if method == :trapezoid
        u = U[:,k] + τ/dt*(F[:,k+1]-F[:,k])
        x = X[:,k] + F[:,k]*τ + τ^2/(2*dt)*(F[:,k+1]-F[:,k])
    elseif method == :hermite_simpson || method == :hermite_simpson_separated
        x1,x2 = X[:,k], X[:,k+1]
        u1,u2 = U[:,k], U[:,k+1]
        f1,f2 = F[:,k], F[:,k+1]
        xm = (x1+x2)/2 + dt/8*(f1-f2)
        um = (U[:,k] + U[:,k+1])/2

        u = (2(τ-dt/2)(τ-dt)U[:,k] - 4τ*(τ-dt)Um + 2τ*(τ-dt/2)*U[:,k+1])/dt^2
        x = X[:,k] + F[:,k]*τ/dt + 1/2*(-3F[:,])
    end
    return x,u
end
