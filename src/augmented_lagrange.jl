using RigidBodyDynamics
using ForwardDiff
using Plots
using Base.Test
using BenchmarkTools

"""
$(SIGNATURES)

Additional controls for producing an infeasible state trajectory
"""
function infeasible_controls(solver::Solver,x0::Array{Float64,2},u::Array{Float64,2})
    ui = zeros(solver.model.n,solver.N-1) # initialize
    x = zeros(solver.model.n,solver.N)
    x[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        solver.fd(view(x,:,k+1),x[:,k],u[:,k])
        ui[:,k] = x0[:,k+1] - x[:,k+1]
        x[:,k+1] .+= ui[:,k]
    end
    ui
end

function infeasible_controls(solver::Solver,x0::Array{Float64,2})
    u = zeros(solver.model.m,solver.N-1)
    infeasible_controls(solver,x0,u)
end

"""
$(SIGNATURES)
Roll out the dynamics for a given control sequence (initial)

Updates `res.X` by propagating the dynamics, using the controls specified in
`res.U`.
"""
function rollout!(res::SolverResults,solver::Solver;infeasible::Bool=false)
    X = res.X; U = res.U

    X[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        solver.fd(view(X,:,k+1), X[:,k], U[1:solver.model.m,k])
        if infeasible
            X[:,k+1] .+= U[solver.model.m+1:end,k]
        end
    end
end

"""
$(SIGNATURES)
Roll out the dynamics using the gains and optimal controls computed by the
backward pass

Updates `res.X` by propagating the dynamics at each timestep, by applying the
gains `res.K` and `res.d` to the difference between states

Will return a flag indicating if the values are finite for all time steps.
"""
function rollout!(res::SolverResults,solver::Solver,alpha::Float64;infeasible::Bool=false)
    N = solver.N
    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[:,1] = solver.obj.x0;
    for k = 2:N
        delta = X_[:,k-1] - X[:,k-1]
        U_[:, k-1] = U[:, k-1] - K[:,:,k-1]*delta - alpha*d[:,k-1]
        solver.fd(view(X_,:,k), X_[:,k-1], U_[1:solver.model.m,k-1])

        if infeasible
            X_[:,k] .+= U_[solver.model.m+1:end,k-1]
        end

        if ~all(isfinite, X_[:,k]) || ~all(isfinite, U_[:,k-1])
            return false
        end
    end
    return true
end

# overloaded cost function to accomodate Augmented Lagrance method
# TODO: Make cost a function only a function of results
function cost(solver::Solver, res::ConstrainedResults, X::Array{Float64,2}, U::Array{Float64,2}; infeasible::Bool=false)
    J = cost(solver, X, U, infeasible=infeasible)
    for k = 1:solver.N-1
        J += 0.5*(res.C[:,k]'*res.Iμ[:,:,k]*res.C[:,k] + res.LAMBDA[:,k]'*res.C[:,k])
    end
    J += 0.5*(res.CN'*res.IμN*res.CN + res.λN'*res.CN)
    return J
end

"""
$(SIGNATURES)
Compute the unconstrained cost
"""
function cost(solver::Solver,X::Array{Float64,2},U::Array{Float64,2};infeasible::Bool=false)
    # pull out solver/objective values
    N = solver.N; Q = solver.obj.Q;xf = solver.obj.xf; Qf = solver.obj.Qf

    if infeasible
        R = solver.opts.infeasible_regularization*eye(solver.model.m+solver.model.n)
        R[1:solver.model.m,1:solver.model.m] = solver.obj.R
    else
        R = solver.obj.R
    end

    J = 0.0
    for k = 1:N-1
      J += 0.5*(X[:,k] - xf)'*Q*(X[:,k] - xf) + 0.5*U[:,k]'*R*U[:,k]
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

"""
$(SIGNATURES)
Propagate dynamics with a line search (in-place)
"""
function forwardpass!(res::ConstrainedResults, solver::Solver, v1::Float64, v2::Float64, c_fun::Function;infeasible::Bool=false)

    # Pull out values from results
    X = res.X
    U = res.U
    K = res.K
    d = res.d
    X_ = res.X_
    U_ = res.U_
    C = res.C
    Iμ = res.Iμ
    LAMBDA = res.LAMBDA
    MU = res.MU

    # Compute original cost
    # J_prev = cost(solver, X, U, C, Iμ, LAMBDA)
    J_prev = cost(solver, res, X, U, infeasible=infeasible)

    pI = solver.obj.pI

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z ≤ solver.opts.c1 || z > solver.opts.c2
        rollout!(res,solver,alpha,infeasible=infeasible)

        # Calcuate cost
        update_constraints!(res,c_fun,pI,X_,U_)
        J = cost(solver, res, X_, U_, infeasible=infeasible)

        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV[1,1]
        if iter < 10
            alpha = alpha/2.
        else
            alpha = alpha/10.
        end

        if iter > solver.opts.iterations_linesearch
            if solver.opts.verbose
                println("max iterations (forward pass)")
            end
            break
        end
        iter += 1
    end

    if solver.opts.verbose
        max_c = max_violation(res)
        println("New cost: $J")
        println("- Max constraint violation: $max_c")
        println("- Expected improvement: $(dV[1])")
        println("- Actual improvement: $(J_prev-J)")
        println("- (z = $z)\n")
    end

    return J

end

"""
$(SIGNATURES)
Solve the dynamic programming problem, starting from the terminal time step

Computes the gain matrices K and d by applying the principle of optimality at
each time step, solving for the gradient (s) and Hessian (S) of the cost-to-go
function. Also returns parameters `v1` and `v2` (see Eq. 25a in Yuval Tassa Thesis)
"""
function backwardpass!(res::ConstrainedResults, solver::Solver, constraint_jacobian::Function;infeasible::Bool=false)
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    Q = solver.obj.Q

    if infeasible
        R = solver.opts.infeasible_regularization*eye(m+n)
        R[1:m,1:m] = solver.obj.R
    else
        R = solver.obj.R
    end

    xf = solver.obj.xf
    Qf = solver.obj.Qf

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA

    Cx, Cu = constraint_jacobian(res.X[:,N])
    S = Qf + Cx'*res.IμN*Cx
    s = Qf*(X[:,N] - xf) + + Cx'*res.IμN*res.CN + Cx'*res.λN
    v1 = 0.
    v2 = 0.

    mu = 0.
    k = N-1
    while k >= 1
        lx = Q*(X[:,k] - xf)
        lu = R*(U[:,k])
        lxx = Q
        luu = R

        if infeasible
            fx, fu = solver.F(X[:,k], U[1:m,k])
            fu = [fu eye(n)]
        else
            fx, fu = solver.F(X[:,k], U[:,k])
        end

        Qx = lx + fx'*s
        Qu = lu + fu'*s
        Qxx = lxx + fx'*S*fx
        Quu = Hermitian(luu + fu'*(S + mu*eye(n))*fu)
        Qux = fu'*(S + mu*eye(n))*fx

        # regularization
        if ~isposdef(Quu)
            mu = mu + solver.opts.mu_regularization;
            k = N-1
            if solver.opts.verbose
                println("regularized")
            end
        end

        # Constraints
        Cx, Cu = constraint_jacobian(X[:,k], U[:,k])
        Qx += Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k]
        Qu += Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k]
        Qxx += Cx'*Iμ[:,:,k]*Cx
        Quu += Cu'*Iμ[:,:,k]*Cu
        Qux += Cu'*Iμ[:,:,k]*Cx
        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)'
        S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]

        # terms for line search
        v1 += float(d[:,k]'*Qu)[1]
        v2 += float(d[:,k]'*Quu*d[:,k])

        k = k - 1;
    end
    return v1, v2
end

"""
$(SIGNATURES)

Evalutes all inequality and equality constraints (in place) for the current state and control trajectories
"""
function update_constraints!(res::ConstrainedResults, c::Function, pI::Int, X::Array, U::Array)::Void
    p, N = size(res.C)
    N += 1 # since C is size (p,N-1), terminal constraints are handled separately
    for k = 1:N-1
        res.C[:,k] = c(X[:,k], U[:,k]) # update results with constraint evaluations

        # Inequality constraints [see equation ref]
        for j = 1:pI
            if res.C[j,k] < 0. || res.LAMBDA[j,k] < 0.
                res.Iμ[j,j,k] = res.MU[j,k] # active (or previously active) inequality constraints are penalized
            else
                res.Iμ[j,j,k] = 0. # inactive inequality constraints are not penalized
            end
        end

        # Equality constraints
        for j = pI+1:p
            res.Iμ[j,j,k] = res.MU[j,k] # equality constraints are penalized
        end
    end

    # Terminal constraint
    res.CN .= c(X[:,N])
    res.IμN .= diagm(res.μN)
    return nothing # TODO allow for more general terminal constraint
end

"""
$(SIGNATURES)

Generate the constraints function C(x,u) and a function to compute the jacobians
Cx, Cu = Jc(x,u) from a `ConstrainedObjective` type. Automatically stacks inequality
and equality constraints and takes jacobians of custom functions with `ForwardDiff`.

Stacks the constraints as follows:
[upper control inequalities
 lower control inequalities
 upper state inequalities
 lower state inequalities
 general inequalities
 general equalities
 (control equalities for infeasible start)]
"""
function generate_constraint_functions(obj::ConstrainedObjective; infeasible::Bool=false)
    m = size(obj.R,1) # number of control inputs
    n = length(obj.x0) # number of states

    p = obj.p # number of constraints
    pI = obj.pI # number of inequality and equality constraints
    pE = p-pI # number of equality constraints
    pE_c = pE  # number of custom equality constraints

    if infeasible
        p += n
        pE += n
        m_aug = m + n
    end

    u_min_active = isfinite.(obj.u_min)
    u_max_active = isfinite.(obj.u_max)
    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    # Inequality on control
    pI_u_max = count(u_max_active)
    pI_u_min = count(u_min_active)
    pI_u = pI_u_max + pI_u_min
    cI_u = zeros(pI_u)
    function c_control(x,u)
        [(obj.u_max - u)[u_max_active];
         (u - obj.u_min)[u_min_active]]
    end

    # Inequality on state
    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min
    function c_state(x,u)
        [(obj.x_max - x)[x_max_active];
         (x - obj.x_min)[x_min_active]]
    end

    # Custom constraints
    pI_c = pI - pI_x - pI_u
    # TODO add custom constraints

    # Form inequality constraint
    CI = zeros(pI)
    function cI(x,u)
        CI[1:pI_u] = c_control(x,u)
        CI[(1:pI_x)+pI_u] = c_state(x,u)
        CI[(1:pI_c)+pI_u+pI_x] = obj.cI(x,u)
        return CI
    end

    # Augment functions together
    C = zeros(p)
    function c_fun(x,u)
        C[1:pI] = cI(x,u[1:m])
        C[(1:pE_c)+pI] = obj.cE(x,u[1:m])
        if infeasible
            C[pI+pE_c+1:end] = u[m+1:end]
        end
        return C
    end

    # TODO make this more general
    function c_fun(x)
        x - obj.xf
    end

    ### Jacobians ###
    # Declare known jacobians
    fx_control = zeros(pI_u,n)
    fx_state = zeros(pI_x,n)
    fx_state[1:pI_x_max, :] = -eye(pI_x_max)
    fx_state[pI_x_max+1:end,:] = eye(pI_x_min)
    fx = zeros(p,n)

    if infeasible
        fx_infeasible = zeros(n,n)
        fu_infeasible = zeros(n,m_aug)
        fu_infeasible[:,m+1:end] = eye(n)
        fu_control = zeros(pI_u,m_aug)
        fu_control[1:pI_u_max, 1:m] = -eye(pI_u_max)
        fu_control[pI_u_max+1:end, 1:m] = eye(pI_u_min)
        fu_state = zeros(pI_x,m_aug)
        fu = zeros(p,m_aug)
    else
        fu_control = zeros(pI_u,m)
        fu_control[1:pI_u_max,:] = -eye(pI_u_max)
        fu_control[pI_u_max+1:end,:] = eye(pI_u_min)
        fu_state = zeros(pI_x,m)
        fu = zeros(p,m)
    end

    fx_N = eye(n)  # Jacobian of final state

    function constraint_jacobian(x::Array,u::Array)
        fx[1:pI_u, :] = fx_control
        fu[1:pI_u, :] = fu_control
        fx[(1:pI_x)+pI_u, :] = fx_state
        fu[(1:pI_x)+pI_u, :] = fu_state
        # F_aug = F([x;u]) # TODO handle general constraints
        # fx = F_aug[:,1:n]
        # fu = F_aug[:,n+1:n+m]

        if infeasible
            fx[pI+pE_c+1:end,:] = fx_infeasible
            fu[pI+pE_c+1:end,:] = fu_infeasible
        end
        return fx, fu
    end

    function constraint_jacobian(x::Array)
        return fx_N
    end

    return c_fun, constraint_jacobian
end

"""
$(SIGNATURES)

Compute the maximum constraint violation. Inactive inequality constraints are
not counted (masked by the Iμ matrix). For speed, the diagonal indices can be
precomputed and passed in.
"""
function max_violation(results::ConstrainedResults,inds=CartesianIndex.(indices(results.Iμ,1),indices(results.Iμ,2)))
    maximum(abs.(results.C.*(results.Iμ[inds,:] .!= 0)))
end

"""
$(SIGNATURES)

Solve constrained optimization problem using an initial control trajectory
"""
function solve_al(solver::Solver,U0::Array{Float64,2})
    solve_al(solver,zeros(solver.model.n,solver.N),U0,infeasible=false)
end

"""
$(SIGNATURES)

Solve constrained optimization problem specified by `solver`
"""
# QUESTION: Should the infeasible tag be able to be changed? What happens if we turn it off with an X0?
function solve_al(solver::Solver,X0::Array{Float64,2},U0::Array{Float64,2};infeasible::Bool=true)::SolverResults
    ## Unpack model, objective, and solver parameters
    N = solver.N # number of iterations for the solver (ie, knotpoints)
    n = solver.model.n # number of states
    m = solver.model.m # number of control inputs
    p = solver.obj.p # number of inequality and equality constraints
    pI = solver.obj.pI # number of inequality constraints

    if infeasible
        ui = infeasible_controls(solver,X0,U0) # generates n additional control input sequences that produce the desired infeasible state trajectory
        m += n # augment the number of control input sequences by the number of states
        p += n # increase the number of constraints by the number of additional control input sequences
    end

    ## Initialize results
    results = ConstrainedResults(n,m,p,N) # preallocate memory for results

    if infeasible
        #solver.obj.x0 = X0[:,1] # TODO not sure this is correct or needs to be here
        results.X .= X0 # initialize state trajectory with infeasible trajectory input
        results.U .= [U0; ui] # augment control with additional control inputs that produce infeasible state trajectory
    else
        results.U .= U0 # initialize control to control input sequence
    end

    # Unpack results for convenience
    X = results.X # state trajectory
    U = results.U # control trajectory
    X_ = results.X_ # updated state trajectory
    U_ = results.U_ # updated control trajectory

    # Diagonal indicies for the Iμ matrix (fast)
    diag_inds = CartesianIndex.(indices(results.Iμ,1),indices(results.Iμ,2))

    # Generate constraint function and jacobian functions from the objective
    c_fun, constraint_jacobian = generate_constraint_functions(solver.obj,infeasible=infeasible)

    ## Solver
    # Initial rollout
    if !infeasible
        X[:,1] = solver.obj.x0 # set state trajector initial conditions
        rollout!(results,solver) # rollout new state trajectoy
    end

    # Evalute constraints for new trajectories
    update_constraints!(results,c_fun,pI,X,U)

    if solver.opts.cache
        # Initialize cache and store initial trajectories and cost
        iter = 1 # counter for total number of iLQR iterations
        results_cache = ResultsCache(solver,solver.opts.iterations*solver.opts.iterations_outerloop+1) #TODO preallocate smaller arrays
        results_cache.result[iter] = results
        results_cache.cost[iter] = cost(solver, results, X, U, infeasible=infeasible)
        iter += 1
    end

    # Outer Loop
    for k = 1:solver.opts.iterations_outerloop
        J_prev = cost(solver, results, X, U, infeasible=infeasible) # calculate cost for current trajectories and constraint violations

        if solver.opts.verbose
            println("Cost ($k): $J_prev\n")
        end

        for i = 1:solver.opts.iterations
            if solver.opts.verbose
                println("--Iteration: $k-($i)--")
            end

            if solver.opts.cache
                t1 = time_ns() # time flag for iLQR inner loop start
            end

            # Backward pass
            if solver.opts.square_root
                v1, v2 = backwards_sqrt(results,solver, constraint_jacobian=constraint_jacobian, infeasible=infeasible) #TODO option to help avoid ill-conditioning [see algorithm xx]
            else
                v1, v2 = backwardpass!(results, solver, constraint_jacobian,infeasible=infeasible) # standard backward pass [see insert algorithm]
            end

            # Forward pass
            J = forwardpass!(results, solver, v1, v2, c_fun,infeasible=infeasible)

            if solver.opts.cache
                t2 = time_ns() # time flag of iLQR inner loop end
            end

            # Update results
            X .= X_
            U .= U_
            dJ = copy(abs(J-J_prev)) # change in cost
            J_prev = copy(J)

            if solver.opts.cache
                # Store current results and performance parameters
                results_cache.result[iter] = results
                results_cache.cost[iter] = J
                results_cache.time[iter] = (t2-t1)/(1.0e9)
                iter += 1
            end

            # Check for cost convergence
            if dJ < solver.opts.eps
                if solver.opts.verbose
                    println("   eps criteria met at iteration: $i\n")
                end
                break
            end
        end

        if solver.opts.cache
            results_cache.iter_type[iter-1] = 1 # flag outerloop update
        end

        ## Outer loop update for Augmented Lagrange Method parameters
        outer_loop_update(results,solver)

        # Check if maximum constraint violation satisfies termination criteria
        max_c = max_violation(results, diag_inds)
        if max_c < solver.opts.eps_constraint
            if solver.opts.verbose
                println("\teps constraint criteria met at outer iteration: $k\n")
            end
            break
        end
    end

    if solver.opts.cache
        # Store final results
        results_cache.termination_index = iter-1
        results_cache.X = results.X
        results_cache.U = results.U
    end

    ## Return dynamically feasible trajectory
    if infeasible
        if solver.opts.cache
            results_cache_2 = feasible_traj(results,solver) # using current control solution, warm-start another solve with dynamics strictly enforced
            return merge_results_cache(results_cache,results_cache_2) # return infeasible results and final enforce dynamics results
        else
            return feasible_traj(results,solver)
        end
    else
        if solver.opts.cache
            return results_cache
        else
            return results
        end
    end
end


"""
$(SIGNATURES)

Updates penalty (μ) and Lagrange multiplier (λ) parameters for Augmented Lagrange Method. λ is updated for equality and inequality constraints according to [insert equation ref] and μ is incremented by a constant term for all constraint types.
"""
function outer_loop_update(results::ConstrainedResults,solver::Solver)::Void
    p,N = size(results.C)
    N += 1
    for jj = 1:N-1
        for ii = 1:p
            if ii <= solver.obj.pI
                results.LAMBDA[ii,jj] .+= results.MU[ii,jj]*min(results.C[ii,jj],0)
            else
                results.LAMBDA[ii,jj] .+= results.MU[ii,jj]*results.C[ii,jj]
            end
            results.MU[ii,jj] .+= solver.opts.mu_al_update
        end
    end
    results.λN .+= results.μN.*results.CN
    results.μN .+= solver.opts.mu_al_update
    return nothing
end

"""
$(SIGNATURES)

Infeasible start solution is run through standard constrained solve to enforce dynamic feasibility. All infeasible augmented controls are removed.
"""
function feasible_traj(results::ConstrainedResults,solver::Solver)
    #solver.opts.iterations_outerloop = 3 # TODO: this should be run to convergence, but can be reduce for speedup
    return solve_al(solver,results.U[1:solver.model.m,:])
end

"""
$(SIGNATURES)

Linear interpolation trajectory between initial and final state(s)
"""
function line_trajectory(x0::Array{Float64,1},xf::Array{Float64,1},N::Int64)::Array{Float64,2}
    x_traj = zeros(size(x0,1),N)
    t = linspace(0,N,N)
    slope = (xf-x0)./N
    for i = 1:size(x0,1)
        x_traj[i,:] = slope[i].*t
    end
    x_traj
end
