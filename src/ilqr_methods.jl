using RigidBodyDynamics
using ForwardDiff
using Plots
using Base.Test
using BenchmarkTools

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Important methods used by the forward and backward iLQR passes
#
#     GENERAL METHODS
#         rollout!: Compute state trajectory X given controls U
#         cost: Compute the cost
#         calc_jacobians: Compute jacobians
#     CONSTRAINTS:
#         update_constraints!: Update constraint values and handle activation of
#             inequality constraints
#         generate_constraint_functions: Given a ConstrainedObjective, generate
#             the constraint function and its jacobians
#         max_violation: Compute the maximum constraint violation
#     INFEASIBLE START:
#         infeasible_controls: Compute the augmented (infeasible) controls
#             required to meet the specified trajectory
#         line_trajectory: Generate a linearly interpolated state trajectory
#             between start and end
#         feasible_traj: Finish an infeasible start solve by removing the
#             augmented controls
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



########################################
###         GENERAL METHODS          ###
########################################

"""
$(SIGNATURES)
Roll out the dynamics for a given control sequence (initial)
Updates `res.X` by propagating the dynamics, using the controls specified in
`res.U`.
"""
function rollout!(res::SolverResults,solver::Solver)
    infeasible = solver.model.m != size(res.U,1)
    X = res.X; U = res.U

    X[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        if solver.control_integration == :foh
            solver.fd(view(X,:,k+1), X[:,k], U[1:solver.model.m,k], U[1:solver.model.m,k+1])
        else
            solver.fd(view(X,:,k+1), X[:,k], U[1:solver.model.m,k])
        end

        if infeasible
            X[:,k+1] .+= U[solver.model.m+1:solver.model.m+solver.model.n,k]
        end
        if ~all(isfinite, X[:,k+1]) || ~all(isfinite, U[:,k])
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)
Roll out the dynamics using the gains and optimal controls computed by the
backward pass
Updates `res.X` by propagating the dynamics at each timestep, by applying the
gains `res.K` and `res.d` to the difference between states
Will return a flag indicating if the values are finite for all time steps.
"""
function rollout!(res::SolverResults,solver::Solver,alpha::Float64)
    infeasible = solver.model.m != size(res.U,1)
    N = solver.N; m = solver.model.m; n = solver.model.n
    if infeasible
        m += n
    end
    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[:,1] = solver.obj.x0;

    if solver.control_integration == :foh
        b = res.b
        du = zeros(m)
        dv = zeros(m)
        du .= d[:,1]
        U_[:,1] .= U[:,1] + du
    end

    for k = 2:N
        delta = X_[:,k-1] - X[:,k-1]

        if solver.control_integration == :foh
            dv .= K[:,:,k]*delta + b[:,:,k]*du + alpha*d[:,k]
            U_[:,k] .= U[:,k] + dv
            solver.fd(view(X_,:,k), X_[:,k-1], U_[1:solver.model.m,k-1], U_[1:solver.model.m,k])
            du .= dv
        else
            U_[:, k-1] = U[:, k-1] - K[:,:,k-1]*delta - alpha*d[:,k-1]
            solver.fd(view(X_,:,k), X_[:,k-1], U_[1:solver.model.m,k-1])
        end

        if infeasible
            X_[:,k] .+= U_[solver.model.m+1:solver.model.m+solver.model.n,k-1]
        end

        if ~all(isfinite, X_[:,k]) || ~all(isfinite, U_[:,k-1])
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)
Quadratic stage cost (with goal state)
"""
function stage_cost(x,u,Q,R,xf)
    0.5*(x - xf)'*Q*(x - xf) + 0.5*u'*R*u
end

"""
$(SIGNATURES)
Evaluate state midpoint using cubic spline interpolation
"""
function xm_func(x,u,y,dt,fc!)
    tmp = zeros(x)
    fc!(tmp,x,u)
    0.75*x + 0.25*dt*tmp + 0.25*y
end

"""
$(SIGNATURES)
Compute the unconstrained cost
"""
function cost(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    # pull out solver/objective values
    N = solver.N; Q = solver.obj.Q;xf = solver.obj.xf; Qf = solver.obj.Qf; m = solver.model.m; n = solver.model.n
    if size(U,1) != m
        m += n
    end
    R = getR(solver)
    # R = solver.obj.R
    J = 0.0
    for k = 1:N-1
        if solver.control_integration == :foh
            Ac, Bc = solver.Fc(X[:,k],U[:,k])
            # println("size Ac: $(size(Ac)), Bc: $(size(Bc))")
            M = 0.25*[3*eye(n)+solver.dt*Ac solver.dt*Bc eye(n) zeros(n,m)]
            # println("size M: $(size(M))")
            # println("size vec: $(size([X[:,k];U[:,k];X[:,k+1];U[:,k+1]]))")
            Xm = M*[X[:,k];U[:,k];X[:,k+1];U[:,k+1]]
            # Xm = xm_func(X[:,k],U[:,k],X[:,k+1],solver.dt,solver.fc)
            Um = (U[:,k] + U[:,k+1])/2
            J += solver.dt/6*(stage_cost(X[:,k],U[:,k],Q,R,xf) + 4*stage_cost(Xm,Um,Q,R,xf) + stage_cost(X[:,k+1],U[:,k+1],Q,R,xf)) # rk3 foh stage cost (integral approximation)
        else
            J += stage_cost(X[:,k],U[:,k],Q,R,xf)
        end
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

""" $(SIGNATURES) Compute the Constrained Cost """
function cost(solver::Solver, res::ConstrainedResults, X::Array{Float64,2}=res.X, U::Array{Float64,2}=res.U)
    J = cost(solver, X, U)
    N = solver.N
    for k = 1:N-1
        if solver.control_integration == :foh
            J += 0.5*(res.C[:,k]'*res.Iμ[:,:,k]*res.C[:,k] + res.LAMBDA[:,k]'*res.C[:,k] + res.C[:,k+1]'*res.Iμ[:,:,k+1]*res.C[:,k+1] + res.LAMBDA[:,k+1]'*res.C[:,k+1])
        else
            J += 0.5*(res.C[:,k]'*res.Iμ[:,:,k]*res.C[:,k] + res.LAMBDA[:,k]'*res.C[:,k])
        end
    end

    J += 0.5*(res.CN'*res.IμN*res.CN + res.λN'*res.CN)
    return J
end

# Equivalent call signature for constrained cost
function cost(solver::Solver, res::UnconstrainedResults, X::Array{Float64,2}=res.X, U::Array{Float64,2}=res.U)
    cost(solver,X,U)
end

"""
$(SIGNATURES)
Calculate Jacobians prior to the backwards pass
Updates both dyanmics and constraint jacobians, depending on the results type.
"""
function calc_jacobians(res::ConstrainedResults, solver::Solver)::Void #TODO change to inplace '!' notation throughout the code
    N = solver.N
    for k = 1:N-1
        if solver.control_integration == :foh
            res.fx[:,:,k], res.fu[:,:,k], res.fv[:,:,k] = solver.Fd(res.X[:,k], res.U[:,k], res.U[:,k+1])
            res.Ac[:,:,k], res.Bc[:,:,k] = solver.Fc(res.X[:,k], res.U[:,k])
        else
            res.fx[:,:,k], res.fu[:,:,k] = solver.Fd(res.X[:,k], res.U[:,k])
        end
        res.Cx[:,:,k], res.Cu[:,:,k] = solver.c_jacobian(res.X[:,k],res.U[:,k])
    end

    if solver.control_integration == :foh
        res.Cx[:,:,N], res.Cu[:,:,N] = solver.c_jacobian(res.X[:,N],res.U[:,N])
    end

    res.Cx_N .= solver.c_jacobian(res.X[:,N])
    return nothing
end

function calc_jacobians(res::UnconstrainedResults, solver::Solver, infeasible=false)::Void
    N = solver.N
    for k = 1:N-1
        if solver.control_integration == :foh
            res.fx[:,:,k], res.fu[:,:,k], res.fv[:,:,k] = solver.Fd(res.X[:,k], res.U[:,k], res.U[:,k+1])
            res.Ac[:,:,k], res.Bc[:,:,k] = solver.Fc(res.X[:,k], res.U[:,k])
        else
            res.fx[:,:,k], res.fu[:,:,k] = solver.Fd(res.X[:,k], res.U[:,k])
        end
    end
    return nothing
end


########################################
### METHODS FOR CONSTRAINED PROBLEMS ###
########################################

"""
$(SIGNATURES)
Evalutes all inequality and equality constraints (in place) for the current state and control trajectories
"""
function update_constraints!(res::ConstrainedResults, solver::Solver, X::Array, U::Array)::Void

    p, N = size(res.C) # note, C is now (p,N)
    c = solver.c_fun
    pI = solver.obj.pI

    if solver.control_integration == :foh
        final_index = N
    else
        final_index = N-1
    end

    for k = 1:final_index
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
function generate_constraint_functions(obj::ConstrainedObjective)
    m = size(obj.R,1) # number of control inputs
    n = length(obj.x0) # number of states

    p = obj.p # number of constraints
    pI = obj.pI # number of inequality and equality constraints
    pE = p-pI # number of equality constraints
    pE_c = pE  # number of custom equality constraints

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
        infeasible = length(u) != m
        C[1:pI] = cI(x,u[1:m])
        C[(1:pE_c)+pI] = obj.cE(x,u[1:m])
        if infeasible
            return [C; u[m+1:end]]
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

    fu_control = zeros(pI_u,m)
    fu_control[1:pI_u_max,:] = -eye(pI_u_max)
    fu_control[pI_u_max+1:end,:] = eye(pI_u_min)
    fu_state = zeros(pI_x,m)
    fu = zeros(p,m)

    fu_infeasible = eye(n)
    fx_infeasible = zeros(n,n)

    fx_N = eye(n)  # Jacobian of final state

    function constraint_jacobian(x::Array,u::Array)
        infeasible = length(u) != m
        fx[1:pI_u, :] = fx_control
        fu[1:pI_u, :] = fu_control
        fx[(1:pI_x)+pI_u, :] = fx_state
        fu[(1:pI_x)+pI_u, :] = fu_state
        # F_aug = F([x;u]) # TODO handle general constraints
        # fx = F_aug[:,1:n]
        # fu = F_aug[:,n+1:n+m]

        if infeasible
            return [fx; fx_infeasible], cat((1,2),fu,fu_infeasible)
        end
        return fx,fu
    end

    function constraint_jacobian(x::Array)
        return fx_N
    end

    return c_fun, constraint_jacobian
end

generate_constraint_functions(obj::UnconstrainedObjective) = (x,u)->nothing, (x,u)->nothing


"""
$(SIGNATURES)
Compute the maximum constraint violation. Inactive inequality constraints are
not counted (masked by the Iμ matrix). For speed, the diagonal indices can be
precomputed and passed in.
"""
function max_violation(results::ConstrainedResults,inds=CartesianIndex.(indices(results.Iμ,1),indices(results.Iμ,2)))
    maximum(abs.(results.C.*(results.Iμ[inds,:] .!= 0)))
end


####################################
### METHODS FOR INFEASIBLE START ###
####################################

"""
$(SIGNATURES)
Additional controls for producing an infeasible state trajectory
"""
function infeasible_controls(solver::Solver,X0::Array{Float64,2},u::Array{Float64,2})
    ui = zeros(solver.model.n,solver.N) # initialize
    m = solver.model.m
    x = zeros(solver.model.n,solver.N)
    x[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        if solver.control_integration == :foh
            solver.fd(view(x,:,k+1),x[:,k],u[1:m,k],u[1:m,k+1])
        else
            solver.fd(view(x,:,k+1),x[:,k],u[1:m,k])
        end
        ui[:,k] = X0[:,k+1] - x[:,k+1]
        x[:,k+1] .+= ui[:,k]
    end
    ui
end

function infeasible_controls(solver::Solver,X0::Array{Float64,2})
    u = zeros(solver.model.m,solver.N)
    infeasible_controls(solver,X0,u)
end

"""
$(SIGNATURES)
Infeasible start solution is run through standard constrained solve to enforce dynamic feasibility. All infeasible augmented controls are removed.
"""
function feasible_traj(results::ConstrainedResults,solver::Solver)
    #solver.opts.iterations_outerloop = 3 # TODO: this should be run to convergence, but can be reduce for speedup
    solver.opts.infeasible = false
    if solver.opts.unconstrained
        obj_uncon = UnconstrainedObjective(solver.obj.Q,solver.obj.R,solver.obj.Qf,solver.obj.tf,solver.obj.x0,solver.obj.xf)
        solver_uncon = Solver(solver.model,obj_uncon,integration=solver.integration,dt=solver.dt,opts=solver.opts)
        return solve(solver_uncon,results.U[1:solver.model.m,:])
    else
        return solve(solver,results.U[1:solver.model.m,:],prevResults=results)
    end
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

function line_trajectory(solver::Solver)
    line_trajectory(solver.obj.x0,solver.obj.xf,solver.N)
end
