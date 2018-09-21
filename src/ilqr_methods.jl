# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Important methods used by the forward and backward iLQR passes
#
#     GENERAL METHODS
#         rollout!: Compute state trajectory X given controls U
#         cost: Compute the cost
#         calculate_jacobians!: Compute jacobians
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
    X = res.X; U = res.U
    flag = rollout!(X, U, solver)
    if solver.control_integration == :foh
        calculate_derivatives!(res,solver,X,U)
    end
    flag
end

function rollout!(res::SolverVectorResults, solver::Solver)
    X, U = res.X, res.U
    infeasible = solver.model.m != size(U[1],1)
    N = solver.N
    m = solver.model.m
    n = solver.model.n

    X[1] = solver.obj.x0
    for k = 1:N-1
        if solver.control_integration == :foh
            solver.fd(X[k+1], X[k], U[k][1:m], U[k+1][1:m])
        else
            solver.fd(X[k+1], X[k], U[k][1:m])
        end

        if infeasible
            X[k+1] .+= U[k][m+1:m+n]
        end

        # Check that rollout has not diverged
        if ~(maximum(X[k]) < solver.opts.max_state_value || maximum(U[k]) < solver.opts.max_control_value)
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
function rollout!(res::SolverVectorResults,solver::Solver,alpha::Float64)
    infeasible = solver.model.m != size(res.U[1],1)
    N = solver.N; m = solver.model.m; n = solver.model.n

    if infeasible
        m0 = m
        m += n
    end

    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[1] = solver.obj.x0;

    if solver.control_integration == :foh
        b = res.b
        du = zeros(m)
        dv = zeros(m)
        du = alpha*d[1]
        U_[1] .= U[1] + du
    end

    for k = 2:N
        delta = X_[k-1] - X[k-1]

        if solver.control_integration == :foh
            dv .= K[k]*delta + b[k]*du + alpha*d[k]
            U_[k] .= U[k] + dv
            solver.fd(X_[k], X_[k-1], U_[k-1], U_[k])
            du = dv
        else
            U_[k-1] = U[k-1] - K[k-1]*delta - alpha*d[k-1]
            solver.fd(X_[k], X_[k-1], U_[k-1])
        end


        if infeasible
            X_[k] .+= U_[k-1][m0.+(1:n)]
        end

        # Check that rollout has not diverged
        if ~(maximum(X_[k]) < solver.opts.max_state_value || maximum(U_[k]) < solver.opts.max_control_value)
            return false
        end
    end

    # Calculate state derivatives
    if solver.control_integration == :foh
        calculate_derivatives!(res,solver,X_,U_)
    end

    return true
end

"""
$(SIGNATURES)
Quadratic stage cost (with goal state)
"""
function stage_cost(x,u,Q::AbstractArray{Float64,2},R::AbstractArray{Float64,2},xf::Vector{Float64})::Union{Float64,ForwardDiff.Dual}
    0.5*(x - xf)'*Q*(x - xf) + 0.5*u'*R*u
end

function stage_cost(obj::Objective, x::Vector, u::Vector)::Float64
    0.5*(x - obj.xf)'*obj.Q*(x - obj.xf) + 0.5*u'*obj.R*u
end

"""
$(SIGNATURES)
Compute the unconstrained cost
"""
function cost(solver::Solver,vars::DircolVars)
    cost(solver,vars.X,vars.U)
end

function _cost(solver::Solver,res::SolverVectorResults,X=res.X,U=res.U)
    # pull out solver/objective values
    N = solver.N; Q = solver.obj.Q; xf::Vector{Float64} = solver.obj.xf; Qf::Matrix{Float64} = solver.obj.Qf; m = solver.model.m; n = solver.model.n
    obj = solver.obj
    dt = solver.dt

    if size(U,1) != m
        m += n
    end

    R = getR(solver)

    J = 0.0
    for k = 1:N-1
        if solver.control_integration == :foh

            xdot1 = res.xdot[k]
            xdot2 = res.xdot[k+1]

            Xm = 0.5*X[k] + dt/8*xdot1 + 0.5*X[k+1] - dt/8*xdot2
            Um = (U[k] + U[k+1])/2

            J += solver.dt/6*(stage_cost(X[k],U[k],Q,R,xf) + 4*stage_cost(Xm,Um,Q,R,xf) + stage_cost(X[k+1],U[k+1],Q,R,xf)) # Simpson quadrature (integral approximation) for foh stage cost
        else
            J += solver.dt*stage_cost(X[k],U[k],Q,R,xf)
        end
    end

    J += 0.5*(X[N] - xf)'*Qf*(X[N] - xf)

    return J
end

""" $(SIGNATURES) Compute the Constraints Cost """
function cost_constraints(solver::Solver, res::ConstrainedIterResults)
    N = solver.N
    J = 0.0
    for k = 1:N-1
        J += (0.5*res.C[k]'*res.Iμ[k]*res.C[k] + res.LAMBDA[k]'*res.C[k])
    end

    if solver.control_integration == :foh
        J += (0.5*res.C[N]'*res.Iμ[N]*res.C[N] + res.LAMBDA[N]'*res.C[N])
    end

    J += 0.5*res.CN'*res.IμN*res.CN + res.λN'*res.CN

    return J
end

function cost(solver::Solver, res::UnconstrainedIterResults, X::Vector=res.X, U::Vector=res.U)
    _cost(solver,res,X,U)
end

function cost(solver::Solver, res::ConstrainedIterResults, X::Vector=res.X, U::Vector=res.U)
    _cost(solver,res,X,U) + cost_constraints(solver,res)
end

"""
$(SIGNATURES)
Compute the unconstrained cost
"""
function cost(solver::Solver,X::AbstractArray{Float64,2},U::AbstractArray{Float64,2})
    # pull out solver/objective values
    N = solver.N; Q = solver.obj.Q; xf = solver.obj.xf; Qf = solver.obj.Qf; m = solver.model.m; n = solver.model.n
    obj = solver.obj
    dt = solver.dt

    if size(U,1) != m
        m += n
    end

    R = getR(solver)

    J = 0.0
    for k = 1:N-1
        if solver.control_integration == :foh

            xdot1 = zeros(n)
            xdot2 = zeros(n)
            solver.fc(xdot1,X[:,k],U[1:solver.model.m,k])
            solver.fc(xdot2,X[:,k+1],U[1:solver.model.m,k+1])
            #
            # # # #TODO use calculate_derivatives!
            # xdot1 = res.xdot[:,k]
            # xdot2 = res.xdot[:,k+1]

            Xm = 0.5*X[:,k] + dt/8*xdot1 + 0.5*X[:,k+1] - dt/8*xdot2
            Um = (U[:,k] + U[:,k+1])/2

            J += solver.dt/6*(stage_cost(X[:,k],U[:,k],Q,R,xf) + 4*stage_cost(Xm,Um,Q,R,xf) + stage_cost(X[:,k+1],U[:,k+1],Q,R,xf)) # rk3 foh stage cost (integral approximation)
        else
            J += solver.dt*stage_cost(X[:,k],U[:,k],Q,R,xf)
        end
    end

    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)

    return J
end

"""
$(SIGNATURES)
    Calculate state derivatives (xdot)
"""
function calculate_derivatives!(results::SolverVectorResults, solver::Solver, X::Vector, U::Vector)
    n,m,N = get_sizes(solver)
    for k = 1:N
        solver.fc(results.xdot[k],X[k],U[k][1:m])
    end
end

"""
$(SIGNATURES)
Calculate Jacobians prior to the backwards pass
Updates both dyanmics and constraint jacobians, depending on the results type.
"""
function calculate_jacobians!(res::ConstrainedIterResults, solver::Solver)::Nothing #TODO change to inplace '!' notation throughout the code
    N = solver.N
    for k = 1:N-1
        if solver.control_integration == :foh
            res.fx[k], res.fu[k], res.fv[k] = solver.Fd(res.X[k], res.U[k], res.U[k+1])
            res.Ac[k], res.Bc[k] = solver.Fc(res.X[k], res.U[k])
        else
            res.fx[k], res.fu[k] = solver.Fd(res.X[k], res.U[k])
        end
        solver.c_jacobian(res.Cx[k], res.Cu[k], res.X[k],res.U[k])
    end

    if solver.control_integration == :foh
        res.Ac[N], res.Bc[N] = solver.Fc(res.X[N], res.U[N])
        solver.c_jacobian(res.Cx[N], res.Cu[N], res.X[N],res.U[N])
    end

    solver.c_jacobian(res.Cx_N, res.X[N])
    return nothing
end

function calculate_jacobians!(res::UnconstrainedIterResults, solver::Solver, infeasible=false)::Nothing
    N = solver.N
    for k = 1:N-1
        if solver.control_integration == :foh
            res.fx[k], res.fu[k], res.fv[k] = solver.Fd(res.X[k], res.U[k], res.U[k+1])
            res.Ac[k], res.Bc[k] = solver.Fc(res.X[k], res.U[k])
        else
            res.fx[k], res.fu[k] = solver.Fd(res.X[k], res.U[k])
        end
    end
    if solver.control_integration == :foh
        res.Ac[N], res.Bc[N] = solver.Fc(res.X[N], res.U[N])
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
function update_constraints!(res::ConstrainedIterResults, solver::Solver, X::Array=res.X, U::Array=res.U)::Nothing

    N = length(res.C) # note, C is now (p,N)
    p = length(res.C[1])
    c = solver.c_fun
    pI = solver.obj.pI

    if solver.control_integration == :foh
        final_index = N
    else
        final_index = N-1
    end

    for k = 1:final_index
        c(res.C[k], X[k], U[k]) # update results with constraint evaluations
        # Inequality constraints [see equation ref]
        for j = 1:pI
            if res.C[k][j] > 0.0 || res.LAMBDA[k][j] > 0.0
                res.Iμ[k][j,j] = res.MU[k][j] # active (or previously active) inequality constraints are penalized
            else
                res.Iμ[k][j,j] = 0. # inactive inequality constraints are not penalized
            end
        end

        # Equality constraints
        for j = pI+1:p
            res.Iμ[k][j,j] = res.MU[k][j] # equality constraints are penalized
        end
    end

    # Terminal constraint
    c(res.CN,X[N])
    res.IμN .= Diagonal(res.μN)
    return nothing # TODO allow for more general terminal constraint
end

function update_constraints!(res::UnconstrainedIterResults, solver::Solver, X::Array=res.X, U::Array=res.U)::Nothing
    return nothing
end

"""
$(SIGNATURES)
    Count the number of constraints of each type from an objective
"""
function count_constraints(obj::ConstrainedObjective)
    n = size(obj.Q,1)
    p = obj.p # number of constraints
    pI = obj.pI # number of inequality and equality constraints
    pE = p-pI # number of equality constraints

    u_min_active = isfinite.(obj.u_min)
    u_max_active = isfinite.(obj.u_max)
    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    pI_u_max = count(u_max_active)
    pI_u_min = count(u_min_active)
    pI_u = pI_u_max + pI_u_min

    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min

    pI_c = pI - pI_x - pI_u
    pE_c = pE

    p_N = obj.p_N
    pI_N = obj.pI_N
    pE_N = p_N - pI_N
    pI_N_c = pI_N
    if obj.use_terminal_constraint
        pE_N_c = pE_N - n
    else
        pE_N_c = pE_N
    end

    return (pI, pI_c, pI_N, pI_N_c), (pE, pE_c, pE_N, pE_N_c)

end

"""
$(SIGNATURES)
    Generate the Jacobian of a general nonlinear constraint function
        -constraint function must be inplace
        -automatic differentition via ForwardDiff.jl
"""
function generate_general_constraint_jacobian(c::Function,p::Int,p_N::Int,n::Int64,m::Int64)::Function
    c_aug! = f_augmented!(c,n,m)
    J = zeros(p,n+m)
    S = zeros(n+m)
    cdot = zeros(p)
    F(J,cdot,S) = ForwardDiff.jacobian!(J,c_aug!,cdot,S)

    function c_jacobian(fx,fu,x,u)
        S[1:n] = x
        S[n+1:n+m] = u
        F(J,cdot,S)
        fx .= J[1:p,1:n]
        fu .= J[1:p,n+1:n+m]
    end

    if p_N > 0
        J_N = zeros(p_N,n)
        xdot = zeros(p_N)
        F_N(J_N,xdot,x) = ForwardDiff.jacobian!(J_N,c,xdot,x)
        function c_jacobian(fx,x)
            F_N(J_N,xdot,x)
            fx .= J_N
        end
    end

    return c_jacobian
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

    # Key: I=> inequality,   E=> equality
    #     _c=> custom   (lack)=> box constraint
    #     _N=> terminal (lack)=> stage

    pI_obj, pE_obj = count_constraints(obj)
    p = obj.p # number of constraints
    pI, pI_c, pI_N, pI_N_c = pI_obj
    pE, pE_c, pE_N, pE_N_c = pE_obj

    # Mask for active (state|control) constraints
    u_min_active = isfinite.(obj.u_min)
    u_max_active = isfinite.(obj.u_max)
    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    # Inequality on control
    pI_u_max = count(u_max_active)
    pI_u_min = count(u_min_active)
    pI_u = pI_u_max + pI_u_min

    function c_control!(c,x,u)
        c[1:pI_u_max] = (u - obj.u_max)[u_max_active]
        c[pI_u_max+1:pI_u_max+pI_u_min] = (obj.u_min - u)[u_min_active]
    end

    # Inequality on state
    pI_x_max = count(x_max_active)
    pI_x_min = count(x_min_active)
    pI_x = pI_x_max + pI_x_min
    function c_state!(c,x,u)
        c[1:pI_x_max] = (x - obj.x_max )[x_max_active]
        c[pI_x_max+1:pI_x_max+pI_x_min] = (obj.x_min - x)[x_min_active]
    end

    # Custom constraints
    pI_c = pI - pI_x - pI_u

    # Form inequality constraint
    function cI!(c,x,u)
        c_control!(view(c,1:pI_u),x,u)
        c_state!(view(c,(1:pI_x).+pI_u),x,u)
        if pI_c > 0
            obj.cI(view(c,(1:pI_c).+pI_u.+pI_x),x,u)
        end
    end

    # Augment functions together
    function c_fun!(c,x,u)
        infeasible = length(u) != m
        cI!(view(c,1:pI),x,u[1:m])
        if pE_c > 0
            obj.cE(view(c,(1:pE_c).+pI),x,u[1:m])
        end
        if infeasible
            c[pI+pE_c+1:pI+pE_c+n] = u[m+1:m+n]
        end
    end

    # Terminal Constraint
    # TODO make this more general
    function c_fun!(c,x)
        c[1:n] = x - obj.xf
    end

    ### Jacobians ###
    # Explicitely declare known jacobians
    In = Matrix(I,n,n)
    fx_control = zeros(pI_u,n)
    fx_state = zeros(pI_x,n)
    fx_state[1:pI_x_max, :] = In[x_max_active,:]
    fx_state[pI_x_max+1:end,:] = -In[x_min_active,:]
    fx = zeros(p,n)

    Im = Matrix(I,m,m)
    fu_control = zeros(pI_u,m)
    fu_control[1:pI_u_max,:] = Im[u_max_active,:]
    fu_control[pI_u_max+1:end,:] = -Im[u_min_active,:]
    fu_state = zeros(pI_x,m)
    fu = zeros(p,m)

    if pI_c > 0
        cI_custom_jacobian! = generate_general_constraint_jacobian(obj.cI,pI_c,pI_N_c,n,m)
    end
    if pE_c > 0
        cE_custom_jacobian! = generate_general_constraint_jacobian(obj.cE,pE_c,0,n,m)
    end

    fu_infeasible = In
    fx_infeasible = zeros(n,n)

    fx_N = In  # Jacobian of final state

    function constraint_jacobian!(fx::AbstractMatrix, fu::AbstractMatrix, x::AbstractArray,u::AbstractArray)
        infeasible = length(u) != m
        fx[1:pI_u, 1:n] = fx_control
        fu[1:pI_u, 1:m] = fu_control
        fx[(1:pI_x).+pI_u, 1:n] = fx_state
        fu[(1:pI_x).+pI_u, 1:m] = fu_state

        if pI_c > 0
            cI_custom_jacobian!(view(fx,pI_x+pI_u+1:pI_x+pI_u+pI_c,1:n), view(fu,pI_x+pI_u+1:pI_x+pI_u+pI_c,1:m), x, u[1:m])
        end
        if pE_c > 0
            cE_custom_jacobian!(view(fx,pI_x+pI_u+pI_c+1:pI_x+pI_u+pI_c+pE_c,1:n), view(fu,pI_x+pI_u+pI_c+1:pI_x+pI_u+pI_c+pE_c,1:m), x, u[1:m])
        end

        if infeasible
            fx[pI_x+pI_u+pI_c+pE_c+1:pI_x+pI_u+pI_c+pE_c+n,1:n] = fx_infeasible
            fu[pI_x+pI_u+pI_c+pE_c+1:pI_x+pI_u+pI_c+pE_c+n,m+1:m+n] = fu_infeasible
        end
    end

    function constraint_jacobian!(j::AbstractArray,x::AbstractArray)
        j .= fx_N
    end

    return c_fun!, constraint_jacobian!
end

generate_constraint_functions(obj::UnconstrainedObjective) = (x,u)->nothing, (x,u)->nothing

"""
$(SIGNATURES)
Compute the maximum constraint violation. Inactive inequality constraints are
not counted (masked by the Iμ matrix). For speed, the diagonal indices can be
precomputed and passed in.
"""
function max_violation(results::ConstrainedIterResults,inds=CartesianIndex.(axes(results.Iμ,1),axes(results.Iμ,2)))
    if size(results.CN,1) != 0
        return max(maximum(norm.(map((x)->x.>0, results.Iμ) .* results.C, Inf)), maximum(abs.(results.CN)))
    else
        return maximum(norm.(map((x)->x.>0, results.Iμ) .* results.C, Inf))
    end
end

function max_violation(results::UnconstrainedIterResults)
    return 0.0
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
Linear interpolation trajectory between initial and final state(s)
"""
function line_trajectory(solver::Solver, method=:trapezoid)::Array{Float64,2}
    N, = get_N(solver,method)
    line_trajectory(solver.obj.x0,solver.obj.xf,N)
end

function line_trajectory(x0::Array{Float64,1},xf::Array{Float64,1},N::Int64)::Array{Float64,2}
    x_traj = zeros(size(x0,1),N)
    t = range(0,stop=N,length=N)
    slope = (xf-x0)./N
    for i = 1:size(x0,1)
        x_traj[i,:] = slope[i].*t
    end
    x_traj
end

"""
$(SIGNATURES)
    Regularization update scheme
        - see Todorov iLQG code (MATLAB)
"""
function regularization_update!(results::SolverResults,solver::Solver,status::Bool)
    if status # increase regularization
        results.dρ[1] = max(results.dρ[1]*solver.opts.ρ_factor, solver.opts.ρ_factor)
        results.ρ[1] = max(results.ρ[1]*results.dρ[1], solver.opts.ρ_min)
    else # decrease regularization
        results.dρ[1] = min(results.dρ[1]/solver.opts.ρ_factor, 1.0/solver.opts.ρ_factor)
        results.ρ[1] = results.ρ[1]*results.dρ[1]*(results.ρ[1]>solver.opts.ρ_factor)
    end
end
