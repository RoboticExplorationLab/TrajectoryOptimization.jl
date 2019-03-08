
struct ADMMCost <: CostFunction
    costs::NamedTuple{M,NTuple{N,C}} where {M,N,C <: CostFunction}
    c::Function
    ∇c::Function
    q::Int              # Number of bodies
    b::Vector
    n::Int            # joint state
    m::Int            # joint control
    part_x::NamedTuple  # Partition for the state vectors
    part_u::NamedTuple  # Partition for the control vectors
end

function stage_cost(cost::ADMMCost, x::BlockVector, u::BlockVector)
    J = 0.0
    for body in keys(cost.costs)
        J += stage_cost(cost.costs[body],x[body],u[body])
    end
    return J
end

function stage_cost(cost::ADMMCost, x::BlockVector)
    J = 0.0
    for body in keys(cost.costs)
        J += stage_cost(cost.costs[body],x[body])
    end
    return J
end


function taylor_expansion(cost::ADMMCost, x::BlockVector, u::BlockVector, body::Symbol)
    taylor_expansion(cost.costs[body],x[body],u[body])
end

function taylor_expansion(cost::ADMMCost, x::BlockVector, body::Symbol)
    taylor_expansion(cost.costs[body],x[body])
end

get_sizes(cost::ADMMCost) = cost.n, cost.m

SolverIterResults <: ConstrainedIterResults
struct ADMMResults <: ConstrainedIterResults
    bodies::NTuple{N,Symbol} where N

    X::Trajectory  # States (n,N)
    U::Trajectory  # Controls (m,N)

    K::NamedTuple # Feedback (state) gain (m,n,N)
    d::NamedTuple  # Feedforward gain (m,N)

    X_::Trajectory # Predicted states (n,N)
    U_::Trajectory # Predicted controls (m,N)

    S::NamedTuple  # Cost-to-go hessian (n,n)
    s::NamedTuple  # Cost-to-go gradient (n,1)

    fdx::Trajectory # State jacobian (n,n,N)
    fdu::Trajectory # Control (k) jacobian (n,m,N-1)

    C::Trajectory      # Constraint values (p,N)
    C_prev::Trajectory # Previous constraint values (p,N)
    Iμ::Trajectory        # fcxtive constraint penalty matrix (p,p,N)
    λ::Trajectory # Lagrange multipliers (p,N)
    μ::Trajectory     # Penalty terms (p,N)

    Cx::Trajectory # State jacobian (n,n,N)
    Cu::Trajectory # Control (k) jacobian (n,m,N-1)

    active_set::Trajectory

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    bp::NamedTuple

    n::NamedTuple
    m::NamedTuple
end


"""$(SIGNATURES) ADMM Results"""
function ADMMResults(bodies::NTuple{B,Symbol},ns::NTuple{B,Int},ms::NTuple{B,Int},p::Int,N::Int,p_N::Int) where B
    @assert length(bodies) == length(ns) == length(ms)
    part_x = create_partition(ns,bodies)
    part_u = create_partition(ms,bodies)
    n = sum(ns)
    m = sum(ms)

    X  = [BlockArray(zeros(n),part_x)   for i = 1:N]
    U  = [BlockArray(zeros(m),part_u)   for i = 1:N-1]

    K  = NamedTuple{bodies}([[zeros(m,n) for i = 1:N-1] for (n,m) in zip(ns,ms)])
    d  =  NamedTuple{bodies}([[zeros(m)   for i = 1:N-1] for m in ms])

    X_  = [BlockArray(zeros(n),part_x)   for i = 1:N]
    U_  = [BlockArray(zeros(m),part_u)   for i = 1:N-1]

    S  =  NamedTuple{bodies}([[zeros(n,n) for i = 1:N] for n in ns])
    s  =  NamedTuple{bodies}([[zeros(n)   for i = 1:N] for n in ns])

    part_xx = NamedTuple{bodies}([(r,r) for r in values(part_x)])
    part_xu = NamedTuple{bodies}([(r1,r2) for (r1,r2) in zip(values(part_x),values(part_u))])
    fdx = [BlockArray(zeros(n,n), part_xx) for i = 1:N-1]
    fdu = [BlockArray(zeros(n,m), part_xu) for i = 1:N-1]

    # Stage Constraints
    C      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    C_prev = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    Iμ     = [i != N ? Diagonal(ones(p)) : Diagonal(ones(p_N)) for i = 1:N]
    λ      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    μ      = [i != N ? ones(p) : ones(p_N)  for i = 1:N]

    part_cx = NamedTuple{bodies}([(1:p,rng) for rng in values(part_x)])
    part_cu = NamedTuple{bodies}([(1:p,rng) for rng in values(part_u)])
    part_cx_N = NamedTuple{bodies}([(1:p_N,rng) for rng in values(part_x)])
    Cx  = [i != N ? BlockArray(zeros(p,n),part_cx) : BlockArray(zeros(p_N,n),part_cx_N)  for i = 1:N]
    Cu  = [BlockArray(zeros(p,m),part_cu) for i = 1:N-1]

    active_set = [i != N ? ones(Bool,p) : ones(Bool,p_N)  for i = 1:N]

    ρ = zeros(1)
    dρ = zeros(1)

    bp = NamedTuple{bodies}([BackwardPass(n,m,N) for (n,m) in zip(ns,ms)])
    n = NamedTuple{bodies}(ns)
    m = NamedTuple{bodies}(ms)

    ADMMResults(bodies,X,U,K,d,X_,U_,S,s,fdx,fdu,
        C,C_prev,Iμ,λ,μ,Cx,Cu,active_set,ρ,dρ,bp,n,m)
end

function _cost(solver::Solver{M,Obj},res::ADMMResults,X=res.X,U=res.U) where {M, Obj<:Objective}
    # pull out solver/objective values
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    costfun = solver.obj.cost
    dt = solver.dt

    J = 0.0
    for k = 1:N-1
        # Get dt if minimum time
        solver.state.minimum_time ? dt = U[k][m̄]^2 : nothing

        # Stage cost
        J += (stage_cost(costfun,X[k],U[k]))*dt

    end

    # Terminal Cost
    J += stage_cost(costfun, X[N])

    return J
end

function generate_constraint_functions(obj::ConstrainedObjective{ADMMCost}; max_dt::Float64=1.0, min_dt::Float64=1e-2)
    costfun = obj.cost
    @assert obj.pI == obj.pI_N == 0
    c_function!(c,x,u) = obj.cE(c,x,u)
    c_function!(c,x) = obj.cE_N(c,x)
    c_labels = ["custom equality" for i = 1:obj.p]
    # c_jacobian(cx,cu,x,u) = obj.∇cE(cx,cu,x,u)
    # c_jacobian(cx,x) = copyto!(cx,Diagonal(I,length(x)))

    return c_function!, obj.∇cE, c_labels
end

get_active_set!(results::ADMMResults,solver::Solver,p::Int,pI::Int,k::Int) = nothing

function _backwardpass_admm!(res::ADMMResults,solver::Solver,b::Symbol)
    # Get problem sizes
    n = res.n[b]
    m = res.m[b]
    N = solver.N
    dt = solver.dt

    X = res.X; U = res.U

    K = res.K[b]; d = res.d[b]; S = res.S[b]; s = res.s[b]
    Qx = res.bp[b].Qx; Qu = res.bp[b].Qu; Qxx = res.bp[b].Qxx; Quu = res.bp[b].Quu; Qux = res.bp[b].Qux
    Quu_reg = res.bp[b].Quu_reg; Qux_reg = res.bp[b].Qux_reg

    # Boundary Conditions
    S[N], s[N] = taylor_expansion(solver.obj.cost::ADMMCost, res.X[N]::BlockVector, b::Symbol)
    S[N] += res.Cx[N][b]'*res.Iμ[N]*res.Cx[N][b]
    s[N] += res.Cx[N][b]'*(res.Iμ[N]*res.C[N] + res.λ[N])

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Backward pass
    k = N-1
    while k >= 1

        Qxx[k],Quu[k],Qux[k],Qx[k],Qu[k] = taylor_expansion(solver.obj.cost::ADMMCost, res.X[k]::BlockVector, res.U[k]::BlockVector, b::Symbol) .* dt

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k][b], res.fdu[k][b]
        C, Cx, Cu = res.C[k],res.Cx[k][b], res.Cu[k][b]
        λ, Iμ = res.λ[k], res.Iμ[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx[k] += fdx'*s[k+1]
        Qu[k] += fdu'*s[k+1]
        Qxx[k] += fdx'*S[k+1]*fdx
        Quu[k] += fdu'*S[k+1]*fdu
        Qux[k] += fdu'*S[k+1]*fdx

        Qx[k] += Cx'*(Iμ*C + λ)
        Qu[k] += Cu'*(Iμ*C + λ)
        Qxx[k] += Cx'*Iμ*Cx
        Quu[k] += Cu'*Iμ*Cu
        Qux[k] += Cu'*Iμ*Cx

        if solver.opts.bp_reg_type == :state
            Quu_reg[k] = Quu[k] + res.ρ[1]*fdu'*fdu
            Qux_reg[k] = Qux[k] + res.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            Quu_reg[k] = Quu[k] + res.ρ[1]*I
            Qux_reg[k] = Qux[k]
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg[k])))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            @logmsg InnerIters "Regularizing Quu "
            regularization_update!(res,solver,:increase)

            # reset backward pass
            k = N-1
            Δv[1] = 0.
            Δv[2] = 0.
            continue
        end

        # Compute gains
        K[k] = -Quu_reg[k]\Qux_reg[k]
        d[k] = -Quu_reg[k]\Qu[k]

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = Qx[k] + K[k]'*Quu[k]*d[k] + K[k]'*Qu[k] + Qux[k]'*d[k]
        S[k] = Qxx[k] + K[k]'*Quu[k]*K[k] + K[k]'*Qux[k] + Qux[k]'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # calculated change is cost-to-go over entire trajectory
        Δv[1] += d[k]'*Qu[k]
        Δv[2] += 0.5*d[k]'*Quu[k]*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

function forwardpass!(res::ADMMResults, solver::Solver, Δv::Array,J_prev::Float64,b::Symbol)
    # Pull out values from results
    X = res.X; U = res.U; X_ = res.X_; U_ = res.U_

    update_constraints!(res,solver,X,U)
    J_prev = cost(solver, res, X, U)

    J = Inf
    alpha = 1.0
    iter = 0
    z = -1.
    expected = 0.

    logger = current_logger()
    # print_header(logger,InnerIters) #TODO: fix, this errored out
    @logmsg InnerIters :iter value=0
    @logmsg InnerIters :cost value=J_prev
    # print_row(logger,InnerIters) #TODO: fix, same issue
    while (z ≤ solver.opts.line_search_lower_bound || z > solver.opts.line_search_upper_bound) && J >= J_prev

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            # set trajectories to original trajectory
            copyto!(X_,X)
            copyto!(U_,U)

            update_constraints!(res,solver,X_,U_)
            J = cost(solver, res, X_, U_)

            z = 0.
            alpha = 0.0
            expected = 0.

            @logmsg InnerLoop "Max iterations (forward pass)"
            regularization_update!(res,solver,:increase) # increase regularization
            res.ρ[1] += solver.opts.bp_reg_fp
            break
        end

        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout!(res,solver,alpha,b)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            @logmsg InnerIters "Non-finite values in rollout"
            iter += 1
            alpha /= 2.0
            continue
        end

        # Calcuate cost
        J = cost(solver, res, X_, U_)   # Unconstrained cost

        expected = -alpha*(Δv[1] + alpha*Δv[2])
        if expected > 0
            z  = (J_prev - J)/expected
        else
            @logmsg InnerIters "Non-positive expected decrease"
            z = -1
        end

        iter += 1
        alpha /= 2.0

        # Log messages
        @logmsg InnerIters :iter value=iter
        @logmsg InnerIters :α value=2*alpha
        @logmsg InnerIters :cost value=J
        @logmsg InnerIters :z value=z
        # print_row(logger,InnerIters)

    end  # forward pass loop

    if res isa ConstrainedIterResults
        @logmsg InnerLoop :c_max value=max_violation(res)
    end
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ value=J_prev-J
    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*alpha
    @logmsg InnerLoop :ρ value=res.ρ[1]

    if J > J_prev
        error("Error: Cost increased during Forward Pass")
    end

    return J
end

function admm_plot(res)
    X1 = to_array(res.X)[part_x.a1,:]
    X2 = to_array(res.X)[part_x.m,:]
    for k = 1:length(res.X)
        p = plot(xlim=(-1,11),ylim=(-5,5),aspectratio=:equal)
        plot!([X1[1,k],X2[1,k]],[X1[2,k],X2[2,k]],color=:red)
        scatter!([X1[1,k]],[X1[2,k]],color=:green,markersize=10)
        scatter!([X2[1,k]],[X2[2,k]],color=:black)
        display(p)
    end
end

function initial_admm_rollout!(solver::Solver,res::ADMMResults,U0)
    for k = 1:solver.N-1
        res.U[k] .= U0[:,k]
    end

    for b in res.bodies
        rollout!(res,solver,0.0,b)
    end
    copyto!(res.X,res.X_);
    copyto!(res.U,res.U_);

    J = cost(solver,res)
    return J
end


function rollout!(res::ADMMResults,solver::Solver,alpha::Float64,b::Symbol)

    dt = solver.dt

    X = res.X; U = res.U;
    X_ = res.X_; U_ = res.U_

    K = res.K[b]; d = res.d[b]

    X_[1] .= solver.obj.x0;

    for k = 2:solver.N
        # Calculate state trajectory difference
        δx = X_[k-1][b] - X[k-1][b]

        # Calculate updated control
        copyto!(U_[k-1][b], U[k-1][b] + K[k-1]*δx + alpha*d[k-1])

        # Propagate dynamics
        solver.fd(X_[k], X_[k-1], U_[k-1], dt)

        # Check that rollout has not diverged
        if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    # Update constraints
    update_constraints!(res,solver,X_,U_)

    return true
end
