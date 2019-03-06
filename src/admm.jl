
struct ADMMCost <: CostFunction
    costs::NamedTuple{M,NTuple{N,C}} where {M,N,C <: CostFunction}
    c::Function
    ∇c::Function
    q::Int            # Number of bodies
    b::Vector
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

function taylor_expansion(cost::ADMMCost, x::BlockVector, u::BlockVector, body::Symbol)
    taylor_expansion(cost.costs[body],x[body],u[body])
end

function taylor_expansion(cost::ADMMCost, x::BlockVector, body::Symbol)
    error("implement terminal")
    taylor_expansion(cost.costs[body],x[body])
end



struct ADMMResults #<: ConstrainedIterResults
    bodies::NTuple{N,Symbol} where N

    X::Trajectory  # States (n,N)
    U::Trajectory  # Controls (m,N)

    K::NamedTuple # Feedback (state) gain (m,n,N)
    d::NamedTuple  # Feedforward gain (m,N)

    X_::Trajectory # Predicted states (n,N)
    U_::Trajectory # Predicted controls (m,N)

    S::NamedTuple  # Cost-to-go hessian (n,n)
    s::NamedTuple  # Cost-to-go gradient (n,1)

    fdx::NamedTuple # State jacobian (n,n,N)
    fdu::NamedTuple # Control (k) jacobian (n,m,N-1)

    C::Trajectory      # Constraint values (p,N)
    C_prev::Trajectory # Previous constraint values (p,N)
    Iμ::Trajectory        # fcxtive constraint penalty matrix (p,p,N)
    λ::Trajectory # Lagrange multipliers (p,N)
    μ::Trajectory     # Penalty terms (p,N)

    Cx::Trajectory # State jacobian (n,n,N)
    Cu::Trajectory # Control (k) jacobian (n,m,N-1)

    ρ::Vector{Float64}
    dρ::Vector{Float64}

    bp::NamedTuple

    n::NamedTuple
    m::NamedTuple
end


"""$(SIGNATURES) ADMM Results"""
function ADMMResults(bodies::NTuple{B,Symbol},ns::NTuple{B,Int},ms::NTuple{B,Int},p::Int,N::Int,p_N::Int) where B
    @assert length(bodies) == length(ns) == length(ms)
    p != p_N ? p_N = p : nothing
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


    fdx = NamedTuple{bodies}([[zeros(n,n) for i = 1:N-1] for n in ns])
    fdu = NamedTuple{bodies}([[zeros(n,m) for i = 1:N-1] for (n,m) in zip(ns,ms)])

    # Stage Constraints
    C      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    C_prev = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    Iμ     = [i != N ? Diagonal(ones(p)) : Diagonal(ones(p_N)) for i = 1:N]
    λ      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    μ      = [i != N ? ones(p) : ones(p_N)  for i = 1:N]

    Cx  = [i != N ? zeros(p,n) : zeros(p_N,n)  for i = 1:N]
    Cu  = [i != N ? zeros(p,m) : zeros(p_N,0)  for i = 1:N]

    ρ = zeros(1)
    dρ = zeros(1)

    bp = NamedTuple{bodies}([BackwardPass(n,m,N) for (n,m) in zip(ns,ms)])
    n = NamedTuple{bodies}(ns)
    m = NamedTuple{bodies}(ms.+p)

    ADMMResults(bodies,X,U,K,d,X_,U_,S,s,fdx,fdu,
        C,C_prev,Iμ,λ,μ,Cx,Cu,ρ,dρ,bp,n,m)
end


function backwardpass!(res::ADMMResults,solver::Solver)
    ΔV = []
    for b in res.bodies
        Δv = _backwardpass_admm!(res,solver,b)
        push!(ΔV,Δv)
    end
    return NamedTuple{bodies}(ΔV)
end

function _backwardpass_admm!(res::ADMMResults,solver::Solver,b::Symbol)
    # Get problem sizes
    n = res.n[b]
    m = res.m[b]

    dt = solver.dt

    X = res.X; U = res.U

    K = res.K[b]; d = res.d[b]; S = res.S[b]; s = res.s[b]
    Qx = res.bp[b].Qx; Qu = res.bp[b].Qu; Qxx = res.bp[b].Qxx; Quu = res.bp[b].Quu; Qux = res.bp[b].Qux
    Quu_reg = res.bp[b].Quu_reg; Qux_reg = res.bp[b].Qux_reg

    # Boundary Conditions
    S[N], s[N] = taylor_expansion(solver.obj.cost::ADMMCost, res.X[N]::BlockVector, b::Symbol)

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Backward pass
    k = N-1
    while k >= 1

        Qxx[k],Quu[k],Qux[k],Qx[k],Qu[k] = taylor_expansion(solver.obj.cost::ADMMCost, res.X[k]::BlockVector, res.U[k]::BlockVector, b::Symbol)

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[b][k], res.fdu[b][k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx[k] += fdx'*s[k+1]
        Qu[k] += fdu'*s[k+1]
        Qxx[k] += fdx'*S[k+1]*fdx
        Quu[k] += fdu'*S[k+1]*fdu
        Qux[k] += fdu'*S[k+1]*fdx

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

"""
$(SIGNATURES)
Propagate dynamics with a line search (in-place)
"""
function forwardpass_admm!(res::ADMMResults,solver::Solver,)

end
function forwardpass!(res::SolverIterResults, solver::Solver, Δv::Array,J_prev::Float64)
    # Pull out values from results
    X = res.X; U = res.U; X_ = res.X_; U_ = res.U_

    # update_constraints!(res,solver,X,U)
    # J_prev = cost(solver, res, X, U)

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
        flag = rollout!(res,solver,alpha)

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
