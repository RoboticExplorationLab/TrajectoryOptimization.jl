
"$(TYPEDEF) ALTRO cost, potentially including infeasible start and minimum time costs"
struct ALTROCost{T} <: CostFunction
    cost::AugmentedLagrangianCost
    R_inf::T
    R_min_time::T
    n::Int # state dimension of original problem
    m::Int # input dimension of original problem
end

function ALTROCost(prob::Problem,cost::AugmentedLagrangianCost,R_inf::T,R_min_time::T) where T
    n = prob.model.n
    ALTROCost(cost,R_inf*Matrix(I,n),R_min_time,n,prob.model.m)
end

function get_sizes(cost::ALTROCost)
    n = cost.n
    m = cost.m

    if cost.R_min_time != NaN
        m̄ = m + 1
        n̄ = n + 1
    else
        m̄ = m
        n̄ = n
    end

    return n,m,n̄,m̄
end

#NOTE don't use these...
function stage_cost(cost::ALTROCost{T}, x::AbstractVector{T}, u::AbstractVector{T}, k::Int) where T
    n,m,n̄,m̄ = get_sizes(cost)

    J = 0.0

    # if cost.R_min_time != NaN
    #     J += cost.R_min_time*u[m̄]^2
    #     dt =
    # end
    #
    # if cost.R_inf!= NaN
    #     u_inf = u[m̄ .+(1:n)]
    #     J += 0.5*cost.R_inf*u_inf'*u_inf
    # end
    #
    # J = stage_cost(cost.cost.cost,x[1:n],u[1:m],k) # stage cost only for original x, u
    # J += stage_constraint_cost(cost.cost,x,u,k) # constraints consider


    J
end

function stage_cost(cost::ALTROCost{T}, x::AbstractVector{T}) where T
    # J0 = stage_cost(cost.cost.cost,x[1:cost.n])
    # J0 + stage_constraint_cost(cost.cost,x)
    0.0
end

"ALTRO cost for X and U trajectories"
function cost(cost::ALTROCost{T},X::VectorTrajectory{T},U::VectorTrajectory{T},dt::T) where T <: AbstractFloat
    N = length(X)
    n,m,n̄,m̄ = get_sizes(cost)

    update_constraints!(cost.cost.C,cost.cost.constraints,X,U)
    update_active_set!(cost.cost.active_set,cost.cost.C,cost.cost.λ)

    J = 0.0

    for k = 1:N-1
        # Minimum time stage cost
        if cost.R_min_time != NaN
            dt = U[k][m̄]^2
            J += cost.R_min_time*dt
        end

        # Infeasible start stage cost
        if cost.R_inf != NaN
            u_inf = u[m̄ .+(1:n)]
            J += 0.5*cost.R_inf*u_inf'*u_inf
        end

        J += cost(cost.cost.cost,X[k][1:n],U[k][1:m],k)*dt
        J += stage_constraint_cost(cost.cost,X[k],U[k],k)
    end

    J += cost(cost.cost.cost,X[N])
    J += stage_constraint_cost(cost.cost,X[N])

    return J
end

"Second-order expansion of ALTRO cost"
function cost_expansion(cost::ALTROCost{T},
        x::AbstractVector{T},u::AbstractVector{T}, k::Int) where T
    #
    # expansion = cost_expansion(cost.cost.cost,x,u)
    # Qxx[k][1:n,1:n],Quu[k][1:m,1:m],Qux[k][1:m,1:n],Qx[k][1:n],Qu[k][1:m] = expansion .* dt
    #
    # # Minimum time expansion components
    # if solver.state.minimum_time
    #     ℓ1 = stage_cost(costfun,x,u)
    #     h = U[k][m̄]
    #     tmp = 2*h*expansion[5]
    #
    #     Qu[k][m̄] = h*(2*ℓ1 + R_minimum_time)
    #     Quu[k][1:m,m̄] = tmp
    #     Quu[k][m̄,1:m] = tmp'
    #     Quu[k][m̄,m̄] = (2*ℓ1 + R_minimum_time)
    #     Qux[k][m̄,1:n] = 2*h*expansion[4]'
    #
    #     Qx[k][n̄] = R_minimum_time*X[k][n̄]
    #     Qxx[k][n̄,n̄] = R_minimum_time
    # end
    #
    # # Infeasible expansion components
    # if solver.state.infeasible
    #     Qu[k][m̄+1:mm] = R_infeasible*U[k][m̄+1:m̄+n]
    #     Quu[k][m̄+1:mm,m̄+1:mm] = R_infeasible
    # end

    return Q,R,H,q,r
end

function cost_expansion(cost::ALTROCost{T},x::AbstractVector{T}) where T


    return Qf,qf
end
