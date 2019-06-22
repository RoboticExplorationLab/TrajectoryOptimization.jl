using ForwardDiff, Plots, LinearAlgebra, BenchmarkTools,
    PartedArrays, TrajectoryOptimization
    # MatrixCalculus

"Time-varying Linear Quadratic Regulator; returns optimal linear feedback matrices and optimal cost-to-go"
function tvlqr_dis(prob::Problem{T},Q::AbstractArray{T},R::AbstractArray{T},Qf::AbstractArray{T}) where T
    n = prob.model.n; m = prob.model.m; N = prob.N
    dt = prob.dt

    K  = [zeros(T,m,n) for k = 1:N-1]
    ∇F = [PartedArray(zeros(T,n,length(prob.model)),create_partition2(prob.model)) for k = 1:N-1]
    P  = [zeros(T,n,n) for k = 1:N]

    P[N] .= Qf

    for k = N-1:-1:1
        jacobian!(∇F[k],prob.model,prob.X[k],prob.U[k],prob.dt)

        A, B = ∇F[k].xx, ∇F[k].xu
        K[k] .= (R*dt + B'*P[k+1]*B)\(B'*P[k+1]*A)
        P[k] .= Q*dt + K[k]'*R*K[k]*dt + (A - B*K[k])'*P[k+1]*(A - B*K[k])
        P[k] .= 0.5*(P[k] + P[k]')
    end

    return K, P
end

function tvlqr_dis_uncertain(prob::Problem{T},Q::AbstractArray{T},R::AbstractArray{T},Qf::AbstractArray{T}) where T
    n = prob.model.n; m = prob.model.m; N = prob.N
    dt = prob.dt

    K  = [zeros(T,m,n) for k = 1:N-1]
    Ad = [zeros(T,n,n) for k = 1:N-1]
    Bd = [zeros(T,n,m) for k = 1:N-1]
    Gd = [zeros(T,n,r) for k = 1:N-1]
    P  = [zeros(T,n,n) for k = 1:N]

    P[N] .= Qf
    fd = prob.model.f
    f(ẋ,z) = fd(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
    ∇f(x,u,w) = ∇f([x;u;w])

    for k = N-1:-1:1
        Fd = ∇f(prob.X[k],prob.U[k],zeros(r))

        Ad[k] = Fd[:,1:n]
        Bd[k] = Fd[:,n .+ (1:m)]
        Gd[k] = Fd[:,(n+m) .+ (1:r)]
        K[k] .= (R*dt + Bd[k]'*P[k+1]*Bd[k])\(Bd[k]'*P[k+1]*Ad[k])
        P[k] .= Q*dt + K[k]'*R*K[k]*dt + (Ad[k] - Bd[k]*K[k])'*P[k+1]*(Ad[k] - Bd[k]*K[k])
        P[k] .= 0.5*(P[k] + P[k]')
    end

    return K, P, Ad, Bd, Gd
end

function tvlqr_con(prob::Problem{T,Discrete},Qr::AbstractArray{T},Rr::AbstractArray{T},
        Qfr::AbstractArray{T},xf::AbstractVector{T},integrator::Symbol) where T

    n = prob.model.n; m = prob.model.m; r = prob.model.r; N = prob.N

    K = [zeros(T,m,n) for k = 1:N-1]
    P = [zeros(T,n,n) for k = 1:N]
    Ps = [zeros(T,n,n) for k = 1:N] # sqrt
    P[N] = Qfr
    Ps[N] = cholesky(Qfr).U

    fc = prob.model.info[:fc]

    # if model_type == :uncertain
    f(ẋ,z) = fc(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
    ∇f(x,u,w) = ∇f([x;u;w])
    # else
    # f(ẋ,z) = fc(ẋ,z[1:n],z[n .+ (1:m)])
    # ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
    # ∇f(x,u) = ∇f([x;u])
    # end

    function robust_dynamics(ż,z,u)
        w = zeros(r)
        x = z[1:n]
        S = reshape(z[n .+ (1:n^2)],n,n)
        ss = inv(S')
        _P = S*S'

        Zc = ∇f([x;u;w])
        Ac = Zc[:,1:n]
        Bc = Zc[:,n .+ (1:m)]

        Kc = Rr\(Bc'*_P)
        Acl = Ac - Bc*Kc

        f(view(ż,1:n),[x;u;w])
        ż[n .+ (1:n^2)] = reshape(-.5*Qr*ss - Ac'*S + .5*_P*Bc*(Rr\(Bc'*S)),n^2)
    end

    Z = [zeros(n+n^2) for k = 1:N]
    Z[N] = [prob.X[end];vec(cholesky(Qfr).U)]

    for k = N-1:-1:1
        _u = prob.U[k]

        function dyn(p,w,t)
            ṗ = zero(p)
            robust_dynamics(ṗ,p,_u)
            return ṗ
        end

        _tf = dt*k
        _t0 = dt*(k-1)

        u0=vec(Z[k+1])
        tspan = (_tf,_t0)
        pro = ODEProblem(dyn,u0,tspan)
        sol = OrdinaryDiffEq.solve(pro,eval(integrator)(),dt=dt)
        Z[k] = sol.u[end]

        Ps[k] = reshape(Z[k][n .+ (1:n^2)],n,n)
        P[k] = Ps[k]*Ps[k]'
        Bc = ∇f(Z[k][1:n],_u)[:,n .+ (1:m)]
        K[k] = Rr\(Bc'*P[k])
    end

    return K,P,Ps
end

function tvlqr_con_uncertain(prob::Problem{T,Discrete},Qr::AbstractArray{T},Rr::AbstractArray{T},
        Qfr::AbstractArray{T},xf::AbstractVector{T},integrator::Symbol) where T

    n = prob.model.n; m = prob.model.m; r = prob.model.r; N = prob.N

    K = [zeros(T,m,n) for k = 1:N-1]
    P = [zeros(T,n,n) for k = 1:N]
    Ps = [zeros(T,n,n) for k = 1:N] # sqrt riccati
    Āc = [zeros(T,n,n) for k = 1:N-1]
    B̄c = [zeros(T,n,m) for k = 1:N-1]
    Ḡc = [zeros(T,n,r) for k = 1:N-1]

    P[N] = Qfr
    Ps[N] = cholesky(Qfr).U

    fc = prob.model.info[:fc]

    f(ẋ,z) = fc(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
    ∇f(x,u,w) = ∇f([x;u;w])

    function robust_dynamics(ż,z,u)
        w = zeros(r)
        x = z[1:n]
        S = reshape(z[n .+ (1:n^2)],n,n)
        ss = inv(S')
        _P = S*S'

        Zc = ∇f([x;u;w])
        Ac = Zc[:,1:n]
        Bc = Zc[:,n .+ (1:m)]

        Kc = Rr\(Bc'*_P)
        Acl = Ac - Bc*Kc

        f(view(ż,1:n),[x;u;w])
        ż[n .+ (1:n^2)] = reshape(-.5*Qr*ss - Ac'*S + .5*_P*Bc*(Rr\(Bc'*S)),n^2)
    end

    Z = [zeros(n+n^2) for k = 1:N]
    Z[N] = [prob.X[end];vec(cholesky(Qfr).U)]

    for k = N-1:-1:1
        _u = prob.U[k]

        function dyn(p,w,t)
            ṗ = zero(p)
            robust_dynamics(ṗ,p,_u)
            return ṗ
        end

        _tf = dt*k
        _t0 = dt*(k-1)

        u0=vec(Z[k+1])
        tspan = (_tf,_t0)
        pro = ODEProblem(dyn,u0,tspan)
        sol = OrdinaryDiffEq.solve(pro,eval(integrator)(),dt=dt)
        Z[k] = sol.u[end]

        Ps[k] = reshape(Z[k][n .+ (1:n^2)],n,n)
        P[k] = Ps[k]*Ps[k]'
        Fc = ∇f(Z[k][1:n],_u,zeros(r))
        Āc[k] = Fc[:,1:n]
        B̄c[k] = Fc[:,n .+ (1:m)]
        Ḡc[k] = Fc[:,(n+m) .+ (1:r)]
        K[k] = Rr\(B̄c[k]'*P[k])
    end

    return K,P,Ps,Āc,B̄c,Ḡc
end

function robust_problem(prob::Problem{T},E1::AbstractArray{T},
    H1::AbstractArray{T},D::AbstractArray{T},Qr::AbstractArray{T},Rr::AbstractArray{T},Qfr::AbstractArray{T},Q::AbstractArray{T},R::AbstractArray{T},
    Qf::AbstractArray{T},xf::AbstractVector{T}) where T
    N = prob.N
    @assert all([prob.obj[k] isa QuadraticCost for k = 1:N]) #TODO generic cost

    n = prob.model.n; m = prob.model.m; r = prob.model.r
    n_robust = 2*n^2 + n*r # number of robust parameters in state vector
    n̄ = n + n_robust
    m1 = m + n^2
    idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)),z=(1:n̄))

    # get nominal trajectories
    rollout!(prob)
    if String(prob.model.info[:integration])[1:6] == "DiffEq"
        integrator = Symbol(split(String(prob.model.info[:integration]),"_")[2])
    else
        @info "Defaulting Riccati solve to Implicit Midpoint"
        integrator = :ImplicitMidpoint
        # error("Must specific DiffEq integrator/solver (for now...)")
    end
    # _K, _P, _S, _A, _B, _G = tvlqr_con_uncertain(prob,Qr,Rr,Qfr,xf,integrator)
    _K, _P, _A, _B, _G = tvlqr_dis_uncertain(prob,Qr,Rr,Qfr)

    _S = [vec(cholesky(_P[k]).U) for k = 1:N]
    S1 = vec(_S[1])
    # S1 = vec(cholesky(_P[1]).U)
    SN = vec(cholesky(Qfr).U)

    _Z = [zeros(n̄) for k = 1:N]
    _E = [zeros(n,n) for k = 1:N]
    _H = [zeros(n,r) for k = 1:N]

    _E[1] = E1
    _H[1] = H1

    _Z[1] = [prob.X[1];vec(_E[1]);vec(_H[1]);vec(S1)]

    for k = 1:N-1
        Acl = _A[k] - _B[k]*_K[k]
        _E[k+1] = Acl*_E[k]*Acl' + _G[k]*_H[k]'*Acl' + Acl*_H[k]*_G[k]' + _G[k]*D*_G[k]'
        _H[k+1] = Acl*_H[k] + _G[k]*D

        _Z[k+1] = [prob.X[k+1];vec(_E[k+1]);vec(_H[k+1]);vec(_S[k+1])]
    end



    # generate optimal feedback matrix function
    f(ẋ,z) = prob.model.info[:fc](ẋ,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r))
    ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
    ∇f(x,u) = ∇f([x;u])

    function K(z,u)
        x = z[idx.x]
        s = z[idx.s]
        P = reshape(s,n,n)*reshape(s,n,n)'
        Bc = ∇f(x,u)[:,n .+ (1:m)]
        R\(Bc'*P)
    end

    # modify cost
    _cost = CostFunction[]
    for k = 1:N-1
        cost_robust = copy(prob.obj[k])
        cost_robust.Q = cat(cost_robust.Q,Diagonal(zeros(n_robust)),dims=(1,2))
        cost_robust.q = [cost_robust.q; zeros(n_robust)]
        cost_robust.H = [cost_robust.H zeros(m,n_robust)]

        # quadratic cost on riccati states
        for j = (n̄-(n^2 -1)):n̄
            cost_robust.Q[j,j] = 1.0
        end

        if k == 1
            cost_robust.R = cat(cost_robust.R,Diagonal(zeros(n^2)),dims=(1,2))
            cost_robust.r = [cost_robust.r; zeros(n^2)]
            cost_robust.H = [cost_robust.H; zeros(n^2,n̄)]
        end
        push!(_cost,cost_robust)
    end

    cost_robust = copy(prob.obj[N])
    cost_robust.Qf = cat(cost_robust.Qf,Diagonal(zeros(n_robust)),dims=(1,2))
    cost_robust.qf = [cost_robust.qf; zeros(n_robust)]

    # quadratic cost on riccati states
    for j = (n̄-(n^2 -1)):n̄
        cost_robust.Qf[j,j] = 1.0
    end

    push!(_cost,cost_robust)

    # create robust objective
    _obj = Objective(_cost)

    ∇sc, ∇²sc, ∇sc_term, ∇²sc_term = gen_robust_exp_funcs(prob.model.info[:fc],idx,Qr,Rr,Qfr,n,m,r)
    _robust_cost = RobustCost(Qr,Rr,Qfr,Q,R,Qf,K,n,m,r,idx,∇sc,∇²sc,∇sc_term,∇²sc_term)
    robust_obj = RobustObjective(_obj,_robust_cost)

    # create robust model
    _robust_model = robust_model(prob.model,Q,R,D)

    constrained = is_constrained(prob)
    con_prob = ConstraintSet[]

    function sN(c,z)
        c[1:n^2] = z[idx.s] - SN
    end

    function ∇sN(C,z)
        C[:,(n+n^2+n*r) .+ (1:n^2)] = 1.0*Diagonal(ones(n^2))
    end

    ctg_term = Constraint{Equality}(sN,∇sN,n^2,:ctg_term,[collect(1:n̄),collect(1:0)],:terminal)

    for k = 1:N-1
        con_uncertain = GeneralConstraint[]
        if constrained
            for cc in prob.constraints[k]
                push!(con_uncertain,robust_constraint(cc,K,idx,prob.model.n,prob.model.m,n̄))
            end
        end
        push!(con_prob,con_uncertain)
    end

    con_uncertain = GeneralConstraint[]
    push!(con_uncertain,ctg_term)
    if constrained
        for cc in prob.constraints[N]
            push!(con_uncertain,robust_constraint(cc,prob.model.n,n̄))
        end
    end
    push!(con_prob,con_uncertain)

    prob_con = ProblemConstraints(con_prob)

    update_problem(prob,model=_robust_model,obj=robust_obj,
        constraints=prob_con,
        X=_Z,U=[k==1 ? [prob.U[k];S1] : prob.U[k] for k = 1:N-1],
        x0=[copy(prob.x0);reshape(E1,n^2);reshape(H1,n*r);S1])
end

"Robust model from model"
function robust_model(model::Model{Uncertain,Discrete},Q::AbstractArray{T},R::AbstractArray{T},D::AbstractArray) where T
    n = model.n; m = model.m; r = model.r

    fc! = prob.model.info[:fc]
    integration = prob.model.info[:integration]

    discretize_model(robust_model(fc!,Q,R,D,n,m,r),integration,prob.dt)
end

function robust_model(model::Model{Uncertain,Continuous},Q::AbstractArray{T},R::AbstractArray{T},D::AbstractArray) where T
    n = model.n; m = model.m; r = model.r

    robust_model(model.f,Q,R,D,n,m,r)
end

"Robust dynamics, includes: nominal, disturbance, and cost-to-go dynamics"
function robust_model(f::Function,Q::AbstractArray{T},R::AbstractArray{T},D::AbstractArray{T},n::Int,m::Int,r::Int) where T
    idx = (x=1:n,u=1:m,w=1:r)
    z_idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)))

    n̄ = n + n^2 + n*r + n^2

    f_aug(ẋ,z) = f(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    ∇f(z) = ForwardDiff.jacobian(f_aug,zeros(eltype(z),n),z)

    function robust_dynamics(ż,z,u,w)
        x = z[z_idx.x]
        u = u[1:m]
        E = reshape(z[z_idx.e],n,n)
        H = reshape(z[z_idx.h],n,r)
        S = reshape(z[z_idx.s],n,n)
        ss = inv(S')
        P = S*S'

        Zc = ∇f([x;u;w])
        Ac = Zc[:,idx.x]
        Bc = Zc[:,n .+ idx.u]
        Gc = Zc[:,(n+m) .+ idx.w]

        Kc = R\(Bc'*P)
        Acl = Ac - Bc*Kc

        f(view(ż,z_idx.x),x,u,w)
        ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
        ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
        ż[z_idx.s] = reshape(-.5*Q*ss - Ac'*S + .5*P*Bc*(R\(Bc'*S)),n^2)
    end

    UncertainModel(robust_dynamics, n̄, m, r)
end

mutable struct RobustCost{T} <: CostFunction
    Qr::AbstractArray{T}
    Rr::AbstractArray{T}
    Qrf::AbstractArray{T}

    Q::AbstractArray{T}
    R::AbstractArray{T}
    Qf::AbstractArray{T}

    K::Function

    n::Int
    m::Int
    r::Int
    idx::NamedTuple

    ∇c_stage::Function
    ∇²c_stage::Function
    ∇c_term::Function
    ∇²c_term::Function
end

function cost(c::RobustCost,Z::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(Z)
    ℓ = 0.
    for k = 1:N-1
        ℓ += stage_cost(c,Z[k],U[k])
    end
    ℓ += stage_cost(c,Z[N])
    ℓ
end

struct RobustObjective <: AbstractObjective
    obj::AbstractObjective
    robust_cost::RobustCost
end

function stage_cost(cost::RobustCost, z::Vector{T}, u::Vector{T}) where T
    idx = cost.idx; n = cost.n; m = cost.m; r = cost.r

    x = z[idx.x]
    E = reshape(z[idx.e],n,n)

    Kc = cost.K(z,u)

    tr((cost.Qr + Kc'*cost.Rr*Kc)*E)
end

function stage_cost(cost::RobustCost, zN::Vector{T}) where T
    idx = cost.idx; n = cost.n
    E = reshape(zN[idx.e],n,n)
    tr(cost.Qrf*E)
end

function cost(robust_obj::RobustObjective, X::VectorTrajectory{T}, U::VectorTrajectory{T})::T where T <: AbstractFloat
    J = cost(robust_obj.robust_cost,X,U)
    J += cost(robust_obj.obj,X,U)
    return J
end

function cost_expansion!(Q::ExpansionTrajectory{T},robust_obj::RobustObjective,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    cost_expansion!(Q,robust_obj.robust_cost,X,U)
    cost_expansion!(Q,robust_obj.obj,X,U)
end

function cost_expansion!(Q::ExpansionTrajectory{T},robust_cost::RobustCost,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(X); n = robust_cost.n; m = robust_cost.m; r = robust_cost.r
    idx = robust_cost.idx
    n̄ = n + n^2 + n*r + n^2
    for k = 1:N-1
        y = [X[k];U[k]]
        ∇c = robust_cost.∇c_stage(y)
        ∇²c = robust_cost.∇²c_stage(y)
        Q[k].x .+= ∇c[idx.z]
        Q[k].u .+= ∇c[n̄ .+ (1:m)]
        Q[k].xx .+= ∇²c[idx.z,idx.z]
        Q[k].uu .+= ∇²c[n̄ .+ (1:m), n̄ .+ (1:m)]
        Q[k].ux .+= ∇²c[idx.z, n̄ .+ (1:m)]'
    end
    Q[N].x .+= robust_cost.∇c_term(X[N])
    Q[N].xx .+= robust_cost.∇²c_term(X[N])
end



# function get_δx(E,i)
#     sqrt(E)[:,i]
# end
#
# function ∇δx(E,i)
#     n = size(E,1)
#
#     function f1(e)
#         ee = eigen(E)
#
#         b = Diagonal(sqrt.(ee.values))*ee.vectors'
#         b[:,i]
#     end
#
#     ForwardDiff.jacobian(f1,vec(E))[:,((i-1)*n + 1):i*n]
# end
#
# function get_δu(K,z,u,idx,j)
#     _K = K(z,u)
#     n = size(_K,2)
#     E = reshape(z[idx.e],n,n)
#     sqrt(_K*E*_K')[:,j]
# end
#
# function ∇δu(K,z,u,idx,j)
#     m = length(u)
#
#     y = [z;u]
#     function f1(y)
#         _K = K(y[idx.z],y[length(idx.z)+1:end])
#         n = size(_K,2)
#         E = reshape(y[idx.e],n,n)
#         tmp = _K*E*_K'
#         ee = eigen(tmp)
#         b = Diagonal(sqrt.(ee.values))*ee.vectors'
#         b[:,j]
#     end
#
#     ForwardDiff.jacobian(f1,y)
# end
#
# "Modify constraint to evaluate all combinations of the disturbed state and control"
# function robust_constraint(c::AbstractConstraint,K::Function,idx::NamedTuple,n::Int,m::Int,n̄::Int)
#     p = c.p
#     p_robust = p*(2*n+1)*(2*m+1)
#
#     function rc(v,z,u)
#         xw = Vector[]
#         uw = Vector[]
#
#         x = z[idx.x]
#         u = u[1:m]
#         E = reshape(z[idx.e],n,n)
#         Ex = sqrt(E)
#         _K = K(z,u)
#         Eu = sqrt(_K*E*_K')
#
#         push!(xw,x)
#         push!(uw,u)
#
#         for i = 1:n
#             if norm(imag.(Ex[:,i])) > 0.1
#                 error("imag. error")
#             end
#             δx = real.(Ex[:,i])
#             push!(xw,x + δx)
#             push!(xw,x - δx)
#         end
#         for j = 1:m
#             if norm(imag.(Eu[:,j])) > 0.1
#                 error("imag. error")
#             end
#             δu = real.(Eu[:,j])
#             push!(uw,u + δu)
#             push!(uw,u - δu)
#         end
#
#         k = 1
#         for _x in xw
#             for _u in uw
#                 c.c(view(v,((k-1)*p+1:k*p)),_x,_u)
#                 k += 1
#             end
#         end
#     end
#
#     function ∇rc(V,z,u)
#         xw = Vector[]
#         uw = Vector[]
#         sx = []
#         su = []
#
#         x = z[idx.x]
#         u = u[1:m]
#         E = reshape(z[idx.e],n,n)
#         Ex = sqrt(E)
#         _K = K(z,u)
#         Eu = sqrt(_K*E*_K')
#
#         push!(xw,x)
#         push!(sx,(0,0))
#
#         push!(uw,u)
#         push!(su,(0,0))
#
#         for i = 1:n
#             δx = real.(Ex[:,i])
#             push!(xw,x + δx)
#             push!(sx,(i,1))
#             push!(xw,x - δx)
#             push!(sx,(i,-1))
#         end
#         for j = 1:m
#             δu = real.(Eu[:,j])
#             push!(uw,u + δu)
#             push!(su,(j,1))
#             push!(uw,u - δu)
#             push!(su,(j,-1))
#         end
#
#         k = 1
#         for (i,_x) in enumerate(xw)
#             for (j,_u) in enumerate(uw)
#
#                 _V = zeros(p,n+m)
#                 idx_p = (((k-1)*p+1):k*p)
#                 c.∇c(_V,_x,_u)
#
#                 _idx = [(1:n)...,((n + n^2 + n*r + n^2) .+ (1:m))...]
#                 copyto!(view(V,idx_p,_idx),_V)
#
#                 cx = _V[:,1:n]
#                 cu = _V[:,n .+ (1:m)]
#                 #
#                 # if i != 1
#                 #     _idx = (n .+ (((sx[i][1]-1)*n + 1): sx[i][1]*n))
#                 #     copyto!(view(V,idx_p,_idx),sx[i][2]*cx*∇δx(E,sx[i][1]))
#                 # end
#                 if j != 1
#                     # _idx = ((n + n^2 + n*r + n^2) .+ (((su[j][1]-1)*m + 1): su[j][1]*m))
#                     _idx = n .+ (1:n^2)
#                     copyto!(view(V,idx_p,_idx),su[j][2]*cu*∇δu(K,z,u,idx,su[j][1]))
#                 end
#
#                 k += 1
#             end
#         end
#     end
#
#     typeof(c)(rc,∇rc,p_robust,c.label,[collect(1:n̄),collect(1:m)],c.type)
# end
#
# function robust_constraint(c::AbstractConstraint,n::Int,n̄::Int)
#     p = c.p
#     p_robust = p*(2*n+1)
#
#     function rc(v,z)
#         xw = Vector[]
#
#         x = z[idx.x]
#         E = reshape(z[idx.e],n,n)
#         Ex = sqrt(E)
#
#         push!(xw,x)
#
#         for i = 1:n
#             if norm(imag.(Ex[:,i])) > 0.1
#                 error("imag. error")
#             end
#             δx = real.(Ex[:,i])
#             push!(xw,x + δx)
#             push!(xw,x - δx)
#         end
#
#         for (k,_x) in enumerate(xw)
#             c.c(view(v,((k-1)*p+1:k*p)),_x)
#         end
#     end
#
#     function ∇rc(V,z)
#         xw = Vector[]
#         sx = []
#
#         x = z[idx.x]
#         E = reshape(z[idx.e],n,n)
#         Ex = sqrt(E)
#
#         push!(xw,x)
#         push!(sx,(0,0))
#
#         for i = 1:n
#             δx = real.(Ex[:,i])
#             push!(xw,x + δx)
#             push!(sx,(i,1))
#             push!(xw,x - δx)
#             push!(sx,(i,-1))
#         end
#
#         for (k,_x) in enumerate(xw)
#             _V = zeros(p,n)
#             idx_p = (((k-1)*p+1):k*p)
#             c.∇c(_V,_x)
#             _idx = (k-1)*n+1:k*n
#             copyto!(view(V,idx_p,_idx),_V)
#
#             cx = _V[:,1:n]
#
#             if k != 1
#                 _idx = (n .+ (((sx[k][1]-1)*n + 1): sx[k][1]*n))
#                 copyto!(view(V,idx_p,_idx),sx[k][2]*cx*∇δx(E,sx[k][1]))
#             end
#         end
#     end
#
#     typeof(c)(rc,∇rc,p_robust,c.label,[collect(1:n̄),collect(1:0)],c.type)
# end
#
# "Update jacobian for robust dynamics that include δx, δu terms in state vector"
# function update_jacobian(_∇F::Function,n::Int,m::Int,r::Int)
#     inds = (x=1:n, u=n .+ (1:m),w = (n+m) .+ (1:r), dt=n+m+r+1, xx=(1:n,1:n),xu=(1:n,n .+ (1:m)),xw=(1:n,(n+m) .+ (1:r)), xdt=(1:n,(n+m+r) .+ (1:1)))
#     # nδ = 2*(n^2 + m^2)
#     n_robust = n^2 + n*r + n^2
#     S0 = zeros(n,n+n_robust+m+r+1)
#     ẋ0 = zeros(n)
#
#     idx_x = 1:n
#     idx_u = 1:m
#     idx_w = 1:r
#     idx = [(idx_x)...,((n+n_robust) .+ (idx_u))...,((n+n_robust+m) .+ (idx_w))...,(n+n_robust+m+r+1)]
#
#     ∇F(S::AbstractMatrix,ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T= begin
#         _∇F(view(S,idx_x,idx),ẋ,x,u,w,dt)
#         return nothing
#     end
#     ∇F(S::AbstractMatrix,x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T = begin
#         _∇F(view(S,idx_x,idx),ẋ0,x,u,w,dt)
#         return nothing
#     end
#     ∇F(x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T = begin
#         _∇F(view(S0,idx_x,idx),ẋ0,x,u,w,dt)
#         return S0
#     end
#     return ∇F
# end

function gen_robust_exp_funcs(fc::Function,idx::NamedTuple,Qr::AbstractArray,Rr::AbstractArray,
        Qfr::AbstractArray,n::Int,m::Int,r::Int)
    n̄ = n+n^2+n*r+n^2

    function K(z,u)
        x = z[idx.x]
        s = z[idx.s]
        P = reshape(s,n,n)*reshape(s,n,n)'
        Zc = zeros(eltype(z),n,n+m+r)
        f_aug(q̇,q) = fc(q̇,q[1:n],q[n .+ (1:m)],zeros(eltype(q),r))
        ∇fc(q) = ForwardDiff.jacobian(f_aug,zeros(eltype(q),n),q)

        Bc = ∇fc([x;u[1:m]])[:,n .+ (1:m)]
        R\(Bc'*P)
    end

    function stage_cost(y)
        z = y[1:n̄]
        u = y[n̄ .+ (1:m)]

        x = z[idx.x]
        E = reshape(z[idx.e],n,n)

        Kc = K(z,u)

        tr((Qr + Kc'*Rr*Kc)*E)
    end

    function stage_cost_term(z)
        E = reshape(z[idx.e],n,n)
        tr(Qfr*E)
    end

    ∇sc(y) = ForwardDiff.gradient(stage_cost,y)
    ∇²sc(y) = ForwardDiff.hessian(stage_cost,y)

    ∇sc_term(y) = ForwardDiff.gradient(stage_cost_term,y)
    ∇²sc_term(y) = ForwardDiff.hessian(stage_cost_term,y)

    return ∇sc, ∇²sc, ∇sc_term, ∇²sc_term
end


function gen_cubic_interp(X,dt)
    N = length(X); n = length(X[1])

    function interp(t)
        j = t/dt + 1.
        x = zeros(eltype(t),n)
        for i = 1:n
            x[i] = interpolate([X[k][i] for k = 1:N],BSpline(Cubic(Line(OnGrid()))))(j)
        end
        return x
    end
end

function gen_zoh_interp(X,dt)
    N = length(X); n = length(X[1])

    function interp(t)
        if (t/dt + 1.0)%floor(t/dt + 1.0) < 0.99999
            j = convert(Int64,floor(t/dt)) + 1
        else
            j = convert(Int64,ceil(t/dt)) + 1
        end

        interpolate(X,BSpline(Constant()))(j)

    end
end
