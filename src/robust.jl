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

        # dt scaling is used to match continous Riccati dynamics
        A, B = ∇F[k].xx, ∇F[k].xu
        K[k] .= (R*dt + B'*P[k+1]*B)\(B'*P[k+1]*A)
        P[k] .= Q*dt + K[k]'*(R*dt)*K[k] + (A - B*K[k])'*P[k+1]*(A - B*K[k])
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
        P[k] .= Q*dt + K[k]'*(R*dt)*K[k] + (Ad[k] - Bd[k]*K[k])'*P[k+1]*(Ad[k] - Bd[k]*K[k])
    end

    return K, P, Ad, Bd, Gd
end

function tvlqr_con_uncertain(prob::Problem{T,Discrete},Q::AbstractArray{T},R::AbstractArray{T},
        Qf::AbstractArray{T},xf::AbstractVector{T},integrator::Symbol) where T

    n = prob.model.n; m = prob.model.m; r = prob.model.r; N = prob.N

    K = [zeros(T,m,n) for k = 1:N-1]
    P = [zeros(T,n,n) for k = 1:N]
    Ps = [zeros(T,n,n) for k = 1:N] # sqrt riccati
    Āc = [zeros(T,n,n) for k = 1:N-1]
    B̄c = [zeros(T,n,m) for k = 1:N-1]
    Ḡc = [zeros(T,n,r) for k = 1:N-1]

    P[N] .= Qf
    Ps[N] .= cholesky(Qf).L

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

        Kc = R\(Bc'*_P)
        Acl = Ac - Bc*Kc

        f(view(ż,1:n),[x;u;w])
        ż[n .+ (1:n^2)] = reshape(-.5*Q*ss - Ac'*S + .5*_P*Bc*(R\(Bc'*S)),n^2)
    end

    Z = [zeros(n+n^2) for k = 1:N]
    Z[N] = [prob.X[end];vec(cholesky(Qf).L)]

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
        K[k] = R\(B̄c[k]'*P[k])
    end

    return K,P,Ps,Āc,B̄c,Ḡc
end

function robust_problem(prob::Problem{T},
    E0,H0,D,
    Q,R,Qf,xf,
    Q_lqr,R_lqr,Qf_lqr,
    Q_r,R_r,Qf_r,
    riccati_cost=1.0,riccati_initial_cost=1.0,
    E_cost=1.0, H_cost=1.0) where T

    N = prob.N
    @assert all([prob.obj[k] isa QuadraticCost for k = 1:N]) #TODO generic cost

    n = prob.model.n; m = prob.model.m; r = prob.model.r
    n_robust = 2*n^2 + n*r # number of robust parameters in state vector
    n̄ = n + n_robust
    m1 = m + n^2
    idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)),z=(1:n̄))

    ## Nominal trajectories

    # X,U
    rollout!(prob)
    if length(String(prob.model.info[:integration])) < 6
        @info "Defaulting Riccati solve to Implicit Midpoint"
        integrator = :ImplicitMidpoint
    elseif String(prob.model.info[:integration])[1:6] == "DiffEq"
        integrator = Symbol(split(String(prob.model.info[:integration]),"_")[2])
        # error("Must specific DiffEq integrator/solver (for now...)")
    end

    # Riccati (sqrt)
    _Kc, _Pc, _Sc, _Ac, _Bc, _Gc = tvlqr_con_uncertain(prob,Q_lqr,R_lqr,Qf_lqr,xf,integrator)
    _K, _P, _A, _B, _G = tvlqr_dis_uncertain(prob,Q_lqr,R_lqr,Qf_lqr)

    S1 = vec(_Sc[1])
    SN = vec(cholesky(Qf_lqr).L)

    _Z = [zeros(n̄) for k = 1:N]
    _E = [zeros(n,n) for k = 1:N]
    _H = [zeros(n,r) for k = 1:N]

    # Disturbances
    _E[1] = E0
    _H[1] = H0

    _Z[1] = [prob.x0;vec(E0);vec(H0);S1]

    for k = 1:N-1
        Acl = _A[k] - _B[k]*_K[k]
        _E[k+1] = Acl*_E[k]*Acl' + _G[k]*_H[k]'*Acl' + Acl*_H[k]*_G[k]' + _G[k]*D*_G[k]'
        _H[k+1] = Acl*_H[k] + _G[k]*D

        _Z[k+1] = [prob.X[k+1];vec(_E[k+1]);vec(_H[k+1]);vec(_Sc[k+1])]
    end


    #generate optimal feedback matrix function (continuous)
    f(ẋ,z) = prob.model.info[:fc](ẋ,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r))
    ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
    ∇f(x,u) = ∇f([x;u])

    K(z,u) = let idx=idx, n=n, m=m, ∇f=∇f, R_lqr=R_lqr
        x = z[idx.x]
        s = z[idx.s]
        P = reshape(s,n,n)*reshape(s,n,n)'
        Bc = ∇f(x,u)[:,n .+ (1:m)]
        R_lqr\(Bc'*P)
    end

    # f(x⁺,z) = prob.model.f(x⁺,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r),z[n+m+1])
    # ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
    # ∇f(x,u) = ∇f([x;u;prob.dt])
    #
    # function K(z,u)
    #     x = z[idx.x]
    #     s = z[idx.s]
    #     P = reshape(s,n,n)*reshape(s,n,n)'
    #     Fd = ∇f(x,u)
    #     Ad = Fd[:,1:n]
    #     Bd = Fd[:,n .+ (1:m)]
    #     (Rr*prob.dt + Bd'*P*Bd)\(Bd'*P*Ad)
    # end

    # modify cost
    _cost = CostFunction[]
    for k = 1:N-1
        cost_robust = copy(prob.obj[k])
        cost_robust.Q = cat(cost_robust.Q,Diagonal([E_cost*ones(n^2);H_cost*ones(n*r);riccati_cost*ones(n^2)]),dims=(1,2))
        cost_robust.q = [cost_robust.q; zeros(n_robust)]
        cost_robust.H = [cost_robust.H zeros(m,n_robust)]

        # quadratic cost on riccati states
        for j = (n̄-(n^2 -1)):n̄
            cost_robust.Q[j,j] = riccati_cost
        end

        if k == 1
            cost_robust.R = cat(cost_robust.R,Diagonal(riccati_initial_cost*ones(n^2)),dims=(1,2))
            cost_robust.r = [cost_robust.r; zeros(n^2)]
            cost_robust.H = [cost_robust.H; zeros(n^2,n̄)]
        end
        push!(_cost,cost_robust)
    end

    cost_robust = copy(prob.obj[N])
    cost_robust.Qf = cat(cost_robust.Qf,Diagonal([E_cost*ones(n^2);H_cost*ones(n*r);riccati_cost*ones(n^2)]),dims=(1,2))
    cost_robust.qf = [cost_robust.qf; zeros(n_robust)]

    # quadratic cost on riccati states
    for j = (n̄-(n^2 -1)):n̄
        cost_robust.Qf[j,j] = riccati_cost
    end

    push!(_cost,cost_robust)

    # create robust objective
    _obj = Objective(_cost)

    ∇sc, ∇²sc, ∇sc_term, ∇²sc_term = robust_cost_expansions(prob.model.info[:fc],K,idx,Q_r,R_r,Qf_r,n,m,r)
    _robust_cost = RobustCost(Q_r,R_r,Qf_r,K,n,m,r,idx,∇sc,∇²sc,∇sc_term,∇²sc_term)
    robust_obj = RobustObjective(_obj,_robust_cost)

    # create robust model
    _robust_model = robust_model(prob.model,Q_lqr,R_lqr,D)

    constrained = is_constrained(prob)
    con_prob = ConstraintSet[]

    # function s1(c,z,u)
    #     c[1:n^2] = u[m .+ (1:n^2)] - z[idx.s]
    # end
    #
    # function ∇s1(C,z,u)
    #     C[:,(n+n^2+n*r) .+ (1:n^2)] = -1.0*Diagonal(ones(n^2))
    #     C[:,(n̄+m) .+ (1:n^2)] = 1.0*Diagonal(ones(n^2))
    # end
    #
    # ctg_init = Constraint{Equality}(s1,∇s1,n̄,m1,n^2,:ctg_init)

    sN(c,z) = let n=n,idx=idx,SN=SN
        c[1:n^2] = z[idx.s] - SN
    end

    ∇sN(C,z) = let n=n,m=m,r=r
        C[:,(n+n^2+n*r) .+ (1:n^2)] = Diagonal(ones(n^2))
    end

    ctg_term = Constraint{Equality}(sN,∇sN,n^2,:ctg_term,[collect(1:n̄),collect(1:0)],:terminal)

    for k = 1:N-1
        con_uncertain = GeneralConstraint[]
        # if k == 1
        #     push!(con_uncertain,ctg_init)
        # end
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
        x0=_Z[1])
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
function robust_model(f::Function,Q_lqr::AbstractArray{T},R_lqr::AbstractArray{T},D::AbstractArray{T},n::Int,m::Int,r::Int) where T
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

        Kc = R_lqr\(Bc'*P)
        Acl = Ac - Bc*Kc

        f(view(ż,z_idx.x),x,u,w)
        ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
        ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
        ż[z_idx.s] = reshape(-.5*Q_lqr*ss - Ac'*S + .5*P*Bc*(R_lqr\(Bc'*S)),n^2)
    end

    UncertainModel(robust_dynamics, n̄, m, r)
end

mutable struct RobustCost{T} <: CostFunction
    Q_r::AbstractArray{T}
    R_r::AbstractArray{T}
    Qf_r::AbstractArray{T}

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

    tr((cost.Q_r + Kc'*cost.R_r*Kc)*E)
end

function stage_cost(cost::RobustCost, zN::Vector{T}) where T
    idx = cost.idx; n = cost.n
    E = reshape(zN[idx.e],n,n)
    tr(cost.Qf_r*E)
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

function robust_cost_expansions(fc::Function, K::Function,idx::NamedTuple,Q_r::AbstractArray,R_r::AbstractArray,
        Qf_r::AbstractArray,n::Int,m::Int,r::Int)
    n̄ = n+n^2+n*r+n^2

    f(ẋ,z) = fc(ẋ,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r))
    ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
    ∇f(x,u) = ∇f([x;u])

    function Bc(x,u)
        ∇f(x,u)[:,n .+ (1:m)]
    end

    Bc(z) = Bc(z[1:n],z[n .+ (1:m)])
    ∇Bc(z) = ForwardDiff.jacobian(Bc,z)

    function ∇stage_cost(y)
        dJ = zeros(eltype(y),n̄+m)
        z = y[idx.z]
        u = y[length(idx.z) .+ (1:m)]
        _E = reshape(z[idx.e],n,n)
        _S = reshape(z[idx.s],n,n)
        _P = _S*_S'
        _K = K(z,u)
        _B = Bc(z)
        _∇B = ∇Bc([z[idx.x];u])

        dJdK = reshape(R_r*_K*_E + R_r'*_K*_E',1,m*n)
        dJdE = vec((Q_r + _K'*R_r*_K)')
        dKdB = kron(_P',inv(R_r))*comm(n,m)
        dKdP = kron(Diagonal(ones(n)),(R_r)\_B')
        dKdS = kron(_S,(R_r)\_B') + kron(Diagonal(ones(n)), (R_r)\(_B'*_S))*comm(n,n)
        dBdX = _∇B[:,1:n]
        dBdU = _∇B[:,n .+ (1:m)]

        dJ[idx.x] = dJdK*dKdB*dBdX
        dJ[idx.e] = dJdE
        dJ[idx.s] = dJdK*dKdS
        dJ[n̄ .+ (1:m)] = dJdK*dKdB*dBdU
        dJ
    end

    function ∇stage_cost_term(zN)
        dJN = zeros(eltype(zN),n̄)
        dJN[idx.e] = vec(Qf_r')
        dJN
    end

    function stage_cost(y)
        z = y[1:n̄]
        u = y[n̄ .+ (1:m)]

        E = reshape(z[idx.e],n,n)

        Kc = K(z,u)

        tr((Q_r + Kc'*R_r*Kc)*E)
    end

    function stage_cost_term(z)
        E = reshape(z[idx.e],n,n)
        tr(Qf_r*E)
    end

    ∇sc(y) = ∇stage_cost(y) #ForwardDiff.gradient(stage_cost,y)
    ∇²sc(y) = ForwardDiff.jacobian(∇stage_cost,y)

    ∇sc_term(y) = ∇stage_cost_term(y) #ForwardDiff.gradient(stage_cost_term,y)
    ∇²sc_term(y) = ForwardDiff.jacobian(∇stage_cost_term,y)

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
