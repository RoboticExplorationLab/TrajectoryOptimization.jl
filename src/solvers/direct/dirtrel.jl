# Pendulum Swing-Up Direct Transcription
using TrajectoryOptimization, LinearAlgebra, Plots, ForwardDiff, MatrixCalculus, Ipopt, MathOptInterface, BenchmarkTools
const MOI = MathOptInterface

"DIRTREL Solver
-contains parameters for initial disturbance trajectories, disturbance costs, LQR controller, and timestep bounds"
struct DIRTRELSolver{T} <: AbstractSolver{T}
    E0::AbstractArray   # initial disturbance
    H0::AbstractArray   # initial off-diagonal term from M (see DIRTREL)
    D::AbstractArray    # disturbance parameter (see DIRTREL)

    # Cost function
    Q::AbstractArray
    R::AbstractArray
    Rh::T # cost on time step h
    Qf::AbstractArray
    xf

    # LQR controller
    Q_lqr::AbstractArray
    R_lqr::AbstractArray
    Qf_lqr::AbstractArray

    # Robust cost
    Q_r::AbstractArray
    R_r::AbstractArray
    Qf_r::AbstractArray

    # Eigen value threshold for padding individual eigen values prior to taking sqrt(E)
    eig_thr::T

    h_max::T # timestep upper bound
    h_min::T # time step lower bound (h_min>0)
end

"DIRTREL Problem
-contains Problem, constraint dimensions/functions/jacobians/structures"
struct DIRTRELProblem <: MOI.AbstractNLPEvaluator
    prob::Problem                       # Trajectory Optimization problem
    p_dynamics::Int                     # number of dynamics constraints (including timestep equality)
    p_dynamics_jac::Int                 # number of non-zero elements in dynamics constraint jacobian
    p_robust::Int                       # number of robust constraints
    p_robust_jac::Int                   # number of non-zero elements in robust constraints jacobian
    cost::Function                      # cost (currently cubic spline on quadratic stage cost :x'Qx + u'Ru; spline has + h)
    ∇cost!::Function                    # cost gradient
    robust_cost::Function               # robust cost from DIRTREL
    ∇robust_cost::Function              # robust cost gradient
    dynamics_constraints!::Function     # dynamics constraints
    ∇dynamics_constraints!::Function    # dynamics constraint jacobian
    robust_constraints!::Function       # robust constraints
    ∇robust_constraints!::Function      # robust constraints jacobian

    jac_struct                          # contains index list of non-zero elements of concatenated constraint jacobian [dynamics;robust]

    packZ::Function                     # stack X,U,H trajectories -> Z = [x1;u1;h1;...;xN]
    unpackZ::Function                   # unpack Z to X,U,H trajectories -> X,U,H = Z

    NN::Int                             # number of decision variables

    enable_hessian::Bool
end

function DIRTRELProblem(prob::Problem,solver::DIRTRELSolver)
    # DIRTREL problem dimensions
    n = prob.model.n; m = prob.model.m; N = prob.N
    NN = (n+m+1)*(N-1) + n  # number of decision variables
    p_dynamics = n*(N+1) + (N-2)
    p_dynamics_jac = n*(N-1)*(n+m+1+n) + 2*n^2 + (N-2)*2
    p_robust, p_robust_jac = num_robust_constraints(prob)

    # Constraint jacobian sparsity structure
    idx = dynamics_constraints_sparsity(n,m,N,NN)
    idx_robust = robust_constraints_sparsity(prob,n,m,N,NN,p_dynamics)
    jac_struc = idx
    append!(jac_struc,idx_robust)

    # DIRTREL functions
    cost, ∇cost!, robust_cost, ∇robust_cost, dynamics_constraints!,∇dynamics_constraints!,robust_constraints!,∇robust_constraints!, packZ, unpackZ = gen_DIRTREL_funcs(prob,solver)

    DIRTRELProblem(prob,p_dynamics,p_dynamics_jac,p_robust,p_robust_jac,cost,
        ∇cost!, robust_cost, ∇robust_cost, dynamics_constraints!,∇dynamics_constraints!,
        robust_constraints!,∇robust_constraints!,
        jac_struc,
        packZ,unpackZ,
        NN,
        false)
end

"Generate functions needed for DIRTREL solve"
function gen_DIRTREL_funcs(prob::Problem,solver::DIRTRELSolver)
    Q = solver.Q; R = solver.R; Rh = solver.Rh; Qf = solver.Qf; xf = solver.xf
    E0 = solver.E0; H0 = solver.H0; D = solver.D; Q_lqr = solver.Q_lqr; R_lqr = solver.R_lqr; Qf_lqr = solver.Qf_lqr
    Q_r = solver.Q_r; R_r = solver.R_r; Qf_r = solver.Qf_r; eig_thr = solver.eig_thr
    n = prob.model.n; m = prob.model.m; r = prob.model.r; N = prob.N
    NN = n*N + m*(N-1) + (N-1)

    x0 = prob.x0

    # Continuous dynamics (uncertain)
    function fc(z)
        ż = zeros(eltype(z),n)
        prob.model.f(ż,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
        return ż
    end

    function fc(x,u,w)
        ẋ = zero(x)
        prob.model.f(ẋ,x,u,w)
        return ẋ
    end

    ∇fc(z) = ForwardDiff.jacobian(fc,z)
    ∇fc(x,u,w) = ∇fc([x;u;w])
    dfcdx(x,u,w) = ∇fc(x,u,w)[:,1:n]
    dfcdu(x,u,w) = ∇fc(x,u,w)[:,n .+ (1:m)]
    dfcdw(x,u,w) = ∇fc(x,u,w)[:,(n+m) .+ (1:r)]

    xm(y,x,u,h) = 0.5*(y + x) + h/8*(fc(x,u,zeros(r)) - fc(y,u,zeros(r)))
    dxmdy(y,x,u,h) = 0.5*I - h/8*dfcdx(y,u,zeros(r))
    dxmdx(y,x,u,h) = 0.5*I + h/8*dfcdx(x,u,zeros(r))
    dxmdu(y,x,u,h) = h/8*(dfcdu(x,u,zeros(r)) - dfcdu(y,u,zeros(r)))
    dxmdh(y,x,u,h) = 1/8*(fc(x,u,zeros(r)) - fc(y,u,zeros(r)))

    # cubic interpolation on state
    F(y,x,u,h) = y - x - h/6*fc(x,u,zeros(r)) - 4*h/6*fc(xm(y,x,u,h),u,zeros(r)) - h/6*fc(y,u,zeros(r))
    F(z) = F(z[1:n],z[n .+ (1:n)],z[(n+n) .+ (1:m)],z[n+n+m+1])
    ∇F(Z) = ForwardDiff.jacobian(F,Z)
    dFdy(y,x,u,h) = I - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdy(y,x,u,h) - h/6*dfcdx(y,u,zeros(r))
    dFdx(y,x,u,h) = -I - h/6*dfcdx(x,u,zeros(r)) - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdx(y,x,u,h)
    dFdu(y,x,u,h) = -h/6*dfcdu(x,u,zeros(r)) - 4*h/6*(dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdu(y,x,u,h) + dfcdu(xm(y,x,u,h),u,zeros(r))) - h/6*dfcdu(y,u,zeros(r))
    dFdh(y,x,u,h) = -1/6*fc(x,u,zeros(r)) - 4/6*fc(xm(y,x,u,h),u,zeros(r)) - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdh(y,x,u,h) - 1/6*fc(y,u,zeros(r))
    dFdw(y,x,u,h) = -h/6*dfcdw(x,u,zeros(r)) - 4*h/6*dfcdw(xm(y,x,u,h),u,zeros(r)) - h/6*dfcdw(y,u,zeros(r))

    function packZ(X,U,H)
        Z = [X[1];U[1];H[1]]

        for k = 2:N-1
            append!(Z,[X[k];U[k];H[k]])
        end
        append!(Z,X[N])

        return Z
    end

    function unpackZ(Z)
        X = [k != N ? Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][1:n] : Z[((k-1)*(n+m+1) + 1):NN] for k = 1:N]
        U = [Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][n .+ (1:m)] for k = 1:N-1]
        H = [Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][(n+m+1)] for k = 1:N-1]
        return X,U,H
    end

    function unpackZ_timestep(Z)
        Z_traj = [k != N ? Z[((k-1)*(n+m+1) + 1):k*(n+m+1)] : Z[((k-1)*(n+m+1) + 1):NN] for k = 1:N]
    end

    # Cost
    function ℓ(x,u)
        (x'*Q*x + u'*R*u)
    end

    function dℓdx(x,u)
        2*Q*x
    end

    function dℓdu(x,u)
        2*R*u
    end

    function gf(x)
        x'*Qf*x
    end

    function dgfdx(x)
        2*Qf*x
    end

    # cubic interpolated stage cost
    function g_stage(y,x,u,h)
        h/6*ℓ(x,u) + 4*h/6*ℓ(xm(y,x,u,h),u) + h/6*ℓ(y,u) + Rh*h
    end

    # g(z) = g(z[1:n],z[n .+ (1:n)],z[(n+n) .+ (1:m)],z[n+n+m+1])
    # ∇g(z) = ForwardDiff.gradient(g,z)

    dgdx(y,x,u,h) = h/6*dℓdx(x,u) + 4*h/6*dxmdx(y,x,u,h)'*dℓdx(xm(y,x,u,h),u)
    dgdy(y,x,u,h) = 4*h/6*dxmdy(y,x,u,h)'*dℓdx(xm(y,x,u,h),u)+ h/6*dℓdx(y,u)
    dgdu(y,x,u,h) = h/6*dℓdu(x,u) + 4*h/6*(dxmdu(y,x,u,h)'*dℓdx(xm(y,x,u,h),u) + dℓdu(xm(y,x,u,h),u)) + h/6*dℓdu(y,u)
    dgdh(y,x,u,h) = 1/6*ℓ(x,u) + 4/6*ℓ(xm(y,x,u,h),u) + 4*h/6*dxmdh(y,x,u,h)'*dℓdx(xm(y,x,u,h),u) + 1/6*ℓ(y,u) + Rh

    # dgdx(y,x,u,h) = vec(h/6*dℓdx(x,u)' + 4*h/6*dℓdx(xm(y,x,u,h),u)'*dxmdx(y,x,u,h))
    # dgdy(y,x,u,h) = vec(4*h/6*dℓdx(xm(y,x,u,h),u)'*dxmdy(y,x,u,h) + h/6*dℓdx(y,u)')
    # dgdu(y,x,u,h) = vec(h/6*dℓdu(x,u)' + 4*h/6*(dℓdx(xm(y,x,u,h),u)'*dxmdu(y,x,u,h) + dℓdu(xm(y,x,u,h),u)') + h/6*dℓdu(y,u)')
    # dgdh(y,x,u,h) = 1/6*ℓ(x,u) + 4/6*ℓ(xm(y,x,u,h),u) + 4*h/6*dℓdx(xm(y,x,u,h),u)'*dxmdh(y,x,u,h) + 1/6*ℓ(y,u) + Rh

    # Robust cost
    robust_cost(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D
        X,U,H = unpackZ(Z)
        Ad = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdx(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
        Bd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdu(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
        Gd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdw(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]

        P = [zeros(eltype(Z),n,n) for k = 1:N]
        K = [zeros(eltype(Z),m,n) for k = 1:N-1]

        E = [zeros(eltype(Z),n,n) for k = 1:N]
        HH = [zeros(eltype(Z),n,r) for k = 1:N]

        P[N] = Qf_lqr
        E[1] = E0
        HH[1] = H0

        for k = N-1:-1:1
            K[k] = (R_lqr + Bd[k]'*P[k+1]*Bd[k])\(Bd[k]'*P[k+1]*Ad[k])
            P[k] = Q_lqr + K[k]'*R_lqr*K[k] + (Ad[k] - Bd[k]*K[k])'*P[k+1]*(Ad[k] - Bd[k]*K[k])
        end

        _ℓ = 0.
        for k = 1:N-1
            Acl = Ad[k] - Bd[k]*K[k]
            E[k+1] = Acl*E[k]*Acl' + Gd[k]*HH[k]'*Acl' + Acl*HH[k]*Gd[k]' + Gd[k]*D*Gd[k]'
            HH[k+1] = Acl*HH[k] + Gd[k]*D
            _ℓ += tr((Q_r + K[k]'*R_r*K[k])*E[k])
        end
        _ℓ += tr(Qf_r*E[N])

        return _ℓ
    end

    # Robust cost gradient
    ∇robust_cost(Z) = ForwardDiff.gradient(robust_cost,Z)

    # Cost
    function cost(Z)
        J = 0.
        Z_traj = unpackZ_timestep(Z)

        for k = 1:N-1
            y = Z_traj[k+1][1:n]
            x = Z_traj[k][1:n]
            u = Z_traj[k][n .+ (1:m)]
            h = Z_traj[k][n+m+1]
            J += g_stage(y,x,u,h)
        end
        xN = Z_traj[N][1:n]
        J += gf(xN)

        return J
    end

    # Cost gradient
    ∇cost!(∇J,Z) = let n=n, m=m, N=N, NN=NN, Q=Q, R=R, Qf=Qf
        Z_traj = unpackZ_timestep(Z)
        k = 1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]
        ∇J[1:(n+m+1+n)] = [dgdx(y,x,u,h);dgdu(y,x,u,h);dgdh(y,x,u,h);dgdx(y,x,u,h)]

        for k = 2:N-1
            y = Z_traj[k+1][1:n]
            x = Z_traj[k][1:n]
            u = Z_traj[k][n .+ (1:m)]
            h = Z_traj[k][n+m+1]
            ∇J[((k-1)*(n+m+1) + 1):(k*(n+m+1)+n)] = [dgdx(y,x,u,h);dgdu(y,x,u,h);dgdh(y,x,u,h);dgdx(y,x,u,h)]
        end
        xN = Z_traj[N][1:n]
        ∇J[(NN - (n-1)):NN] = dgfdx(xN)

        return nothing
    end

    # Evaluate disturbance trajectory E
    gen_E(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D
        X,U,H = unpackZ(Z)
        Ad = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdx(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
        Bd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdu(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
        Gd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdw(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]

        P = [zeros(eltype(Z),n,n) for k = 1:N]
        K = [zeros(eltype(Z),m,n) for k = 1:N-1]

        E = [zeros(eltype(Z),n,n) for k = 1:N]
        HH = [zeros(eltype(Z),n,r) for k = 1:N]
        E_vec = zeros(eltype(Z),N*n^2)

        P[N] = Qf_lqr
        E[1] = E0
        HH[1] = H0
        E_vec[1:n^2] = E0

        for k = N-1:-1:1
            K[k] = (R_lqr + Bd[k]'*P[k+1]*Bd[k])\(Bd[k]'*P[k+1]*Ad[k])
            P[k] = Q_lqr + K[k]'*R_lqr*K[k] + (Ad[k] - Bd[k]*K[k])'*P[k+1]*(Ad[k] - Bd[k]*K[k])
        end

        for k = 1:N-1
            Acl = Ad[k] - Bd[k]*K[k]
            E[k+1] = Acl*E[k]*Acl' + Gd[k]*HH[k]'*Acl' + Acl*HH[k]*Gd[k]' + Gd[k]*D*Gd[k]'
            HH[k+1] = Acl*HH[k] + Gd[k]*D
            E_vec[k*(n^2) .+ (1:n^2)] = vec(E[k+1])
        end

        return E_vec
    end

    # Evaluate state disturbance trajectory δx
    gen_δx(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D, eig_thr=eig_thr
        E = gen_E(Z)
        Esqrt = zero(E)

        for k = 1:N
            ee = reshape(E[(k-1)*n^2 .+ (1:n^2)],n,n)

            # make E >= 0
            if !isposdef(ee)
                _eig = eigen(ee)
                _eig.values .= real.(_eig.values)
                for i = 1:n
                    if _eig.values[i] < 0.
                        _eig.values[i] = eig_thr
                    end
                end
                Esqrt[(k-1)*n^2 .+ (1:n^2)] = Diagonal(sqrt.(_eig.values))*_eig.vectors'
            else
                Esqrt[(k-1)*n^2 .+ (1:n^2)] = real.(vec(sqrt(ee)))
            end
        end

        return Esqrt
    end

    # Gradient of disturbance trajectory wrt decision variables
    dEdZ(Z) = ForwardDiff.jacobian(gen_E,Z)

    # Gradient of δx trajectory wrt decision variables
    dδxdZ(Z) = let n=n,m=m,N=N,NN=NN
        res = zeros(eltype(Z),N*n^2,NN)
        Esqrt = gen_δx(Z)
        ∇E = dEdZ(Z)
        In = Diagonal(ones(n))

        for k = 1:N
            idx = (k-1)*n^2 .+ (1:n^2)
            _Esqrt = reshape(Esqrt[idx],n,n)
            res[idx,1:NN] = pinv(kron(In,_Esqrt) + kron(_Esqrt',In))*∇E[idx,1:NN]
            # res[idx,1:NN] = (kron(In,_Esqrt) + kron(_Esqrt',In))\∇E[idx,1:NN]
        end

        res
    end

    # Evaluate KEK trajectory (intermediate result needed for δu since matrix sqrt is not ForwardDiff compl.)
    gen_KEK(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D
        X,U,H = unpackZ(Z)
        Ad = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdx(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
        Bd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdu(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
        Gd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdw(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]

        P = [zeros(eltype(Z),n,n) for k = 1:N]
        K = [zeros(eltype(Z),m,n) for k = 1:N-1]

        E = [zeros(eltype(Z),n,n) for k = 1:N]
        HH = [zeros(eltype(Z),n,r) for k = 1:N]
        KEK_vec = zeros(eltype(Z),(N-1)*m^2)

        P[N] = Qf_lqr
        E[1] = E0
        HH[1] = H0

        for k = N-1:-1:1
            K[k] = (R_lqr + Bd[k]'*P[k+1]*Bd[k])\(Bd[k]'*P[k+1]*Ad[k])
            P[k] = Q_lqr + K[k]'*R_lqr*K[k] + (Ad[k] - Bd[k]*K[k])'*P[k+1]*(Ad[k] - Bd[k]*K[k])
        end

        for k = 1:N-1
            KEK_vec[(k-1)*(m^2) .+ (1:m^2)] = vec(K[k]*E[k]*K[k]')

            Acl = Ad[k] - Bd[k]*K[k]
            E[k+1] = Acl*E[k]*Acl' + Gd[k]*HH[k]'*Acl' + Acl*HH[k]*Gd[k]' + Gd[k]*D*Gd[k]'
            HH[k+1] = Acl*HH[k] + Gd[k]*D
        end

        return KEK_vec
    end

    # Evaluate δu trajectory
    gen_δu(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D, eig_thr=eig_thr
        KEK = gen_KEK(Z)
        KEKsqrt = zero(KEK)

        for k = 1:N-1
            kek = reshape(KEK[(k-1)*m^2 .+ (1:m^2)],m,m)

            if !isposdef(kek)
                _eig = eigen(kek)
                _eig.values .= real.(_eig.values)
                for i = 1:m
                    if _eig.values[i] < 0.
                        _eig.values[i] = eig_thr
                    end
                end
                KEKsqrt[(k-1)*m^2 .+ (1:m^2)] = Diagonal(sqrt.(_eig.values))*_eig.vectors'
            else
                KEKsqrt[(k-1)*m^2 .+ (1:m^2)] = real.(vec(sqrt(kek)))
            end

        end

        return KEKsqrt
    end

    # Gradient of KEK traj. wrt decision variables
    dKEKdZ(Z) = ForwardDiff.jacobian(gen_KEK,Z)

    # Gradient of δu traj. wrt decision variables
    dδudZ(Z) = let n=n,m=m,N=N,NN=NN
        res = zeros(eltype(Z),(N-1)*m^2,NN)
        ∇KEK = dKEKdZ(Z)
        KEKsqrt = gen_δu(Z)
        Im = Diagonal(ones(m))

        for k = 1:N-1
            idx = (k-1)*m^2 .+ (1:m^2)
            _KEKsqrt = reshape(KEKsqrt[idx],m,m)

            res[idx,1:NN] = (kron(Im,_KEKsqrt) + kron(_KEKsqrt',Im))\∇KEK[idx,1:NN]
        end

        res
    end

    # Dynamics constraints
    dynamics_constraints!(g,Z) = let n=n, m=m, N=N, NN=NN, x0=x0, xf=xf
        Z_traj = unpackZ_timestep(Z)
        k = 1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]
        con = x-x0

        for k = 1:N-1
            y = Z_traj[k+1][1:n]
            x = Z_traj[k][1:n]
            u = Z_traj[k][n .+ (1:m)]
            h = Z_traj[k][n+m+1]
            append!(con,F(y,x,u,h))

            if k != N-1
                h₊ = Z_traj[k+1][n+m+1]
                append!(con,h - h₊)
            end
        end
        xN = Z_traj[N][1:n]
        append!(con,xN-xf)

        copyto!(g,con)
        return nothing
    end

    # Dynamics constraint jacobian
    ∇dynamics_constraints!(∇con,Z) = let n=n, m=m, N=N, NN=NN
        # ∇con = zeros(p,NN)
        Z_traj = unpackZ_timestep(Z)

        shift = 0
        # con = x-x0
        copyto!(view(∇con,1:n^2),vec(Diagonal(ones(n))))
        shift += n^2

        for k = 1:N-1
            y = Z_traj[k+1][1:n]
            x = Z_traj[k][1:n]
            u = Z_traj[k][n .+ (1:m)]
            h = Z_traj[k][n+m+1]

            # dynamics constraint jacobians
            copyto!(view(∇con,shift .+ (1:(n+m+1+n)*(n))),vec([dFdx(y,x,u,h) dFdu(y,x,u,h) dFdh(y,x,u,h) dFdy(y,x,u,h)]))
            shift += (n+m+1+n)*n

            if k != N-1
                copyto!(view(∇con,shift .+ (1:2)),[1.;-1.])
                shift += 2
            end
        end
        k = N
        xN = Z_traj[k][1:n]
        copyto!(view(∇con,shift .+ (1:n^2)),vec(Diagonal(ones(n))))
        shift += n^2

        return nothing
    end

    # Robust constraints
    robust_constraints!(g,Z) = let n=n, m=m, N=N, NN=NN
        pp = 0
        shift = 0
        # δx = gen_δx(Z)
        δu = gen_δu(Z)

        X,U,H = unpackZ(Z)

        for k = 1:N-1
            # δx_k = δx[(k-1)*n^2 .+ (1:n^2)]
            Xr = [X[k]]
            δx_sign = [0]
            # for i = 1:n
            #     δx_k_i = δx_k[(i-1)*n .+ (1:n)]
            #     push!(Xr,X[k]+δx_k_i)
            #     push!(Xr,X[k]-δx_k_i)
            #
            #     push!(δx_sign,1)
            #     push!(δx_sign,-1)
            # end

            δu_k = δu[(k-1)*m^2 .+ (1:m^2)]
            Ur = [U[k]]
            δu_sign = [0]
            for j = 1:m
                δu_k_j = δu_k[(j-1)*m .+ (1:m)]
                push!(Ur,U[k]+δu_k_j)
                push!(Ur,U[k]-δu_k_j)

                push!(δu_sign,1)
                push!(δu_sign,-1)
            end

            con_ineq, con_eq = split(prob.constraints[k])
            for con in con_ineq
                p_con = length(con)

                for (i,xr) in enumerate(Xr)
                    for (j,ur) in enumerate(Ur)
                        if !(j == i && (i == 1 || j == 1))
                            # println("$i,$j")
                            if p_con != 0
                                evaluate!(view(g,shift .+ (1:p_con)),con,xr,ur)
                                shift += p_con
                            end
                        end
                    end
                end
            end
        end

        # k = N
        # con_ineq, con_eq = split(prob.constraints[N])
        # for con in con_ineq
        #     if con isa BoundConstraint
        #         p_con = sum(con.active.x_max + con.active.x_min)
        #     else
        #         p_con = length(con)
        #     end
        #     pp += p_con*(2*n)
        #     δx_N = δx[(k-1)*n^2 .+ (1:n^2)]
        #     Xr = []
        #     δxN_sign = []
        #     for i = 1:n
        #         δx_N_i = δx_N[(i-1)*n .+ (1:n)]
        #         push!(Xr,X[k]+δx_N_i)
        #         push!(Xr,X[k]-δx_N_i)
        #
        #         push!(δxN_sign,1)
        #         push!(δxN_sign,-1)
        #     end
        #
        #     for (i,xr) in enumerate(Xr)
        #         if p_con != 0
        #             evaluate!(view(g,shift .+ (1:p_con)),con,xr)
        #             shift += p_con
        #         end
        #     end
        # end

        return nothing
    end

    # Robust constraints jacobian
    ∇robust_constraints!(G,Z) = let n=n, m=m, N=N, NN=NN
        pp = 0
        shift = 0
        # δx = gen_δx(Z)
        δu = gen_δu(Z)

        # ∇δx = dδxdZ(Z)
        ∇δu = dδudZ(Z)

        X,U,H = unpackZ(Z)

        In = Diagonal(ones(n))
        Im = Diagonal(ones(m))

        for k = 1:N-1
            # ∇δx_k = ∇δx[(k-1)*n^2 .+ (1:n^2),1:NN]
            ∇δu_k = ∇δu[(k-1)*m^2 .+ (1:m^2),1:NN]

            x_idx = (k-1)*(n+m+1) .+ (1:n)
            u_idx = ((k-1)*(n+m+1) + n) .+ (1:m)

            # δx_k = δx[(k-1)*n^2 .+ (1:n^2)]
            Xr = [X[k]]
            δx_sign = [(0,0)] # (sign,column index)
            # for i = 1:n
            #     δx_k_i = δx_k[(i-1)*n .+ (1:n)]
            #     push!(Xr,X[k]+δx_k_i)
            #     push!(Xr,X[k]-δx_k_i)
            #
            #     push!(δx_sign,(1,i))
            #     push!(δx_sign,(-1,i))
            # end

            δu_k = δu[(k-1)*m^2 .+ (1:m^2)]
            Ur = [U[k]]
            δu_sign = [(0,0)]
            for j = 1:m
                δu_k_j = δu_k[(j-1)*m .+ (1:m)]
                push!(Ur,U[k]+δu_k_j)
                push!(Ur,U[k]-δu_k_j)

                push!(δu_sign,(1,j))
                push!(δu_sign,(-1,j))
            end

            con_ineq, con_eq = split(prob.constraints[k])
            for con in con_ineq
                p_con = length(con)

                pp += p_con*((2*n+1)*(2*m+1) - 1)

                C = zeros(p_con,n+m)
                jacobian!(C,con,X[k],U[k])
                Cx = C[:,1:n]
                Cu = C[:,n .+ (1:m)]

                for (i,xr) in enumerate(Xr)
                    for (j,ur) in enumerate(Ur)
                        if !(j == i && (i == 1 || j == 1))
                            # println("$i,$j")
                            if p_con != 0
                                if false #i != 1
                                    ∇δx_k_i = δx_sign[i][1]*∇δx_k[(δx_sign[i][2]-1)*n .+ (1:n),1:NN]
                                else # no δx
                                    ∇δx_k_i = zeros(n,NN)
                                end

                                if j != 1
                                    ∇δu_k_j = δu_sign[j][1]*∇δu_k[(δu_sign[j][2]-1)*m .+ (1:m),1:NN]
                                else # no δu
                                    ∇δu_k_j = zeros(m,NN)
                                end

                                ∇δx_k_i[1:n,x_idx] += In
                                ∇δu_k_j[1:m,u_idx] += Im

                                tmp = Cu*∇δu_k_j + Cx*∇δx_k_i

                                G[shift .+ (1:p_con*NN)] = vec(tmp)
                                shift += p_con*NN
                            end
                        end
                    end
                end
            end
        end

        # k = N
        # con_ineq, con_eq = split(prob.constraints[N])
        # for con in con_ineq
        #     if con isa BoundConstraint
        #         p_con = sum(con.active.x_max + con.active.x_min)
        #     else
        #         p_con = length(con)
        #     end
        #     pp += p_con*(2*n)
        #     δx_N = δx[(k-1)*n^2 .+ (1:n^2)]
        #     Xr = []
        #     δxN_sign = []
        #     for i = 1:n
        #         δx_N_i = δx_N[(i-1)*n .+ (1:n)]
        #         push!(Xr,X[k]+δx_N_i)
        #         push!(Xr,X[k]-δx_N_i)
        #
        #         push!(δxN_sign,(1,i))
        #         push!(δxN_sign,(-1,i))
        #     end
        #
        #     CxN = zeros(p_con,n)
        #     jacobian!(CxN,con,X[N])
        #
        #     for (i,xr) in enumerate(Xr)
        #         if p_con != 0
        #             ∇δx_N_i = copy(∇δx_N[(δxN_sign[i][2]-1)*n .+ (1:n),1:NN])
        #             ∇δx_N_i[1:n,x_idx] += δxN_sign[i][1]*In
        #
        #             G[shift .+ (1:p_con*NN)] = vec(Cx*∇δx_N_i)
        #             shift += p_con*NN
        #         end
        #     end
        # end

        return nothing
    end

    return cost, ∇cost!, robust_cost, ∇robust_cost, dynamics_constraints!,∇dynamics_constraints!,robust_constraints!,∇robust_constraints!, packZ, unpackZ
end

"Add row and column indices to existing lists"
function add_rows_cols!(row,col,_r,_c)
    for cc in _c
        for rr in _r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
end

"Generate sparsity indices for dynamics constraint jacobian"
function dynamics_constraints_sparsity(n,m,N,NN)

    row = []
    col = []

    r = 1:n
    c = 1:n

    add_rows_cols!(row,col,r,c)

    for k = 1:N-1
        # dynamics constraint jacobians
        c_idx = ((k-1)*(n+m+1)) .+ (1:(n+m+1+n))
        r_idx = (n + (k-1)*(n + 1)) .+ (1:n)

        add_rows_cols!(row,col,r_idx,c_idx)

        if k != N-1
            c_idx = ((k-1)*(n+m+1) + n + m) .+ 1
            c2_idx = ((k)*(n+m+1) + n + m) .+ 1
            r_idx = (n + n + (k-1)*(n + 1)) .+ 1

            add_rows_cols!(row,col,r_idx,c_idx)

            add_rows_cols!(row,col,r_idx,c2_idx)
        end
    end

    k = N
    r = ((n + (k-1)*(n+1) - 1) .+ (1:n))
    c = (NN - (n-1)):NN

    add_rows_cols!(row,col,r,c)

    return collect(zip(row,col))
end

"Calculate number of robust constraints and non-zero jacobian elements"
function num_robust_constraints(prob::TrajectoryOptimization.Problem)
    n = prob.model.n; m = prob.model.m; N = prob.N
    NN = n*N + m*(N-1) + (N-1)

    p = 0
    p_jac = 0
    for k = 1:N-1
        con_ineq, con_eq = split(prob.constraints[k])
        for con in con_ineq
            p_con = length(con)
            # p += p_con*((2*n+1)*(2*m+1) - 1)
            # p_jac += p_con*((2*n+1)*(2*m+1) - 1)*NN
            p += p_con*(2*m)
            p_jac += p_con*(2*m)*NN
        end
    end

    # con_ineq, con_eq = split(prob.constraints[N])
    # for con in con_ineq
    #     if con isa BoundConstraint
    #         p_con = sum(con.active.x_max + con.active.x_min)
    #     else
    #         p_con = length(con)
    #     end
    #     p += p_con*(2*n)
    #     p_jac += p_con*(2*n)*NN
    # end

    return p,p_jac
end

"Return robust constraint jacobian non-zero element indices"
function robust_constraints_sparsity(prob,n,m,N,NN,shift=0)
    idx_col = 1:NN

    row = []
    col = []

    for k = 1:N-1
        con_ineq, con_eq = split(prob.constraints[k])
        for con in con_ineq
            p_con = length(con)

            for i = 1:1#(2*n + 1)
                for j = 1:(2*m + 1)
                    if !(j == i && (i == 1 || j == 1))
                        # println("$i,$j")
                        if p_con != 0
                            idx_row = shift .+ (1:p_con)
                            add_rows_cols!(row,col,idx_row,idx_col)
                            shift += p_con
                        end
                    end
                end
            end
        end
    end

    # k = N
    # con_ineq, con_eq = split(prob.constraints[N])
    # for con in con_ineq
    #     if con isa BoundConstraint
    #         p_con = sum(con.active.x_max + con.active.x_min)
    #     else
    #         p_con = length(con)
    #     end
    #
    #     for i = 1:2*n
    #         if p_con != 0
    #
    #             idx_row = shift .+ (1:p_con)
    #             add_rows_cols!(row,col,idx_row,idx_col)
    #             shift += p_con
    #         end
    #     end
    # end

    return collect(zip(row,col))
end

"Robust constraint bounds"
function robust_constraint_bounds(prob)
    p, = num_robust_constraints(prob)
    bL = -Inf*ones(p)
    bU = zeros(p)

    return bL,bU
end

"Bounds on X, U trajectories from BoundConstraints"
function get_XU_bounds!(prob::Problem)
    n,m,N = size(prob)
    bounds = [BoundConstraint(n,m) for k = 1:prob.N]

    # All time steps
    for k = 1:prob.N
        bnd = remove_bounds!(prob.constraints[k])
        if !isempty(bnd)
            bounds[k] = bnd[1]::BoundConstraint
        end
    end

    return bounds
end

"Bounds on X,U,H trajectories"
function primal_bounds(prob,h_max,h_min)
    bnds = get_XU_bounds!(copy(prob))

    Z_low = []
    Z_up = []

    for k = 1:N-1
        append!(Z_low,[bnds[1].x_min;bnds[1].u_min;h_min])
        append!(Z_up,[bnds[1].x_max;bnds[1].u_max;h_max])
    end
    append!(Z_low,bnds[1].x_min)
    append!(Z_up,bnds[1].x_max)

    return Z_low, Z_up
end

"Dynamics constraint bounds (all zeros for equality constraints)"
function dynamics_constraint_bounds(p::Int)
    ceq_low = zeros(p); ceq_up = zeros(p)

    return ceq_low,ceq_up
end

# MOI functions
MOI.features_available(pd::DIRTRELProblem) = [:Grad, :Jac]
MOI.initialize(pd::DIRTRELProblem, features) = nothing
MOI.jacobian_structure(pd::DIRTRELProblem) = pd.jac_struct
MOI.hessian_lagrangian_structure(pd::DIRTRELProblem) = []
#
function MOI.eval_objective(pd::DIRTRELProblem, Z)
    return pd.cost(Z) + pd.robust_cost(Z)
end

function MOI.eval_objective_gradient(pd::DIRTRELProblem, grad_f, Z)
    pd.∇cost!(grad_f, Z)
    grad_f + pd.∇robust_cost(Z)
end

function MOI.eval_constraint(pd::DIRTRELProblem, g, Z)
    g_dynamics = view(g,1:pd.p_dynamics)
    g_robust = view(g,pd.p_dynamics .+ (1:pd.p_robust))
    pd.dynamics_constraints!(g_dynamics,Z)
    pd.robust_constraints!(g_robust,Z)
end

function MOI.eval_constraint_jacobian(pd::DIRTRELProblem, jac, Z)
    jac_dynamics = view(jac,1:pd.p_dynamics_jac)
    jac_robust = view(jac,pd.p_dynamics_jac .+ (1:pd.p_robust_jac))
    pd.∇dynamics_constraints!(jac_dynamics,Z)
    pd.∇robust_constraints!(jac_robust,Z)
end

MOI.eval_hessian_lagrangian(pr::DIRTRELProblem, H, x, σ, μ) = nothing

"Method for solving DIRTREL problem"
function solve!(prob::Problem,solver::DIRTRELSolver)
    # Create DIRTREL problem
    pd = DIRTRELProblem(prob,solver)

    # Initial guess for decision variables
    ZZ = pd.packZ(prob.X,prob.U,[prob.dt for k = 1:N-1])

    # bounds on primals and jacobian block
    Z_low, Z_up = primal_bounds(prob,solver.h_max,solver.h_min)
    c_low, c_up = dynamics_constraint_bounds(pd.p_dynamics)
    r_low, r_up = robust_constraint_bounds(pd.prob)

    g_low = [c_low;r_low]; g_up = [c_up;r_up]

    nlp_bounds = MOI.NLPBoundsPair.(g_low,g_up)
    block_data = MOI.NLPBlockData(nlp_bounds,pd,true)

    # NLP solver
    opt_solver = Ipopt.Optimizer(tol=1.0e-3,constr_viol_tol=1.0e-2)

    Z = MOI.add_variables(opt_solver,pd.NN)

    for i = 1:pd.NN
        zi = MOI.SingleVariable(Z[i])
        MOI.add_constraint(opt_solver, zi, MOI.LessThan(Z_up[i]))
        MOI.add_constraint(opt_solver, zi, MOI.GreaterThan(Z_low[i]))
        MOI.set(opt_solver, MOI.VariablePrimalStart(), Z[i], ZZ[i])
    end

    # Solve the problem
    MOI.set(opt_solver, MOI.NLPBlock(), block_data)
    MOI.set(opt_solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(opt_solver)

    # Get the solution
    res = MOI.get(opt_solver, MOI.VariablePrimal(), Z)

    # Update problem w/ solution
    X,U,H = pd.unpackZ(res)
    copyto!(prob.X,X)
    copyto!(prob.U,U)
end
