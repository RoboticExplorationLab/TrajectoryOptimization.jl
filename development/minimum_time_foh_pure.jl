using TrajectoryOptimization: rk3_foh, fd_augmented_foh!,norm,norm2,to_array,copyto!
using ForwardDiff
using Statistics
using LinearAlgebra
using Plots
### Solver options ###
z_min = 1e-8
z_max = 10.0
gradient_intermediate_tolerance = 1e-4
cost_intermediate_tolerance = 1e-3
constraint_tolerance = 1e-3
gradient_tolerance = 1e-4
cost_tolerance = 1e-3
iterations_outerloop = 50
iterations = 250
iterations_linesearch = 10
τ = 0.5
γ = 10.0
γ_no = 1.0
ρ = 0.0
dρ = 1.0
ρ_factor = 1.6
ρ_max = 1e8
ρ_min = 1e-8
μ_max = 1e8
live_plotting = false

######################

### Set up model, objective, solver ###
# Model, objective (unconstrained)
dt = 0.1
function pendulum_dynamics!(xdot,x,u)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    xdot[1] = x[2]
    xdot[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/I
end
n,m = 2,1

m̄ = m + 1
mm = m̄
p = 2*m + 2 + 1
pI = 4

f_discrete = rk3_foh(pendulum_dynamics!,dt)
f_discrete_aug = fd_augmented_foh!(f_discrete,n,m)

nm = n + m + 1 + m + 1

# Initialize discrete and continuous dynamics Jacobians
Jd = zeros(nm, nm)
Sd = zeros(nm)
Sdotd = zero(Sd)
Fd!(Jd,Sdotd,Sd) = ForwardDiff.jacobian!(Jd,f_discrete_aug,Sdotd,Sd)

function f_discrete_jacobians!(x,u,v)
    # Assign state, control (and h = sqrt(dt)) to augmented vector
    Sd[1:n] = x
    Sd[n+1:n+m] = u[1:m]
    Sd[n+m+1] = u[m̄]
    Sd[n+m+1+1:n+m+1+m] = v[1:m]
    Sd[n+m+1+m+1] = v[m̄]

    # Calculate Jacobian
    Fd!(Jd,Sdotd,Sd)
    return Jd[1:n,1:n], Jd[1:n,n+1:n+m̄], Jd[1:n,n+m+1+1:n+m+1+m̄] # fx, fū, fv̄
end

# pendulum
u_min = -2
u_max = 2
dt_max = 0.2
dt_min = 0.1
R_mintime = 1.0
Wf = Array(0.0*Matrix(I,n,n))
x0 = [0.;0.]
xf = [pi; 0.0]

N = 50

X = [zeros(n) for k in 1:N]
U = [zeros(m̄) for k in 1:N]

X̄ = [zeros(n) for k in 1:N]
Ū = [zeros(m̄) for k in 1:N]

K = [zeros(m̄,n) for k in 1:N]
b = [zeros(m̄,m̄) for k in 1:N]
d = [zeros(m̄) for k in 1:N]

c = [zeros(p) for k = 1:N]
c_prev = [zeros(p) for k = 1:N]
Iμ = [zeros(p,p) for k in 1:N]
λ = [zeros(p) for k in 1:N]
μ = [ones(p) for k in 1:N]

cN = zeros(n)
cN_prev = zeros(n)
IμN = zeros(n,n)
λN = zeros(n)
μN = ones(n)

# Key functions for first-order hold minimum time
function constraints_mintime!(c,x,u,y,v)
    c[1:m] = u[1:m] .- u_max
    c[m̄] = u[m̄] - sqrt(dt_max)
    c[m̄+1:m̄+m] = u_min .- u[1:m]
    c[m̄+m̄] = sqrt(dt_min) - u[m̄]
    c[m̄+m̄+1] = u[m̄] - v[m̄]
    return c
end

function constraints_mintime!(c,x)
    c[1:n] = (x - xf)
    return c
end

function update_constraints_mintime!(X,U,c,Iμ,μ,λ,cN,IμN,μN,λN)
    for k = 1:N-1
        constraints_mintime!(c[k],X[k],U[k],X[k+1],U[k+1])
    end
    constraints_mintime!(c[N],X[N],U[N],zeros(n),zeros(m̄))

    c[N][m̄] = 0.0
    c[N][m̄+m̄] = 0.0
    c[N-1][m̄+m̄+1] = 0.0
    c[N][m̄+m̄+1] = 0.0

    constraints_mintime!(cN,X[N])
    for k = 1:N
        for j = 1:pI
            if c[k][j] >= 0.0 || λ[k][j] > 0.0
                Iμ[k][j,j] = μ[k][j] # active (or previously active) inequality constraints are penalized
            else
                Iμ[k][j,j] = 0. # inactive inequality constraints are not penalized
            end
        end
        # Equality constraints
        for j = pI+1:p
            Iμ[k][j,j] = μ[k][j] # equality constraints are penalized
        end
    end

    # Terminal constraint
    IμN .= Diagonal(μN)

    return c,Iμ,cN,IμN
end

cx = zeros(p,n)

cu = zeros(p,m̄)
cu[1:m̄,1:m̄] = Array(Matrix(I,m̄,m̄))
cu[m̄+1:m̄+m̄,1:m̄] = Array(-1.0*Matrix(I,m̄,m̄))
cu[m̄+m̄+1,m̄] = 1

cy = zeros(p,m̄)

cv = zeros(p,m̄)
cv[m̄+m̄+1,m̄] = -1

cxN = Array(Matrix(I,n,n))

function cost_mintime(X,U,c,Iμ,λ,cN,IμN,λN)
    J = 0.0
    # Stage costs
    for k = 1:N-1
        J += R_mintime*U[k][m̄]^2
        J += 0.5*c[k]'*Iμ[k]*c[k] + λ[k]'*c[k]
    end
    J += 0.5*c[N]'*Iμ[N]*c[N] + λ[N]'*c[N]

    # Terminal cost
    J += 0.5*cN'IμN*cN + λN'*cN
    J += 0.5*(X[N] - xf)'*Wf*(X[N] - xf)

    return J
end

function _backwardpass_foh_mintime!(X,U,K,b,d,c,Iμ,λ,cN,IμN,λN,ρ,dρ)
    # Boundary conditions
    S = zeros(n+mm,n+mm)
    s = zeros(n+mm)
    S[1:n,1:n] = Wf
    s[1:n] = Wf*(X[N]-xf)

    # Terminal constraints
    S[1:n,1:n] += cxN'*IμN*cxN
    s[1:n] += cxN'*IμN*cN + cxN'*λN

    # Include the the k = N expansions here for a cleaner backward pass
    # c, cx, cu, Iμ, λ = c[N], cx[N], cu[N], Iμ[N], λ[N]

    s[1:n] += cx'*Iμ[N]*c[N] + cx'*λ[N]
    s[n+1:n+mm] += cu'*Iμ[N]*c[N] + cu'*λ[N]

    S[1:n,1:n] += cx'*Iμ[N]*cx
    S[n+1:n+mm,n+1:n+mm] += cu'*Iμ[N]*cu
    S[1:n,n+1:n+mm] += cx'*Iμ[N]*cu
    S[n+1:n+mm,1:n] += cu'*Iμ[N]*cx

    # Create a copy of boundary conditions in case of regularization
    SN = copy(S)
    sN = copy(s)

    # Backward pass
    k = N-1
    Δv = [0. 0.] # Initialization of expected change in cost-to-go
    while k >= 1
        # Unpack results
        fdx, fdu, fdv = f_discrete_jacobians!(X[k],U[k],U[k+1])

        h = U[k][m̄]

        # Assemble expansion
        Lx = zeros(n)
        Lu = [zeros(m); (2*R_mintime*h)]
        Ly = zeros(n)
        Lv = zeros(m̄)

        Lxx = zeros(n,n)
        Luu = [zeros(m,m) zeros(m); zeros(m)' 2*R_mintime]
        Lyy = zeros(n,n)
        Lvv = zeros(m̄,m̄)

        Lxu = zeros(n,m̄)
        Lxy = zeros(n,n)
        Lxv = zeros(n,m̄)
        Luy = zeros(m̄,n)
        Luv = zeros(m̄,m̄)
        Lyv = zeros(n,m̄)

        # Constraints
        # c, cx, cu, Iμ, λ = c[k], cx[k], cu[k], Iμ[k], λ[k]
        Lx += cx'*Iμ[k]*c[k] + cx'*λ[k]
        Lu += cu'*Iμ[k]*c[k] + cu'*λ[k]
        Ly += cy'*Iμ[k]*c[k] + cy'*λ[k]
        Lv += cv'*Iμ[k]*c[k] + cv'*λ[k]

        Lxx += cx'*Iμ[k]*cx
        Luu += cu'*Iμ[k]*cu
        Lyy += cy'*Iμ[k]*cy
        Lvv += cv'*Iμ[k]*cv

        Lxu += cx'*Iμ[k]*cu
        Lxy += cx'*Iμ[k]*cy
        Lxv += cx'*Iμ[k]*cv
        Luy += cu'*Iμ[k]*cy
        Luv += cu'*Iμ[k]*cv
        Lyv += cy'*Iμ[k]*cv


        # Unpack cost-to-go P
        Sy = s[1:n]
        Sv = s[n+1:n+mm]
        Syy = S[1:n,1:n]
        Svv = S[n+1:n+mm,n+1:n+mm]
        Syv = S[1:n,n+1:n+mm]

        # Substitute in discrete dynamics (second order approximation)
        Qx = vec(Lx) + fdx'*vec(Ly) + fdx'*vec(Sy)
        Qu = vec(Lu) + fdu'*vec(Ly) + fdu'*vec(Sy)
        Qv = vec(Lv) + fdv'*vec(Ly) + fdv'*vec(Sy) + Sv

        Qxx = Lxx + Lxy*fdx + fdx'*Lxy' + fdx'*Lyy*fdx + fdx'*Syy*fdx
        Quu = Luu + Luy*fdu + fdu'*Luy' + fdu'*Lyy*fdu + fdu'*Syy*fdu
        Qvv = Lvv + Lyv'*fdv + fdv'*Lyv + fdv'*Lyy*fdv + fdv'*Syy*fdv + fdv'*Syv + Syv'*fdv + Svv
        Qxu = Lxu + Lxy*fdu + fdx'*Luy' + fdx'*Lyy*fdu + fdx'*Syy*fdu
        Qxv = Lxv + Lxy*fdv + fdx'*Lyv + fdx'*Lyy*fdv + fdx'*Syy*fdv + fdx'*Syv
        Quv = Luv + Luy*fdv + fdu'*Lyv + fdu'*Lyy*fdv + fdu'*Syy*fdv + fdu'*Syv

        # Regularization
        Qvv_reg = Qvv + ρ*I
        Qxv_reg = Qxv
        Quv_reg = Quv

        if !isposdef(Hermitian(Array(Qvv_reg)))
            ρ,dρ = ρ,dρ = regularization_update(ρ,dρ,:increase)
            # Reset BCs
            S = copy(SN)
            s = copy(sN)
            ############
            k = N-1
            Δv = [0. 0.]
            continue
        end

        # calculate gains
        K[k+1] = -Qvv_reg\Qxv_reg'
        b[k+1] = -Qvv_reg\Quv_reg'
        d[k+1] = -Qvv_reg\vec(Qv)

        # calculate optimized values
        Q̄x = vec(Qx) + K[k+1]'*vec(Qv) + Qxv*vec(d[k+1]) + K[k+1]'Qvv*d[k+1]
        Q̄u = vec(Qu) + b[k+1]'*vec(Qv) + Quv*vec(d[k+1]) + b[k+1]'*Qvv*d[k+1]
        Q̄xx = Qxx + Qxv*K[k+1] + K[k+1]'*Qxv' + K[k+1]'*Qvv*K[k+1]
        Q̄uu = Quu + Quv*b[k+1] + b[k+1]'*Quv' + b[k+1]'*Qvv*b[k+1]
        Q̄xu = Qxu + K[k+1]'*Quv' + Qxv*b[k+1] + K[k+1]'*Qvv*b[k+1]

        # cache (approximate) cost-to-go at timestep k
        s[1:n] = Q̄x
        s[n+1:n+mm] = Q̄u
        S[1:n,1:n] = Q̄xx
        S[n+1:n+mm,n+1:n+mm] = Q̄uu
        S[1:n,n+1:n+mm] = Q̄xu
        S[n+1:n+mm,1:n] = Q̄xu'

        # line search terms
        Δv += [vec(Qv)'*vec(d[k+1]) 0.5*vec(d[k+1])'*Qvv*vec(d[k+1])]

        if k == 1
            # regularize Quu_
            Q̄uu_reg = Q̄uu + ρ*I

            if !isposdef(Array(Hermitian(Q̄uu_reg)))
                ρ,dρ = regularization_update(ρ,dρ,:increase)

                ## Reset BCs ##
                S = copy(SN)
                s = copy(sN)
                ################
                k = N-1
                Δv = [0. 0.]
                continue
            end

            K[1] = -Q̄uu_reg\Q̄xu'
            b[1] = zeros(mm,mm)
            d[1] = -Q̄uu_reg\vec(Q̄u)

            Δv += [vec(Q̄u)'*vec(d[1]) 0.5*vec(d[1])'*Q̄uu*vec(d[1])]
        end

        k = k - 1;
    end
    ρ,dρ = regularization_update(ρ,dρ,:increase)
    return Δv, K,b,d,ρ,dρ
end

function forwardpass_mintime!(X,U,X̄,Ū,K,b,d,c,Iμ,λ,cN,IμN,λN,Δv,ρ,dρ)
    J_prev = cost_mintime(X,U,c,Iμ,λ,cN,IμN,λN)

    J = Inf
    alpha = 1.0
    iter = 0
    z = 0.
    expected = 0.

    while z ≤ z_min || z > z_max

        # Check that maximum number of line search decrements has not occured
        if iter > 11

            # set trajectories to original trajectory
            X̄ .= deepcopy(X)
            Ū .= deepcopy(U)

            update_constraints_mintime!(X̄,Ū,c,Iμ,μ,λ,cN,IμN,μN,λN)
            J = copy(J_prev)
            z = 0.
            alpha = 0.0
            expected = 0.

            ρ,dρ = regularization_update(ρ,dρ,:increase) # increase regularization
            ρ += 1
            break
        end

        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout_mintime!(X,U,X̄,Ū,K,b,d,c,Iμ,λ,cN,IμN,λN,alpha)

        # Check if rollout completed
        if ~flag
            iter += 1
            alpha /= 2.0
            continue
        end

        # Calcuate cost
        J = cost_mintime(X̄,Ū,c,Iμ,λ,cN,IμN,λN)

        expected = -alpha*(Δv[1] + alpha*Δv[2])
        if expected > 0
            z  = (J_prev - J)/expected
        else
            z = -1
        end

        iter += 1
        alpha /= 2.0
    end

    println("J: $J")
    println("   α: $(2*alpha)")
    println("   z: $z")
    println("   ρ: $ρ")

    return J,X̄,Ū,c,Iμ,cN,IμN, ρ, dρ
end

function rollout_initial!(X,U,c,Iμ,μ,λ,cN,IμN,μN,λN)
    X[1] = x0
    for k = 1:N-1
        dt = U[k][m̄]^2
        f_discrete(X[k+1], X[k], U[k][1:m], U[k+1][1:m], dt) # get new state

        # Check that rollout has not diverged
        if ~(norm(X[k+1],Inf) < 1e8 && norm(U[k],Inf) < 1e8)
            return false, X,U, c,Iμ,cN,IμN
        end
    end
    update_constraints_mintime!(X,U,c,Iμ,μ,λ,cN,IμN,μN,λN)

    return true, X,U, c,Iμ,cN,IμN
end

function rollout_mintime!(X,U,X̄,Ū,K,b,d,c,Iμ,μ,λ,cN,IμN,μN,λN,alpha)
    X̄[1] = x0;

    dv = zeros(mm)
    du = alpha*d[1]
    Ū[1] = U[1] + du

    for k = 2:N
        dt = Ū[k-1][m̄]^2
        δx = X̄[k-1] - X[k-1]

        dv = K[k]*δx + b[k]*du + alpha*d[k]
        Ū[k] = U[k] + dv
        f_discrete(X̄[k], X̄[k-1], Ū[k-1][1:m], Ū[k][1:m], dt)
        du = copy(dv)

        # Check that rollout has not diverged
        if ~(norm(X̄[k],Inf) < 1e8 && norm(Ū[k-1],Inf) < 1e8)
            return X, U, c,Iμ,cN,IμN
        end
    end

    update_constraints_mintime!(X̄,Ū,c,Iμ,μ,λ,cN,IμN,μN,λN)

    return true, X̄,Ū, c,Iμ,cN,IμN
end

function regularization_update(ρ,dρ,status::Symbol=:increase)
    if status == :increase # increase regularization
        dρ = max(dρ*ρ_factor, ρ_factor)
        ρ = min(max(ρ*dρ, ρ_min), ρ_max)
        if ρ >= ρ_max
            error("Max regularization exceeded")
        end
    elseif status == :decrease # decrease regularization
        dρ = min(dρ/ρ_factor, 1.0/ρ_factor)
        ρ = max(ρ*dρ*(ρ*dρ>ρ_min), ρ_min)
    end
    return ρ,dρ
end

function _solve(X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ)
    flag, X,U,c,Iμ,cN,IμN = rollout_initial!(X,U,c,Iμ,μ,λ,cN,IμN,μN,λN) # rollout new state trajectoy

    dJ = Inf
    gradient = Inf
    Δv = [Inf, Inf]

    for j = 1:iterations_outerloop

        if j == 1
            c_prev .= deepcopy(c)
            cN_prev .= deepcopy(cN)
        end

        c_max = 0.0  # Initialize max constraint violation to increase scope
        J_prev = cost_mintime(X,U,c,Iμ,λ,cN,IμN,λN)

        #****************************#
        #         INNER LOOP         #
        #****************************#

        for ii = 1:iterations

            # BACKWARD PASS
            Δv, K,b,d,ρ,dρ = _backwardpass_foh_mintime!(X,U,K,b,d,c,Iμ,λ,cN,IμN,λN,ρ,dρ)

            # FORWARDS PASS
            J,X̄,Ū,c,Iμ,cN,IμN, ρ, dρ = forwardpass_mintime!(X,U,X̄,Ū,K,b,d,c,Iμ,λ,cN,IμN,λN,Δv,ρ,dρ)

            # UPDATE RESULTS
            X .= deepcopy(X̄)
            U .= deepcopy(Ū)

            dJ = copy(abs(J_prev-J)) # change in cost
            J_prev = copy(J)

            # live plotting for debugging
            if live_plotting
                p1 = plot(to_array(X)',label="",ylabel="state")
                p2 = plot(to_array(U)[:,1:N-1]',label="",ylabel="control")
                plt = plot(p1,p2,layout=(2,1))
                display(plt)
            end
            c_max = max_violation(c,Iμ,cN)
            gradient = calculate_todorov_gradient(U,d)

            println("   c_max: $c_max")
            println("   gradient: $gradient")

            # Check for gradient convergence
            if (gradient < gradient_tolerance || (gradient < gradient_intermediate_tolerance && iteration != iterations_outerloop))
                break
            elseif (gradient < gradient_tolerance && c_max < constraint_tolerance)
                break
            end

            # Check for cost convergence
            if (dJ < cost_tolerance || (dJ < cost_intermediate_tolerance && iteration != iterations_outerloop))
                break
            elseif (dJ < cost_tolerance && c_max < constraint_tolerance)
                break
            end
        end

        #****************************#
        #      OUTER LOOP UPDATE     #
        #****************************#
        println("Outer loop: $j")
        for k = 1:N
            for i = 1:p
                if p <= pI
                    # multiplier update
                    λ[k][i] = λ[k][i] + μ[k][i]*c[k][i]
                    λ[k][i] = max(0.0,λ[k][i])
                    if max(0.0,c[k][i]) <= τ*max(0.0,c_prev[k][i])
                        # no penalty update
                        μ[k][i] = min(μ_max, γ_no*μ[k][i])
                    else
                        #penalty update
                        μ[k][i] = min(μ_max, γ*μ[k][i])
                    end
                else
                    # multiplier update
                    λ[k][i] = λ[k][i] + μ[k][i]*c[k][i]
                    if abs(c[k][i]) <= τ*abs(c_prev[k][i])
                        # no penalty update
                        μ[k][i] = min(μ_max, γ_no*μ[k][i])
                    else
                        # penalty update
                        μ[k][i] = min(μ_max, γ*μ[k][i])
                    end
                end
            end
        end

        # Terminal constraints
        for i = 1:n
            λN[i] = λN[i] + μN[i]*cN[i]
            if abs(cN[i]) <= τ*abs(cN_prev[i])
                μN[i] = min(μ_max, γ_no*μN[i])
            else
                μN[i] = min(μ_max, γ*μN[i])
            end
        end

        ρ = 0.0
        dρ = 1.0

        ## Store current constraints evaluations for next outer loop update
        c_prev .= deepcopy(c)
        cN_prev .= deepcopy(cN)

        if c_max < constraint_tolerance && (dJ < cost_tolerance || gradient < gradient_tolerance)
            break
        end
    end

    @info "***Solve Complete***"
    return X,U
end

function calculate_todorov_gradient(U,d)
    maxes = zeros(N)
    for k = 1:N
        maxes[k] = maximum(abs.(d[k])./(abs.(U[k]).+1))
    end
    return mean(maxes)
end

function max_violation(c,Iμ,cN)
    return max(maximum(norm.(map((x)->x.>0, Iμ) .* c, Inf)), norm(cN,Inf))
end

## Test
u0 = ones(m̄,N)
u0[m̄,:] .= sqrt(dt)
copyto!(U,u0)

flag, X,U,c,Iμ,cN,IμN = rollout_initial!(X,U,c,Iμ,μ,λ,cN,IμN,μN,λN) # rollout new state trajectory

#****************************#
#         INNER LOOP         #
#****************************#

function inner_loop(X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ)
    dJ = Inf
    gradient = Inf
    Δv = [Inf, Inf]


    c_prev = deepcopy(c)
    cN_prev = deepcopy(cN)

    c_max = 0.0  # Initialize max constraint violation to increase scope
    J_prev = cost_mintime(X,U,c,Iμ,λ,cN,IμN,λN)

    # BACKWARD PASS
    Δv, K,b,d,ρ,dρ = _backwardpass_foh_mintime!(X,U,K,b,d,c,Iμ,λ,cN,IμN,λN,ρ,dρ)

    # FORWARDS PASS
    J,X̄,Ū,c,Iμ,cN,IμN, ρ, dρ = forwardpass_mintime!(X,U,X̄,Ū,K,b,d,c,Iμ,λ,cN,IμN,λN,Δv,ρ,dρ)

    # UPDATE RESULTS
    X .= deepcopy(X̄)
    U .= deepcopy(Ū)

    dJ = copy(abs(J_prev-J)) # change in cost
    J_prev = copy(J)

    # live plotting for debugging
    if live_plotting
        p1 = plot(to_array(X)',label="",ylabel="state")
        p2 = plot(to_array(U)[:,1:N-1]',label="",ylabel="control")
        plt = plot(p1,p2,layout=(2,1))
        display(plt)
    end
    c_max = max_violation(c,Iμ,cN)
    gradient = calculate_todorov_gradient(U,d)

    println("   c_max: $c_max")
    println("   gradient: $gradient")

    # # Check for gradient convergence
    # if (gradient < gradient_tolerance || (gradient < gradient_intermediate_tolerance && iteration != iterations_outerloop))
    #     break
    # elseif (gradient < gradient_tolerance && c_max < constraint_tolerance)
    #     break
    # end
    #
    # # Check for cost convergence
    # if (dJ < cost_tolerance || (dJ < cost_intermediate_tolerance && iteration != iterations_outerloop))
    #     break
    # elseif (dJ < cost_tolerance && c_max < constraint_tolerance)
    #     break
    # end
    return X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ
end

#****************************#
#      OUTER LOOP UPDATE     #
#****************************#
function outer_loop(X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ)
    for k = 1:N
        for i = 1:p
            if p <= pI
                # multiplier update
                λ[k][i] = λ[k][i] + μ[k][i]*c[k][i]
                λ[k][i] = max(0.0,λ[k][i])
                if max(0.0,c[k][i]) <= τ*max(0.0,c_prev[k][i])
                    # no penalty update
                    μ[k][i] = min(μ_max, γ_no*μ[k][i])
                else
                    #penalty update
                    μ[k][i] = min(μ_max, γ*μ[k][i])
                end
            else
                # multiplier update
                λ[k][i] = λ[k][i] + μ[k][i]*c[k][i]
                if abs(c[k][i]) <= τ*abs(c_prev[k][i])
                    # no penalty update
                    μ[k][i] = min(μ_max, γ_no*μ[k][i])
                else
                    # penalty update
                    μ[k][i] = min(μ_max, γ*μ[k][i])
                end
            end
        end
    end

    # Terminal constraints
    for i = 1:n
        λN[i] = λN[i] + μN[i]*cN[i]
        if abs(cN[i]) <= τ*abs(cN_prev[i])
            μN[i] = min(μ_max, γ_no*μN[i])
        else
            μN[i] = min(μ_max, γ*μN[i])
        end
    end

    ρ = 0.0
    dρ = 1.0

    ## Store current constraints evaluations for next outer loop update
    c_prev .= deepcopy(c)
    cN_prev .= deepcopy(cN)

    # if c_max < constraint_tolerance && (dJ < cost_tolerance || gradient < gradient_tolerance)
    #     break
    # end
    return X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ
end

## Test
u0 = ones(m̄,N)
u0[m̄,:] .= sqrt(dt)
copyto!(U,u0)

# flag, X,U,c,Iμ,cN,IμN = rollout_initial!(X,U,c,Iμ,μ,λ,cN,IμN,μN,λN) # rollout new state trajectory
ρ = 0.0
X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ = inner_loop(X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ)
# X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ = outer_loop(X,U,X̄,Ū,K,b,d,c,c_prev,Iμ,μ,λ,cN,cN_prev,IμN,μN,λN,ρ,dρ)

Iμ

plot(to_array(U)')
