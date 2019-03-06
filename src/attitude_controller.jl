"""
@(SIGNATURES)
    Generate an approximately continuous LQR controller from a discrete controller for use in simulation for attitude
        -requires X and U nominal trajectories and gains K, in matrix form; number of knot points and time step size from solve
        -option for first-order hold interpolation (cubic state, linear gains and control)
        -option for controller saturation with max/min limits
"""

function attitude_simulation(f!::Function,f_gains!::Function,integration,X_lqr::Matrix, U_lqr::Matrix,dt_lqr::Float64,x0_lqr::AbstractVector,tf::Float64,Q_lqr::AbstractArray{Float64,2}, R_lqr::AbstractArray{Float64,2}, Qf_lqr::AbstractArray{Float64,2})
    """
    X - (n,N)
    U - zoh: (m,N-1), foh: (m,N)

    """
    n=size(X_lqr,1)
    m=size(U_lqr,1)
    t_sim = t0:dt_lqr:tf
    if length(t_sim) > size(X_lqr,2)
        t_sim = t0:dt:(tf-dt)
    end

    # Discrete dynamics for simulation
    discretizer = eval(integration)
    fd! = discretizer(f!, dt)
    f_aug! = f_augmented!(f! , n , m )
    fd_aug! = discretizer(f_aug!)

    # Discrete dynamics for calculating gains
    fd_gains! = discretizer(f_gains!, dt)
    f_aug_gains! = f_augmented!(f_gains! , n , m )
    fd_aug_gains! = discretizer(f_aug_gains!)

    #determine number of knot points
    N_sim = length(t_sim)

    #find function dynamics (A,B,K)
    K = attitude_lqr(fd_aug_gains!,dt,X_lqr,U_lqr,Q_lqr,R_lqr,Qf_lqr)

    # Allocate memory for simulated state and control trajectories
    X_sim = zeros(n,N_sim)
    X_sim[:,1] = x0_lqr
    U_sim = zeros(m,N_sim)
    dX = zeros(n-2,N_sim)


    # simulate
    for (k,t) in enumerate(range(0,length=N_sim-1,stop=tf-dt))
        dX[1:3,k] = X_sim[1:3,k] - X_lqr[1:3,k]
        dX[4:6,k] = [zeros(3,1) Array(1.0*Diagonal(I,3))]*qmult(q_inv(X_lqr[4:7,k]),X_sim[4:7,k])
        U_sim[:,k] = U_lqr[:,k] - K[:,:,k]*dX[:,k]
        fd!(view(X_sim,:,k+1),X_sim[:,k],U_sim[:,k],dt)
    end
    return X_sim, U_sim, dX, K

end

function attitude_lqr(Aq::Array{Float64,3}, Bq::Array{Float64,3}, Q_lqr::AbstractArray{Float64,2}, R_lqr::AbstractArray{Float64,2}, Qf_lqr::AbstractArray{Float64,2},X_lqr::Matrix)
#generates gain and S matrices

    n,m,N = size(Bq)
    n = n-1

    # attribute matrices
    A = zeros(6,6,N)
    B = zeros(6,3,N)
    for k = 1:N-1
        #renormalize to find quaternion error gain matrices
        qk = X_lqr[4:7,k]
        sk = qk[1]
        vk = qk[2:4]

        qn = X_lqr[4:7,k+1]
        sn = qn[1]
        vn = qn[2:4]

        Gk = [-vk'; sk*Array(1.0*Diagonal(I,3)) + hat(vk)]
        Gn = [-vn'; sn*Array(1.0*Diagonal(I,3)) + hat(vn)]

        #recompute A and B matrices through permutation matrices
        perm_Gn = zeros(6,7)
        perm_Gk = zeros(7,6)
        perm_Gn[1:3,1:3]= Array(1.0*Diagonal(I,3))
        perm_Gn[4:6,4:7] = Gn'
        perm_Gk[1:3,1:3]= Array(1.0*Diagonal(I,3))
        perm_Gk[4:7,4:6] = Gk
        A[:,:,k] = perm_Gn*Aq[:,:,k]*perm_Gk
        B[:,:,k] = perm_Gn*Bq[:,:,k]
    end

    #calculate gains
    S = zeros(6,6,N)
    K = zeros(3,6,N-1)
    S[:,:,N] = Qf_lqr
    for k = (N-1):-1:1
        # Compute control gains
        K[:,:,k] = (R_lqr + B[:,:,k]'*S[:,:,k+1]*B[:,:,k])\(B[:,:,k]'*S[:,:,k+1]*A[:,:,k])        # Calculate cost-to-go for backward propagation
        S[:,:,k]= Q_lqr + K[:,:,k]'*R_lqr*K[:,:,k] + (A[:,:,k]-B[:,:,k]*K[:,:,k])'*S[:,:,k+1]*(A[:,:,k]-B[:,:,k]*K[:,:,k])
    end
    K
end

function attitude_lqr(fd_aug_gains!,dt,X_lqr,U_lqr,Q_lqr,R_lqr,Qf_lqr)
    n=size(X_lqr,1)
    m=size(U_lqr,1)
    N=size(X_lqr,2)
    nm1 = n + m + 1
    Jd = zeros(nm1, nm1)
    Sd = zeros(nm1)
    Sdotd = zero(Sd)
    Fd_gains!(Jd,Sdotd,Sd) = ForwardDiff.jacobian!(Jd,fd_aug_gains!,Sdotd,Sd)

    # Assign state, control (and dt) to augmented vector
    Aq=zeros(n-1,n-1,N) #ignore time in state
    Bq=zeros(n-1,m,N) #ignore time in state
    for k=1:N-1
        Sd[1:n] = X_lqr[:,k]
        Sd[n+1:n+m] = U_lqr[:,k]
        Sd[end] = dt

        # Calculate Jacobian
        Fd_gains!(Jd,Sdotd,Sd)
        Aq[:,:,k] = Jd[1:n-1,1:n-1] #ignore time
        Bq[:,:,k] = Jd[1:n-1,n+1:n+m] #ignore time
    end
    attitude_lqr(Aq,Bq,Q_lqr,R_lqr,Qf_lqr,X_lqr)
end


function rk4(f!::Function, dt::Float64)
    # Runge-Kutta 4
    fd!(xdot,x,u,dt=dt) = begin
        k1 = k2 = k3 = k4 = zero(xdot)
        f!(k1, x, u);         k1 *= dt;
        f!(k2, x + k1/2, u); k2 *= dt;
        f!(k3, x + k2/2, u); k3 *= dt;
        f!(k4, x + k3, u);    k4 *= dt;
        copyto!(xdot, x + (k1 + 2*k2 + 2*k3 + k4)/6)
    end
end

function rk4(f_aug!::Function)
    # Runge-Kutta 4
    fd!(dS,S::Array) = begin
        dt = S[end]^2
        k1 = k2 = k3 = k4 = zero(S)
        f_aug!(k1,S);         k1 *= dt;
        f_aug!(k2,S + k1/2); k2 *= dt;
        f_aug!(k3,S + k2/2); k3 *= dt;
        f_aug!(k4,S + k3);    k4 *= dt;
        copyto!(dS, S + (k1 + 2*k2 + 2*k3 + k4)/6)
    end
end


function f_augmented!(f!::Function, n::Int, m::Int)
    f_aug!(dS::AbstractArray, S::Array) = f!(dS, S[1:n], S[n+1:n+m])
end

function f_augmented(f::Function, n::Int, m::Int)
    f_aug(S::Array) = f(S[1:n], S[n+1:n+m])
end

function qrot(q,r)
      r + 2*cross(q[2:4],cross(q[2:4],r) + q[1]*r)
end

function qmult(q1,q2)
      [q1[1]*q2[1] - q1[2:4]'*q2[2:4]; q1[1]*q2[2:4] + q2[1]*q1[2:4] + cross(q1[2:4],q2[2:4])]
end

function q_inv(q)
    [q[1]; -q[2:4]]
end

function q_log(q)
    q[1]*q[2:4]
end

function hat(x)
    [  0   -x[3]  x[2]
         x[3]   0   -x[1]
        -x[2]  x[1]  0];
end

function rk3(f!::Function, dt::Float64) #TODO - test that this is correct
    # Runge-Kutta 3 (zero order hold)
    fd!(xdot,x,u,dt=dt) = begin
        k1 = k2 = k3 = zero(x)
        f!(k1, x, u);               k1 *= dt;
        f!(k2, x + k1/2, u);       k2 *= dt;
        f!(k3, x - k1 + 2*k2, u);  k3 *= dt;
        copyto!(xdot, x + (k1 + 4*k2 + k3)/6)
    end
end

function rk3(f_aug!::Function)
    # Runge-Kutta 3 augmented (zero order hold)
    fd!(dS,S::Array) = begin
        dt = S[end]^2
        k1 = k2 = k3 = zero(S)
        f_aug!(k1,S);              k1 *= dt;
        f_aug!(k2,S + k1/2);      k2 *= dt;
        f_aug!(k3,S - k1 + 2*k2); k3 *= dt;
        copyto!(dS, S + (k1 + 4*k2 + k3)/6)
    end
end

function interpolate_trajectory(integration, X, U, t_lqr)
    X = copy(X)
    U = copy(U)
    n = size(X,1)
    m = size(U,1)
    time = collect(TrajectoryOptimization.get_time(solver))
    if integration == :midpoint
        interp_X = interpolate_rows(time, X, :midpoint)
    else  # :rk3, rk4
        interp_X = interpolate_rows(time, X, :cubic)
    end

    if solver.control_integration == :zoh
        interp_U = interpolate_rows(time, U, :zoh)
    else  # :zoh
        interp_U = interpolate_rows(time, U, :linear)
    end
    Xnew = interp_X(t_lqr)
    Unew = interp_U(t_lqr)
    return Xnew, Unew
end


"""
Generates a function that interpolates the rows of a matrix
# Arguments
* interpolation: interpolation scheme. Options are [:zoh, :midpoint, :linear, :cubic]
"""
function interpolate_rows(t::Vector{T}, X::Matrix{T}, interpolation=:cubic) where T
    n,N = size(X)
    bc = Cubic(Line(OnGrid()))

    # Use zero order hold for :zoh and :midpoint
    interpolation in [:linear, :cubic] ? zoh = false : zoh = true

    # Create interpolator
    itr = interpolate(X, (NoInterp(), BSpline(bc) ))
    dt = t[2] - t[1]

    # Interpolation function
    function interp(t_)
        i = t_./dt .+ 1
        if zoh
            i = floor.(i)
        end
        itr(1:n,i)
    end
end
