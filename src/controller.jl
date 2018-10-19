"""
@(SIGNATURES)
    Generate an approximately continuous LQR controller from a discrete controller for use in simulation
        -requires X and U nominal trajectories and gains K, in matrix form; number of knot points and time step size from solve
        -option for first-order hold interpolation (cubic state, linear gains and control)
        -option for controller saturation with max/min limits
"""
function generate_controller(X::Matrix,U::Matrix,K::Array{Float64,3},N::Int64,dt::Float64,integration::Symbol=:zoh,u_min=-Inf,u_max=Inf)
    """
    X - (n,N)
    U - zoh: (m,N-1), foh: (m,N)
    K - zoh: (m,n,N-1), foh: (m,n,N)

    """
    m, n, _ = size(K)
    if integration == :zoh

        # Get matrices in correct format for Interpolations.jl
        K_zoh = [K[:,:,k] for k = 1:N-1]
        X_zoh = [X[:,k] for k = 1:N-1]
        U_zoh = [U[:,k] for k = 1:N-1]

        # For zoh and Interpolations.jl we repeate the final K,X,U
        push!(K_zoh,K[:,:,N-1])
        push!(X_zoh,X[:,N-1])
        push!(U_zoh,U[:,N-1])

        # zero-order hold interpolation on gains, and trajectories
        K_interp = interpolate(K_zoh,BSpline(Constant()))
        X_interp = interpolate(X_zoh,BSpline(Constant()))
        U_interp = interpolate(U_zoh,BSpline(Constant()))

        function controller_zoh(x,t)
            j = t/dt + 1
            return max.(min.(K_interp(floor(Int64,j))*(x - X_interp(floor(Int64,j))) + U_interp(floor(Int64,j)),u_max),u_min)
        end
        return controller_zoh

    elseif integration == :foh
        # Get matrices in correct format for Interpolations.jl
        K_foh = [K[:,:,k] for k = 1:N]
        # Interpolations.jl does not work with matrices and cubic interpolation so we loop
        X_foh = []
        for i = 1:n
            push!(X_foh,[X[i,k] for k = 1:N])
        end
        U_foh = [U[:,k] for k = 1:N]

        # Linear interpolation on gains and control trajectory, cubic interpolation for state trajectory
        K_interp = interpolate(K_foh,BSpline(Linear()))
        function X_interp(j)
            x = zeros(n)
            for i = 1:n
                x[i] = interpolate(X_foh[i],BSpline(Cubic(Line(OnGrid()))))(j)
            end
            return x
        end
        U_interp = interpolate(U_foh,BSpline(Linear()))

        function controller_foh(x,t)
            j = t/dt + 1
            return max.(min.(K_interp(j)*(x - X_interp(j)) + U_interp(j),u_max),u_min)
        end

        return controller_foh
    end
end

"""
@(SIGNATURES)
    Time Varying Discrete Linear Quadratic Regulator (TVLQR)
"""
function lqr(A::Array{Float64,3}, B::Array{Float64,3}, Q::AbstractArray{Float64,2}, R::AbstractArray{Float64,2}, Qf::AbstractArray{Float64,2})::Array{Float64,3}
    n,m,N = size(B)
    N += 1
    K = zeros(m,n,N-1)
    S = zeros(n,n)

    # Boundary conditions
    S .= Qf
    for k = 1:N-1
        # Compute control gains
        K[:,:,k] = (R + B[:,:,k]'*S*B[:,:,k])\(B[:,:,k]'*S*A[:,:,k])
        # Calculate cost-to-go for backward propagation
        S .= Q + A[:,:,k]'*S*A[:,:,k] - A[:,:,k]'*S*B[:,:,k]*K[:,:,k]
    end
    K
end

function lqr(results::SolverResults, solver::Solver)
    n, m, N = get_sizes(solver)
    Q, R, Qf, xf = get_cost_matrices(solver)
    A, B = results.fx, results.fu
    lqr(A,B,Q,R,Qf)
end

"""
@(SIGNATURES)
    Simulate LQR controller
"""
function simulate_controller(f::Function,integration::Symbol,controller::Function,n::Int64,m::Int64,dt::Float64,x0::AbstractVector,tf::Float64,u_min=-Inf,u_max=Inf)
    t_sim = 0:dt:tf
    N_sim = length(t_sim)
    if integration == :ode45
        function closed_loop_dynamics(t,x)
            xdot = zeros(n)
            f(xdot,x,controller(x,t))
            xdot
        end

        t_sim, X_sim = ode45(closed_loop_dynamics, x0, 0:dt:tf)
        X_sim = to_array(X_sim)
        U_sim = zeros(m,N_sim)
        for k = 1:N_sim-1
            U_sim[:,k] = controller(X_sim[:,k],t_sim[k])
        end
        return X_sim, U_sim
    else
        # Discrete dynamics
        discretizer = eval(integration)
        fd = discretizer(f, dt)

        # determine number of knot points for simulation
        N_sim = convert(Int64,floor(tf/dt))+1

        # Allocate memory for simulated state and control trajectories
        X_sim = zeros(n,N_sim)
        X_sim[:,1] = x0
        U_sim = zeros(m,N_sim)

        # simulate
        for (k,t) in enumerate(range(0,length=N_sim-1,stop=tf-dt))
            U_sim[:,k] = controller(X_sim[:,k],t)
            fd(view(X_sim,:,k+1), X_sim[:,k],U_sim[:,k])
        end
        return X_sim, U_sim
    end
end
