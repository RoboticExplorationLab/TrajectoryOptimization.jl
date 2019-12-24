using StaticArrays, LinearAlgebra, BenchmarkTools
import TrajectoryOptimization: dynamics, jacobian, discrete_dynamics, discrete_jacobian

# Get Quadrotor Model
model = Dynamics.Cartpole()
n,m = size(model)
N = 101  # knot points

# Build trajectory
X = [@SVector rand(n) for k = 1:N]
U = [@SVector rand(m) for k = 1:N-1]
dt = fill(0.1, N)
Z = Traj(X,U,dt)  # vector of KnotPoints

# Allocate storage arrays
fVal = [@SVector zeros(n) for k = 1:N-1]
∇f = [@SMatrix zeros(n,n+m) for k = 1:N-1]
∇fd = [@SMatrix zeros(n,n+m+1) for k = 1:N-1]  # discrte Jacobian also takes Jacobian wrt time


# Continuous Dynamics
function dynamics(fVal, model, Z)
    @inbounds for k in eachindex(fVal)
        # Most straightforward, passes through 2 "conversion functions":
        #  dynamics(model, z::KnotPoint) = dynamics(model, state(z), control(z), z.t)
        #  dynamics(model, x, u, t) = dynamics(model, x, u)
        #       The first extracts the state, control, and time from KnotPoint
        #       The second will drop dependence on time, by default
        fVal[k] = dynamics(model, Z[k])

        # ALTERNATE: call the dynamics function directly, by-passing the above pipeline
        #   dynamics(model, state(Z[k]), control(Z[k]))
        #
        # This function is defined in the Dynamics module
    end
end

# Continuous Jacobian
function jacobian(∇f, model, Z)
    @inbounds for k in eachindex(∇f)
        # This function is defined in static_model.jl::93
        #   It's best to pass in the KnotPoint directly, since having the state and control
        #   stacked together avoids some allocations when using ForwardDiff
        ∇f[k] = jacobian(model, Z[k])
    end
end

# Discrete Dynamics
function discrete_dynamics(fVal, model, Z)
    @inbounds for k in eachindex(fVal)
        # Most straightforward, passes through 1 "conversion function":
        #   discrete_dynamics(::Type{Q}, model, z::KnotPoint) = discrete_dynamics(Q, model, state(z), control(z), z.t, z.dt)
        fVal[k] = discrete_dynamics(RK3, model, Z[k])

        # ALTERNATE: bypass the extra function by making the call directly to the RK3 function
        #   defined in integration.jl::5
        #   discrete_dynamics(RK3, model, state(z), control(z), z.t, z.dt)
        end
    end
end

# Discrete Dynamics jacobian
function discrete_jacobian(∇fd, model, Z)
    @inbounds for k in eachindex(∇fd)
        # This function is defined in static_model.jl::138
        ∇fd[k] = discrete_jacobian(RK3, model, Z[k])
    end
end

# Test functions
dynamics(fVal, model, Z)
jacobian(∇f, model, Z)
discrete_dynamics(fVal, model, Z)
discrete_jacobian(∇fd, model, Z)

# Benchmark functions (should all have 0 allocations)
@btime dynamics($fVal, $model, $Z)
@btime jacobian($∇f, $model, $Z)
@btime discrete_dynamics($fVal, $model, $Z)
@btime discrete_jacobian($∇fd, $model, $Z)
