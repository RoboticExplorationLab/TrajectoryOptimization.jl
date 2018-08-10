include("solver_options.jl")
import Base: copy, length, size

"""
$(SIGNATURES)
Determine if the dynamics in model are in place. i.e. the function call is of
the form `f!(xdot,x,u)`, where `xdot` is modified in place. Returns a boolean.
"""
function is_inplace_dynamics(model::Model)::Bool
    x = rand(model.n)
    u = rand(model.m)
    xdot = rand(model.n)
    try
        model.f(xdot,x,u)
    catch x
        if x isa MethodError
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)
Makes the dynamics function `f(x,u)` appear to operate as an inplace operation of the
form `f!(xdot,x,u)`.
"""
function wrap_inplace(f::Function)
    f!(xdot,x,u) = copy!(xdot, f(x,u))
end


"""
$(TYPEDEF)
Contains all information to solver a trajectory optimization problem.

The Solver type is immutable so is unique to the particular problem. However,
anything in `Solver.opts` can be changed dynamically.
"""
struct Solver
    model::Model         # Dynamics model
    obj::Objective       # Objective (cost function and constraints)
    opts::SolverOptions  # Solver options (iterations, method, convergence criteria, etc)
    dt::Float64          # Time step
    fd::Function         # Discrete in place dynamics function, `fd(ẋ,x,u)`
    F::Function          # Jacobian of discrete dynamics, `fx,fu = F(x,u)`
    c_fun::Function
    c_jacobian::Function
    N::Int               # Number of time steps

    function Solver(model::Model, obj::Objective; integration::Symbol=:rk4, dt=0.01, opts::SolverOptions=SolverOptions(), infeasible=false)
        N = Int(floor(obj.tf/dt));
        n,m = model.n, model.m

        # Make dynamics inplace
        if is_inplace_dynamics(model)
            f! = model.f
        else
            f! = wrap_inplace(model.f)
        end

        # Get integration scheme
        if isdefined(TrajectoryOptimization,integration)
            discretizer = eval(integration)
        else
            throw(ArgumentError("$integration is not a defined integration scheme"))
        end

        # Generate discrete dynamics equations
        fd! = discretizer(f!, dt)
        f_aug! = f_augmented!(f!, model.n, model.m)
        fd_aug! = discretizer(f_aug!)
        F!(J,Sdot,S) = ForwardDiff.jacobian!(J,fd_aug!,Sdot,S)

        fx = zeros(n,n)
        fu = zeros(n,m)

        nm1 = model.n + model.m + 1
        J = zeros(nm1, nm1)
        S = zeros(nm1)

        # Auto-diff discrete dynamics
        function Jacobians!(x,u)
            infeasible = length(u) != m
            S[1:n] = x
            S[n+1:end-1] = u[1:m]
            S[end] = dt
            Sdot = zeros(S)
            F_aug = F!(J,Sdot,S)
            fx .= F_aug[1:model.n,1:model.n]
            fu .= F_aug[1:model.n,model.n+1:model.n+model.m]
            if infeasible
                return fx, [fu zeros(n,n)]
            end
            return fx, fu

        end

        c_fun, c_jacob = generate_constraint_functions(obj)

        # Copy solver options so any changes don't modify the options passed in
        options = copy(opts)
        options.infeasible = infeasible

        new(model, obj, options, dt, fd!, Jacobians!, c_fun, c_jacob, N)

    end
end

generate_constraint_functions(obj::UnconstrainedObjective) = (x,u)->nothing, (x,u)->nothing

"""
$(SIGNATURES)
Return the quadratic control stage cost R

If using an infeasible start, will return the augmented cost matrix
"""
function getR(solver::Solver)::Array{Float64,2}
    if solver.opts.infeasible
        R = solver.opts.infeasible_regularization*eye(solver.model.m+solver.model.n)
        R[1:solver.model.m,1:solver.model.m] = solver.obj.R
        return R
    else
        return solver.obj.R
    end
end
