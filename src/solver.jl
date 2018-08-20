include("solver_options.jl")
import Base: copy, length, size

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Solver type and related methods
#
#     TYPES
#         Solver
#
#     METHODS
#         is_inplace_dynamics: Checks if dynamics in Model are in-place
#         wrap_inplace: Makes non-inplace dynamics look in-place
#         getR: Return the quadratic control state cost (augmented if necessary)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    fd::Function         # Discrete in place dynamics function, `fd(x?,x,u)`
    Fd::Function         # Jacobian of discrete dynamics, `fx,fu = F(x,u)`
    fc::Function         # Continuous dynamics function (inplace)
    Fc::Function         # Jacobian of continuous dynamics
    c_fun::Function
    c_jacobian::Function
    N::Int               # Number of time steps
    integration::Symbol

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
        if integration == :rk3_foh
            fd_aug! = f_augmented_foh!(fd!,model.n,model.m)
        end

        Fd!(J,Sdot,S) = ForwardDiff.jacobian!(J,fd_aug!,Sdot,S)
        Fc!(J,Sdot,S) = ForwardDiff.jacobian!(J,f_aug!,Sdot,S)

        fx = zeros(n,n)
        fu = zeros(n,m)
        if integration == :rk3_foh
            fv = zeros(n,m)
            nm1 = model.n + model.m + model.m + 1
        else
            nm1 = model.n + model.m + 1
        end

        Jd = zeros(nm1, nm1)
        Sd = zeros(nm1)

        # Auto-diff discrete dynamics
        function Jacobians_Discrete!(x,u,v=zeros(size(u)))
            infeasible = length(u) != m
            Sd[1:n] = x
            Sd[n+1:n+m] = u[1:m]

            if integration == :rk3_foh
                Sd[n+m+1:n+m+m] = v[1:m]
            end

            Sd[end] = dt
            Sdot = zeros(Sd)
            Fd_aug = Fd!(Jd,Sdot,Sd)
            fx .= Fd_aug[1:model.n,1:model.n]
            fu .= Fd_aug[1:model.n,model.n+1:model.n+model.m]

            if infeasible
                return fx, [fu zeros(n,n)] # TODO add foh functionality
            end

            if integration == :rk3_foh
                fv .= Fd_aug[1:model.n,model.n+model.m+1:model.n+model.m+model.m]
                return fx, fu, fv
            end

            return fx, fu

        end

        # autodifferentiate continuous dynamics #TODO combined these two functions into 1
        Jc = zeros(model.n+model.m+1,model.n+model.m+1)
        Sc = zeros(model.n+model.m+1)

        function Jacobians_Continuous!(x,u)
            infeasible = length(u) != m
            Sc[1:n] = x
            Sc[n+1:n+m] = u[1:m]
            Sc[end] = dt
            Sdot = zeros(Sc)
            F_aug = Fc!(Jc,Sdot,Sc)
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

        new(model, obj, options, dt, fd!, Jacobians_Discrete!, model.f, Jacobians_Continuous!, c_fun, c_jacob, N, integration)

    end
end


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
