include("solver_options.jl")
import Base: copy, length, size

mutable struct SolverState
    constrained::Bool # Constrained solve
    minimum_time::Bool # Minimum time solve
    infeasible::Bool # Infeasible solve

    second_order_dual_update::Bool # Second order update for dual variables (Lagrange multipliers)
    fixed_constraint_jacobians::Bool # If no custom constraints are provided, all constraint Jacobians are fixed and only need to be updated once

    function SolverState()
        new(false,false,false,false,false)
    end
end

include("solver_options.jl")
import Base: copy, length, size

struct Solver{O<:Objective}
    model::Model         # Dynamics model
    obj::O               # Objective (cost function and constraints)
    opts::SolverOptions  # Solver options (iterations, method, convergence criteria, etc)
    state::SolverState   # Solver state
    dt::Float64          # Time step
    fd::Function         # Discrete in place dynamics function, `fd(_,x,u)`
    Fd::Function         # Jacobian of discrete dynamics, `fx,fu = F(x,u)`
    c_fun::Function
    c_jacobian::Function
    c_labels::Vector{String}  # Constraint labels
    N::Int64             # Number of time steps
    integration::Symbol

    function Solver(model::Model, obj::O; integration::Symbol=:rk4, dt::Float64=NaN, N::Int=-1, opts::SolverOptions=SolverOptions()) where {O}
        state = SolverState()

        # Check for minimum time
        if obj.tf == 0
            state.minimum_time = true
            dt = 0.
            if N==-1
                throw(ArgumentError("N must be specified for a minimum-time problem"))
            end
        else
            state.minimum_time = false

            # Handle combination of N and dt
            if isnan(dt) && N>0
                dt = obj.tf / (N-1)
            elseif ~isnan(dt) && N==-1
                N, dt = calc_N(obj.tf, dt)
            elseif isnan(dt) && N==-1
                @warn "Neither dt or N were specified. Setting N = 50"
                N = 50
                dt = obj.tf/N
            elseif ~isnan(dt) && N>0
                if dt !== obj.tf/(N-1)
                    throw(ArgumentError("Specified time step, number of knot points, and final time do not agree ($dt ≢ $(obj.tf)/$(N-1))"))
                end
            end
            if dt == 0
                throw(ArgumentError("dt must be non-zero for non-minimum time problems"))
            end
        end

        # Check N, dt for valid entries
        if N < 0
            err = ArgumentError("$N is not a valid entry for N. Number of knot points must be a positive integer.")
            throw(err)
        elseif dt < 0
            err = ArgumentError("$dt is not a valid entry for dt. Time step must be positive.")
            throw(err)
        end

        if O <: ConstrainedObjective
            state.constrained = true
        end

        f! = model.f

        n, m = model.n, model.m

        m̄ = m
        n̄ = n
        if state.minimum_time
            m̄ += 1
            n̄ += 1
            state.constrained = true
        end

        # Get integration scheme
        if isdefined(TrajectoryOptimization,integration)
            discretizer = eval(integration)
        else
            throw(ArgumentError("$integration is not a defined integration scheme"))
        end

        # Generate discrete dynamics equations
        fd! = discretizer(f!, dt)
        f_aug! = f_augmented!(f!, n, m)


        """
        s = [x;u;h]
        x ∈ R^n
        u ∈ R^m
        h ∈ R, h = sqrt(dt_k)
        """
        fd_aug! = discretizer(f_aug!)
        nm1 = n + m + 1

        # Initialize discrete and continuous dynamics Jacobians
        Jd = zeros(nm1, nm1)
        Sd = zeros(nm1)
        Sdotd = zero(Sd)
        Fd!(Jd,Sdotd,Sd) = ForwardDiff.jacobian!(Jd,fd_aug!,Sdotd,Sd)

        # Discrete dynamics Jacobians
        function fd_jacobians_zoh!(x,u)
            # Check for infeasible solve
            infeasible = length(u) != m̄

            # Assign state, control (and dt) to augmented vector
            Sd[1:n] = x
            Sd[n+1:n+m] = u[1:m]
            state.minimum_time ? Sd[n+m+1] = u[m̄] : Sd[n+m+1] = √dt

            # Calculate Jacobian
            Fd!(Jd,Sdotd,Sd)

            if infeasible
                return Jd[1:n,1:n], [Jd[1:n,n.+(1:m̄)] I] # fx, [fū I]
            else
                return Jd[1:n,1:n], Jd[1:n,n.+(1:m̄)] # fx, fū
            end
        end
        fd_jacobians! = fd_jacobians_zoh!

        # Generate constraint functions
        c!, c_jacobian!, c_labels = generate_constraint_functions(obj, max_dt = opts.max_dt, min_dt = opts.min_dt)

        # Copy solver options so any changes don't modify the options passed in
        options = copy(opts)

        new{O}(model, obj, options, state, dt, fd!, fd_jacobians!, c!, c_jacobian!, c_labels, N, integration)
    end
end

function Solver(solver::Solver; model=solver.model, obj=solver.obj,integration=solver.integration, dt=solver.dt, N=solver.N, opts=solver.opts)
     Solver(model, obj, integration=integration, dt=dt, N=N, opts=opts)
 end

""" $(SIGNATURES) Descriptive labels of the constraints
Any label with a "* " prefix is added by the solver and not in the original
constraints specified by the user
"""
function get_constraint_labels(solver::Solver)
     n,m,N = get_sizes(solver)
     c_labels = copy(solver.c_labels)
     if solver.state.infeasible
         lbl_inf = ["* infeasible control" for i = 1:n]
         append!(c_labels, lbl_inf)
     end
     if solver.state.minimum_time
         push!(c_labels, "* √dt (equality)")
     end
     return c_labels
 end


""" $(SIGNATURES)
Return boolean indices to the "original" constraints,
i.e. the constraints specified by the user (and not added by the solver).
"""
 function original_constraint_inds(solver::Solver)
     labels = get_constraint_labels(solver)
     map(x -> x[1:2] != "* ", labels)
 end

function calc_N(tf::Float64, dt::Float64)::Tuple
    N = convert(Int64,floor(tf/dt)) + 1
    dt = tf/(N-1)
    return N, dt
end
