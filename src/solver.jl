include("solver_options.jl")
import Base: copy, length, size, reset

"""
$(TYPEDEF)
    Type for solver states
"""
mutable struct SolverState
    constrained::Bool # Constrained solve
    minimum_time::Bool # Minimum time solve
    infeasible::Bool # Infeasible solve

    unconstrained_original_problem::Bool # Original problem is unconstrained but Solve converts to constrained for Infeasible or Minimum Time problem
    second_order_dual_update::Bool # Second order update for dual variables (Lagrange multipliers)
    fixed_constraint_jacobians::Bool # If no custom constraints are provided, all constraint Jacobians are fixed and only need to be updated once
    fixed_terminal_constraint_jacobian::Bool
    penalty_only::Bool  # initial phase where only penalty term is updated each outer loop

    function SolverState()
        new(false,false,false,false,false,false,false,false)
    end
end

function reset(state::SolverState)
    state.constrained = false
    state.minimum_time = false
    state.infeasible = false
    state.unconstrained_original_problem = false
    state.second_order_dual_update = false
    state.fixed_constraint_jacobians = false
    state.fixed_terminal_constraint_jacobian = false
    state.penalty_only = false
    return nothing
end

"""
$(TYPEDEF)
Responsible for storing all solve-dependent variables and solve parameters.
"""
struct Solver{M<:Model,O<:Objective}
    model::M             # Dynamics model
    obj::O               # Objective (cost function and constraints)
    opts::SolverOptions  # Solver options (iterations, method, convergence criteria, etc)
    state::SolverState   # Solver state
    dt::Float64          # Time step
    fd::Function         # Discrete in place dynamics function, `fd(_,x,u)`
    Fd::Function         # Jacobian of discrete dynamics, `fx,fu = F(x,u)`
    Fc::Function         # Jacobian of continuous dynamics, `fx, fu = F(x,u)`
    c_fun::Function
    c_jacobian::Function
    c_labels::Vector{String}  # Constraint labels
    N::Int64             # Number of time steps
    evals::Vector{Int}   # Evaluation counts
    integration::Symbol

    """
    $(SIGNATURES)
    Create a Solver from a model and objective. The user should specify either N (number of knot points) or dt (time step), but typically not both.

    Integration specifies the integration method for discretizing the continuous dynamics and must be (:midpoint, :rk3, or :rk4).

    Solver parameters can be passed in using opts.

    """
    function Solver(model::M, obj::O; integration::Symbol=:rk4, dt::Float64=NaN, N::Int=-1, opts::SolverOptions=SolverOptions()) where {M<:Model,O<:Objective}
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

        # Evaluation counts
        evals = zeros(Int,4)  # [Fd,Fc,c,C]
        reset(model)

        # The dynamics function increments an eval counter each time it's called
        f!(xdot,x,u) = dynamics(model,xdot,x,u)

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

        # Get continuous dynamics jacobian
        Jc = zeros(n+m,n+m)
        Sc = zeros(n+m)
        Scdot = zero(Sc)
        Fc!(Jc,dS,S) = ForwardDiff.jacobian!(Jc,f_aug!,dS,S)
        function fc_jacobians!(x,u)
            # infeasible = size(u,1) != m̄
            Sc[1:n] = x
            Sc[n+1:n+m] = u[1:m]
            Fc!(Jc,Scdot,Sc)
            return Jc[1:n,1:n], Jc[1:n,n+1:n+m] # fx, fu
        end


        """
        s = [x;u;h]
        x ∈ R^n
        u ∈ R^m
        h ∈ R, h = sqrt(dt_k)
        """
        fd_aug! = discretizer(f_aug!)
        nm1 = n + m + 1

        In = 1.0*Matrix(I,n,n)

        # Initialize discrete and continuous dynamics Jacobians
        Jd = zeros(nm1, nm1)
        Sd = zeros(nm1)
        Sdotd = zero(Sd)
        Fd!(Jd,Sdotd,Sd) = ForwardDiff.jacobian!(Jd,fd_aug!,Sdotd,Sd)

        # Discrete dynamics Jacobians
        function fd_jacobians!(fdx,fdu,x,u)
            # Check for infeasible solve
            infeasible = length(u) != m̄

            # Assign state, control (and dt) to augmented vector
            Sd[1:n] = x
            Sd[n+1:n+m] = u[1:m]
            state.minimum_time ? Sd[n+m+1] = u[m̄] : Sd[n+m+1] = √dt

            # Calculate Jacobian
            Fd!(Jd,Sdotd,Sd)

            # if infeasible
            #     return Jd[1:n,1:n], [Jd[1:n,n.+(1:m̄)] I] # fx, [fū I]
            # else
            #     return Jd[1:n,1:n], Jd[1:n,n.+(1:m̄)] # fx, fū
            # end

            fdx[1:n,1:n] = Jd[1:n,1:n]
            fdu[1:n,1:m̄] = Jd[1:n,n.+(1:m̄)]

            if infeasible
                fdu[1:n,m̄+1:m̄+n] = In
            end
            if state.minimum_time
                fdu[n̄,m̄] = 1.
            end

        end
        # Discrete dynamics Jacobian for stacked variables
        function fd_jacobians!(dz,z)
            Sd[1:n+m] = z
            Fd!(Jd,Sdotd,Sd)
            copyto!(dz,Jd[1:n,1:n+m])
        end

        # Generate constraint functions
        c!, c_jacobian!, c_labels = generate_constraint_functions(obj, max_dt = opts.max_dt, min_dt = opts.min_dt)

        # Copy solver options so any changes don't modify the options passed in
        options = copy(opts)

        new{M,O}(model, obj, options, state, dt, fd!, fd_jacobians!, fc_jacobians!, c!, c_jacobian!, c_labels, N, evals, integration)
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


"""$(SIGNATURES) Reset the number of function evaluations and solver state"""
reset(solver::Solver) = begin reset(solver.state); reset_evals(solver) end
reset_evals(solver::Solver) = begin solver.evals .= zeros(4); reset(solver.model) end

"""$(SIGNATURES) Get the number of function evaluations for a given function """
function evals(solver::Solver, fun::Symbol)
    if fun in [:Fd, :discrete_dynamics_jacobian]
        return solver.evals[1]
    elseif fun in [:Fc, :continuous_dynamics_jacobian, :dynamics_jacobian]
        return solver.evals[2]
    elseif fun in [:c, :C, :constraints, :constraint_function, :c_fun]
        return solver.evals[3]
    elseif fun in [:∇c, :constraint_jacobian, :c_jacob, :c_jacobian]
        return solver.evals[4]
    elseif fun in [:f, :dynamics, :continuous_dynamics]
        return evals(solver.model)
    end
end

# Get functions from Solver (incrementing evals)
""" $(SIGNATURES) Return the constraint function, and optionally count the evaluations """
function constraint_function(solver::Solver, count::Bool=true)
    function c_fun_count!(c,x,u)
        solver.c_fun(c,x,u)
        count ? solver.evals[3] += 1 : nothing
    end
    function c_fun_count!(c,x)
        solver.c_fun(c,x)
        count ? solver.evals[3] += 1 : nothing
    end
    return c_fun_count!
end

""" $(SIGNATURES) Return the discrete dynamics jacobian function, and optionally count the evaluations """
function discrete_dynamics_jacobian(solver::Solver, count::Bool=true)
    function Fd!(A,B,x,u)
        solver.Fd(A,B,x,u)
        count ? solver.evals[1] += 1 : nothing
    end
    return Fd!
end

""" $(SIGNATURES) Return the constraint jacobian function, and optionally count the evaluations """
function constraint_jacobian(solver::Solver, count::Bool=true)
    function c_jacob!(Cx,Cu,x,u)
        solver.c_jacobian(Cx,Cu,x,u)
        count ? solver.evals[4] += 1 : nothing
    end
    function c_jacob!(Cx,x)
        solver.c_jacobian(Cx,x)
        count ? solver.evals[4] += 1 : nothing
    end
    return c_jacob!
end

""" $(SIGNATURES) Return the continuous dynamics jacobian function, and optionally count the evaluations """
function dynamics_jacobian(solver::Solver, count::Bool=true)
    function Fc!(x,u)
        count ? solver.evals[2] += 1 : nothing
        solver.Fc(x,u)
    end
    return Fc!
end
