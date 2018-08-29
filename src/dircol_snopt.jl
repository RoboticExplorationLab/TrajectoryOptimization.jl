using Base.Test
using Snopt
using ForwardDiff
include("dircol.jl")

"""
$(SIGNATURES)
Generate the custom function to be passed into SNOPT, as well as `eval_f` and
`eval_g` used to calculate the objective and constraint functions.

# Arguments
* model: TrajectoryOptimization.Model for the dynamics
* obj: TrajectoryOptimization.ConstrainedObjective to describe the cost function
* dt: time step
* pack: tuple of important sizes (n,m,N) -> (num states, num controls, num knot points)
* method: Collocation method. Either :trapezoid or :hermite_simpson_separated
* grads: Specifies the gradient information provided to SNOPT.
    :none - returns no gradient information to SNOPT
    :auto - uses ForwardDiff to calculate gradients
    :quadratic - uses functions exploiting quadratic cost functions
"""
function gen_usrfun(model::Model, obj::ConstrainedObjective, dt::Float64, pack::Tuple, method::Symbol; grads=:none)::Tuple
    n,m,N = pack

    # Weights (Trapezoidal)
    weights = get_weights(method,N)*dt

    # Jacobian for Dynamics
    f_aug! = f_augmented!(model.f, model.n, model.m)
    zdot = zeros(n)
    F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)

    # Count constraints
    pI = 0
    pE = (N-1)*n # Collocation constraints
    p_colloc = pE

    pI_obj, pE_obj = count_constraints(obj)
    pI_c,   pE_c   = pI_obj[2], pE_obj[2]
    pI_N_c, pE_N_c = pI_obj[4], pE_obj[4]

    pI += pI_c*(N-1) + pI_N_c  # Add custom inequality constraints
    pE += pE_c*(N-1) + pE_N_c  # Add custom equality constraints

    # Evaluate the Cost (objective) function)
    function eval_f(Z)
        X,U = unpackZ(Z,pack)
        cost(X,U,weights,obj)
    end

    # Evaluate the equality constraint function
    function eval_ceq(Z)
        X,U = unpackZ(Z,pack)
        gE = zeros(eltype(Z),pE)
        g_colloc = collocation_constraints(X,U,method,dt,model.f)
        gE[1:p_colloc] = g_colloc

        # Custom constraints
        if pE_c > 0
            gE_c = zeros(eltype(Z),pE_c,N-1)
            for k = 1:N-1
                gE_c[:,k] = obj.cE(X[:,k],U[:,k])
            end
            gE[p_colloc+1 : end-pE_N_c] = vec(gE_c)
        end
        if pE_N_c > 0
            gE[end-pE_N_c+1:end] = obj.cE(X[:,N])
        end
        return g_colloc
    end

    # Evaluate the inequality constraint function
    function eval_c(Z)
        X,U = unpackZ(Z,pack)
        gI = zeros(eltype(Z),pI)

        # Custom constraints
        if pI_c > 0
            gI_c = zeros(eltype(Z),pI_c,N-1)
            for k = 1:N-1
                gI[k] = obj.cI(X[:,k],U[:,k])
            end
            gI[p_colloc+1 : end-pI_N_c] = vec(gI_c)
        end
        if pI_N_c > 0
            gI[end-pI_N_c+1:end] = obj.cI(X[:,N])
        end
        return gI
    end

    """
    Stack constraints as follows:
    [ general stage inequality,
      general terminal inequality,
      general stage equality,
      general terminal equality,
      collocation constraints ]
    """
    function eval_g(Z)
        g = zeros(eltype(Z),pI+pE)
        g[1:pI] = eval_c(Z)
        g[pI+1:end] = eval_ceq(Z)
        return g
    end

    # User defined function passed to SNOPT (via Snopt.jl)
    function usrfun(Z)
        # Objective Function (Cost)
        J = eval_f(Z)

        # Constraints (dynamics only for now)
        c = Float64[] # No inequality constraints for now
        ceq = eval_ceq(Z)

        fail = false

        if grads == :none
            return J, c, ceq, fail
        else
            X,U = unpackZ(Z,pack)

            # Gradient of Objective
            if grads == :auto
                grad_f = ForwardDiff.gradient(eval_f,Z)
            else
                grad_f = cost_gradient(X,U,weights,obj)
            end

            # Constraint Jacobian
            jacob_c = Float64[]
            if grads == :auto
                jacob_ceq = ForwardDiff.jacobian(eval_g,Z)
            else
                jacob_ceq = constraint_jacobian(X,U,dt,method,F)
            end

            return J, c, ceq, grad_f, jacob_c, jacob_ceq, fail
        end
    end

    return usrfun, eval_f, eval_c, eval_ceq
end

"""
$(SIGNATURES)
Solve a trajectory optimization problem with direct collocation

# Arguments
* model: TrajectoryOptimization.Model for the dynamics
* obj: TrajectoryOptimization.ConstrainedObjective to describe the cost function
* dt: time step. Used to determine the number of knot points. May be modified
    by the solver in order to achieve an integer number of knot points.
* method: Collocation method.
    :trapezoidal - First order interpolation on states and zero order on control.
    :hermite_simpson_separated - Hermite Simpson collocation with the midpoints
        included as additional decision variables with constraints
* grads: Specifies the gradient information provided to SNOPT.
    :none - returns no gradient information to SNOPT
    :auto - uses ForwardDiff to calculate gradients
    :quadratic - uses functions exploiting quadratic cost functions
"""
function dircol(model::Model,obj::ConstrainedObjective,dt::Float64;
        method::Symbol=:hermite_simpson_separated, grads::Symbol=:quadratic)

    # Constants
    nSeg = Int(floor(obj.tf/dt)); # Number of segments
    dt = obj.tf/nSeg
    if method == :trapezoid
        N = nSeg + 1
    elseif method == :hermite_simpson_separated
        N = 2*nSeg + 1
    end
    n,m = model.n, model.m
    pack = (n,m,N)

    # Generate the objective/constraint function and its gradients
    usrfun, eval_f, eval_g = gen_usrfun(model, obj, dt, pack, method, grads=grads)

    # Set up the problem
    Z0 = get_initial_state(obj,N)
    lb,ub = get_bounds(obj,N)

    # Set options
    options = Dict{String, Any}()
    options["Derivative option"] = 0
    options["Verify level"] = 1

    # Solve the problem
    println("Passing Problem to SNOPT...")
    @time z_opt, fopt, info = snopt(usrfun, Z0, lb, ub, options)
    # @time snopt(usrfun, Z0, lb, ub, options)
    # xopt, fopt, info = Z0, Inf, "Nothing"
    x_opt,u_opt = unpackZ(z_opt,pack)

    println(info)
    return x_opt, u_opt, fopt
end















#
# Z0 = get_initial_state(obj,N)
# options = Dict{String, Any}()
# lb,ub = get_bounds(obj,N)
# options["Derivative option"] = 0
# options["Verify level"] = 1
#
# @time xopt, fopt, info = snopt(gradfree, Z0, lb, ub, options)
# x_soln,u_soln = unpackZ(xopt)
# plot(x_soln')
# plot(u_soln')
# eval_g(xopt)
#
# # Tests
# X,U = unpackZ(Z0)
# X = rand(size(X))
# U = rand(size(U))
# Z = packZ(X,U)
#
# X2,U2 = unpackZ(Z)
# @test X2 == X
# @test U2 == U
#
# # Test cost
# J1 = eval_f(Z)
# J2 = cost(X,U)
# @test J1 == J2
#
# # Test Constraints
# g = eval_g(Z)
#
# # Test Gradient of Objective
# grad_f = zeros((n+m)*N)
# eval_grad_f(Z,grad_f)
# grad_f
# grad_f_auto = ForwardDiff.gradient(eval_f,Z)
# @test grad_f == grad_f_auto
#
# # Gradient of Constraints
# jacob_g = eval_jac_g(Z)
# jacob_g_auto = ForwardDiff.jacobian(eval_g,Z)
# @test jacob_g_auto == jacob_g
#
# # Test usrfun
# f_val,g2,df,dg,fail = usrfun(Z)
# @test f_val == J1
# @test g2 == g
# @test df == grad_f
# @test dg == jacob_g
# @test ~fail
