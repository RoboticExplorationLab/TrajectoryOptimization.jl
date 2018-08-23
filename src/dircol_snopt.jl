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
    weights = get_weights(method,N)

    # Jacobian for Dynamics
    f_aug! = f_augmented!(model.f, model.n, model.m)
    zdot = zeros(n)
    F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)

    # Evaluate the Cost (objective) function)
    function eval_f(Z)
        X,U = unpackZ(Z,pack)
        cost(X,U,weights,obj)
    end

    # Evaluate the constraint function
    function eval_g(Z)
        X,U = unpackZ(Z,pack)
        calc_constraints(X,U,method,dt,model.f)
    end

    # User defined function passed to SNOPT (via Snopt.jl)
    function usrfun(Z)
        # Objective Function (Cost)
        J = eval_f(Z)

        # Constraints (dynamics only for now)
        c = Float64[] # No inequality constraints for now
        ceq = eval_g(Z)

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

    return usrfun, eval_f, eval_g
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
    print_info("Passing Problem to SNOPT...")
    @time z_opt, fopt, info = snopt(usrfun, Z0, lb, ub, options)
    @time snopt(usrfun, Z0, lb, ub, options)
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
