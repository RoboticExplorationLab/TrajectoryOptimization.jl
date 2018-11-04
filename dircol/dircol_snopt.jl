using Base.Test
using Snopt
using ForwardDiff
include("dircol.jl")


function gen_usrfun(model, obj, dt, pack, method; grads=:none)
    n,m,N = pack

    # Weights (Trapezoidal)
    weights = get_weights(method,N)

    # Jacobian for Dynamics
    f_aug! = f_augmented!(model!.f, model.n, model.m)
    zdot = zeros(n)
    F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)

    function eval_f(Z)
        X,U = unpackZ(Z,pack)
        cost(X,U,weights,obj)
    end

    function eval_g(Z)
        X,U = unpackZ(Z,pack)
        calc_constraints(X,U,method,dt,model.f)
    end

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

function dircol(model,obj,dt,grads)

    method = :trapezoid
    method = :hermite_simpson_separated

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

    usrfun, eval_f, eval_g = gen_usrfun(model, obj, dt, pack, method, grads=grads)


    # Set up the problem
    Z0 = get_initial_state(obj,N)
    lb,ub = get_bounds(obj,N)


    # Set options
    options = Dict{String, Any}()
    options["Derivative option"] = 0
    options["Verify level"] = 1


    # Solve the problem
    @time z_opt, fopt, info = snopt(usrfun, Z0, lb, ub, options)
    @time snopt(usrfun, Z0, lb, ub, options)
    # xopt, fopt, info = Z0, Inf, "Nothing"
    x_opt,u_opt = unpackZ(z_opt,pack)

    println(info)
    return x_opt, u_opt, fopt
end

model = TrajectoryOptimization.Dynamics.pendulum[1]
model! = TrajectoryOptimization.Dynamics.pendulum![1]
obj = TrajectoryOptimization.Dynamics.pendulum[2]
obj = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-2, u_max=2)
dt = 0.1

using Juno
# @profiler dircol(model, obj, dt, :other)
x_opt, u_opt, fopt = dircol(model, obj, dt, :none)
plot(x_opt')

weights = get_weights(:hermite_simpson_separated,N)
N = size(x_opt,2)
pack = (model.n,model.m,N)
X,U = x_opt,u_opt
Z = packZ(X,U)
f_aug! = f_augmented!(model!.f, model.n, model.m)
zdot = zeros(model.n)
F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)
usrfun, eval_f, eval_g = gen_usrfun(model,obj,dt, pack, :hermite_simpson_separated, grads=:auto)

# Constraints
jacob_g = constraint_jacobian(X,U,dt,:hermite_simpson_separated,F)
jacob_g_auto = ForwardDiff.jacobian(eval_g,Z)
errors = jacob_g_auto == jacob_g

# Objective
grad_f = cost_gradient(X,U,weights,obj)
grad_f_auto = ForwardDiff.gradient(eval_f,Z)
errors = grad_f - grad_f_auto
errors[end-10:end]
grad_f ≈ grad_f_auto













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
