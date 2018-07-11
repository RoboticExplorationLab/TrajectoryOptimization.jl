struct Solver
    model::Model
    obj::Objective
    dt::Float64
    fd::Function  # discrete dynamics
    F::Function
    N::Int
    function Solver(model::Model, obj::Objective, dt::Float64)
        obj_n = size(obj.Q, 1)
        obj_m = size(obj.R, 1)
        @assert obj_n == model.n
        @assert obj_m == model.m

        # RK4 integration
        fd = rk4(model.f, dt)
        F(x,u) = Jacobian(fd,x,u)
        N = Int(floor(obj.tf/dt))
        new(model, obj, dt, fd, F, N)
    end

    # function Solver(model, obj, dt=0.1)
    #     n, m = model.n, model.m
    #     fd = f_midpoint(model.f, dt)     # Discrete dynamics
    #     f_aug = f_augmented(model)  # Augmented continuous dynamics
    #     fd_aug = f_midpoint(f_aug)  # Augmented discrete dynamics
    #
    #     out = zeros(n+m+1)
    #     Df(S::Array) = ForwardDiff.jacobian(fd_aug, S)
    #
    #     function f_jacobian(x::Array,u::Array,dt::Float64)
    #         Df_aug = Df([x;u;dt])
    #         A = Df_aug[1:n,1:n]
    #         B = Df_aug[1:n,n+1:n+m]
    #         return A,B
    #     end
    #
    #     N = Int(floor(obj.tf/dt));
    #     new(model, obj, dt, fd, f_jacobian, N)
    # end
end

# function f_midpoint(f::Function, dt::Float64)
#     dynamics_midpoint(x,u)  = x + f(x + f(x,u)*dt/2, u)*dt
# end
#
# function f_midpoint(f::Function)
#     dynamics_midpoint(S::Array)  = S + f(S + f(S)*S[end]/2)*S[end]
# end
#
# function f_midpoint!(f_aug!::Function)
#
#     function dynamics_midpoint(out::AbstractVector, S::Array)
#         # out = zeros(7)
#         f_aug!(out, S)
#         f_aug!(out, S + out*S[end]/2)
#         copy!(out, S + out*S[end])
#     end
# end
#
#
# function f_augmented(model::Model)
#     n, m = model.n, model.m
#     f_aug = f_augmented(model.f, n, m)
#     f(S::Array) = [f_aug(S); zeros(m+1)]
# end
#
# function f_augmented!(model::Model)
#     n, m = model.n, model.m
#     f_aug! = f_augmented!(model.f, n, m)
#     f!(out::AbstractVector, S::Array) = [f_aug!(out, S); zeros(m+1)]
# end
#
# function f_augmented(f::Function, n::Int, m::Int)
#     f_aug(S::Array) = f(S[1:n], S[n+(1:m)])
# end
#
# function f_augmented!(f::Function, n::Int, m::Int)
#     f_aug!(out::AbstractVector, S::Array) = copy!(out, f(S[1:n], S[n+(1:m)]))
# end

function rk4(f::Function,dt::Float64)
    # Runge-Kutta 4
    k1(x,u) = dt*f(x,u)
    k2(x,u) = dt*f(x + k1(x,u)/2.,u)
    k3(x,u) = dt*f(x + k2(x,u)/2.,u)
    k4(x,u) = dt*f(x + k3(x,u), u)
    fd(x,u) = x + (k1(x,u) + 2.*k2(x,u) + 2.*k3(x,u) + k4(x,u))/6.
end

function midpoint(f::Function,dt::Float64)
    fd(x,u) = x + f(x + f(x,u)*dt/2., u)*dt
end

function Jacobian(f::Function,x::Array{Float64,1},u::Array{Float64,1})
    f1 = a -> f(a,u)
    f2 = b -> f(x,b)
    fx = ForwardDiff.jacobian(f1,x)
    fu = ForwardDiff.jacobian(f2,u)
    return fx, fu
end
