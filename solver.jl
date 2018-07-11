struct Solver
    model::Model
    obj::Objective 
    dt::Float64     # time step
    fd::Function    # discrete dynamics
    F::Function     # jacobian
    N::Int          # number of time steps

    function Solver(model::Model, obj::Objective, discretizer::Function=rk4; dt::Float64=0.01)
        obj_n = size(obj.Q, 1)
        obj_m = size(obj.R, 1)
        @assert obj_n == model.n
        @assert obj_m == model.m
        n,m = model.n, model.m

        # Dynamics integration
        f_aug = f_augmented(model)   # Augmented continuous dynamics
        fd_aug = discretizer(f_aug)  # Augmented discrete dynamics
        fd = (x,u) -> fd_aug([x;u;dt])[1:n]    # Discrete dynamics

        Df(S::Array) = ForwardDiff.jacobian(fd_aug, S)

        function f_jacobian(x::Array, u::Array)
            Df_aug = Df([x;u;dt])
            A = Df_aug[1:n,1:n]
            B = Df_aug[1:n,n+1:n+m]
            return A,B
        end
        
        N = Int(floor(obj.tf/dt))
        new(model, obj, dt, fd, f_jacobian, N)
    end

end


# Augmenting dynamics functions
function f_augmented(model::Model)
    n, m = model.n, model.m
    f_aug(S::Array) = [model.f(S[1:n], S[n+(1:m)]); zeros(m+1)]
end


# Discrete Integration Functions
function rk4(f_aug::Function)
    # Runge-Kutta 4
    fd(S::Array) = begin
        k1(S) = S[end]*f_aug(S)
        k2(S) = S[end]*f_aug(S + k1(S)/2.)
        k3(S) = S[end]*f_aug(S + k2(S)/2.)
        k4(S) = S[end]*f_aug(S + k3(S))
        (k1(S) + 2.*k2(S) + 2.*k3(S) + k4(S))/6.
    end
end

<<<<<<< HEAD
function Jacobian(f::Function,x::Array{Float64,1},u::Array{Float64,1})
    f1 = a -> f(a,u)
    f2 = b -> f(x,b)
    fx = ForwardDiff.jacobian(f1,x)
    fu = ForwardDiff.jacobian(f2,u)
    return fx, fu
end
=======
function midpoint(f::Function)
    fd(S::Array)  = S + f(S + f(S)*S[end]/2)*S[end]
end
>>>>>>> df1b01e9b179450b0a37331c241231b1d93661a1
