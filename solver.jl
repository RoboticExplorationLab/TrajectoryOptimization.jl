include("solver_options.jl")

struct Solver
    model::Model
    obj::Objective
    opts::SolverOptions
    dt::Float64
    fd::Function  # discrete dynamics
    F::Function
    N::Int

    function Solver(model::Model, obj::Objective, discretizer::Function=rk4; dt=0.01, opts::SolverOptions=SolverOptions())
        N = Int(floor(obj.tf/dt));

        if opts.inplace_dynamics
            fd! = rk4!(model.f, dt)
            f_aug! = f_augmented!(model.f, model.n, model.m)
            fd_aug! = rk4!(f_aug!)
            F!(J,Sdot,S) = ForwardDiff.jacobian!(J,fd_aug!,Sdot,S)

            function Jacobians!(x,u)
                nm1 = model.n + model.m + 1
                J = zeros(nm1, nm1)
                S = zeros(nm1)
                S[1:model.n] = x
                S[model.n+1:end-1] = u
                S[end] = dt
                Sdot = zeros(S)
                F_aug = F!(J,Sdot,S)
                fx = F_aug[1:model.n,1:model.n]
                fu = F_aug[1:model.n,model.n+1:model.n+model.m]
                return fx, fu
            end
            new(model, obj, opts, dt, fd!, Jacobians!, N)
        else
            fd = discretizer(model.f, dt)     # Discrete dynamics
            f_aug = f_augmented(model)  # Augmented continuous dynamics
            fd_aug = discretizer(f_aug)  # Augmented discrete dynamics
            F(S) = ForwardDiff.jacobian(fd_aug, S)

            function Jacobians(x::Array,u::Array)
                F_aug = F([x;u;dt])
                fx = F_aug[1:model.n,1:model.n]
                fu = F_aug[1:model.n,model.n+1:model.n+model.m]
                return fx, fu
            end
            new(model, obj, opts, dt, fd, Jacobians, N)
        end


    end
end

abstract type SolverResults end

struct UnconstrainedResults <: SolverResults
    X::Array{Float64,2}
    U::Array{Float64,2}
    K::Array{Float64,3}
    d::Array{Float64,2}
    X_::Array{Float64,2}
    U_::Array{Float64,2}
end

struct ConstrainedResults <: SolverResults
    X::Array{Float64,2}
    U::Array{Float64,2}
    K::Array{Float64,3}
    d::Array{Float64,2}

    C::Array{Float64,2}
    Iμ::Array{Float64,3}
    LAMBDA::Array{Float64,2}
    MU::Array{Float64,2}

    function ConstrainedResults(X,U,K,d,C,Iμ,LAMBDA,MU)
        new(X,U,K,d,C,Iμ,LAMBDA,MU)
    end
end
function ConstrainedResults(n,m,p,N)
    X = zeros(n,N)
    U = zeros(m,N-1)
    K = zeros(m,n,N-1)
    d = zeros(m,N-1)
    C = zeros(p,N)
    Iμ = zeros(p,p,N)
    LAMBDA = zeros(p,N)
    MU = zeros(p,N)
end

# struct SolverResultsConstrained <: SolverResults
#     C::Array{Float64}
# end

# Midpoint Integrator
function midpoint(f::Function, dt::Float64)
    dynamics_midpoint(x,u)  = x + f(x + f(x,u)*dt/2, u)*dt
end

function midpoint(f::Function)
    dynamics_midpoint(S::Array)  = S + f(S + f(S)*S[end]/2)*S[end]
end

# RK4 Integrator
function rk4(f::Function,dt::Float64)
    # Runge-Kutta 4
    k1(x,u) = dt*f(x,u)
    k2(x,u) = dt*f(x + k1(x,u)/2.,u)
    k3(x,u) = dt*f(x + k2(x,u)/2.,u)
    k4(x,u) = dt*f(x + k3(x,u), u)
    fd(x,u) = x + (k1(x,u) + 2.*k2(x,u) + 2.*k3(x,u) + k4(x,u))/6.
end

function rk4(f_aug::Function)
    # Runge-Kutta 4
    fd(S::Array) = begin
        k1(S) = S[end]*f_aug(S)
        k2(S) = S[end]*f_aug(S + k1(S)/2.)
        k3(S) = S[end]*f_aug(S + k2(S)/2.)
        k4(S) = S[end]*f_aug(S + k3(S))
        S + (k1(S) + 2.*k2(S) + 2.*k3(S) + k4(S))/6.
    end
end

function rk4!(f!::Function, dt::Float64)
    # Runge-Kutta 4
    fd!(xdot,x,u) = begin
        k1 = k2 = k3 = k4 = zeros(x)
        f!(k1, x, u);         k1 *= dt;
        f!(k2, x + k1/2., u); k2 *= dt;
        f!(k3, x + k2/2., u); k3 *= dt;
        f!(k4, x + k3, u);    k4 *= dt;
        copy!(xdot, x + (k1 + 2.*k2 + 2.*k3 + k4)/6.)
    end
end

function rk4!(f_aug!::Function)
    # Runge-Kutta 4
    fd!(dS,S::Array) = begin
        dt = S[end]
        k1 = k2 = k3 = k4 = zeros(S)
        f_aug!(k1,S);         k1 *= dt;
        f_aug!(k2,S + k1/2.); k2 *= dt;
        f_aug!(k3,S + k2/2.); k3 *= dt;
        f_aug!(k4,S + k3);    k4 *= dt;
        copy!(dS, S + (k1 + 2.*k2 + 2.*k3 + k4)/6.)
    end
end



# Assembled augmented function
function f_augmented(model::Model)
    f_aug = f_augmented(model.f, model.n, model.m)
    f(S::Array) = [f_aug(S); zeros(model.m+1,1)]
end

function f_augmented(f::Function, n::Int, m::Int)
    f_aug(S::Array) = f(S[1:n], S[n+(1:m)])
end

function f_augmented!(f!::Function, n::Int, m::Int)
    f_aug!(dS::AbstractArray, S::Array) = f!(dS, S[1:n], S[n+(1:m)])
end
