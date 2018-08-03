
"""
$(SIGNATURES)
In place Midpoint integration

Defines methods for both separated and augmented forms. Returns a discrete version
of a continuous dynamics function.

# Arguments
* f!: in place dynamics function, i.e. `f!(ẋ,x,u)`
* dt: time step

"""
function midpoint(f!::Function, dt::Float64)
    fd!(xdot,x,u) = begin
        f!(xdot,x,u)
        xdot .*= dt/2.
        f!(xdot, x + xdot, u)
        copy!(xdot,x + xdot*dt)
    end
end

function midpoint(f_aug!::Function)
    fd_aug!(dS, S) = begin
        dt = S[end]
        f_aug!(dS, S)
        dS .*= dt/2.
        f_aug!(dS, S + dS)
        copy!(dS,S + dS*dt)
    end
end

"""
$(SIGNATURES)
In place Runge Kutta 4 integration

Defines methods for both separated and augmented forms. Returns a discrete version
of a continuous dynamics function.

# Arguments
* f!: in place dynamics function, i.e. `f!(ẋ,x,u)` for separate or `f!(Ṡ,S)` for augmented dynamics
* dt: time step
"""
function rk4(f!::Function, dt::Float64)
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

function rk4(f_aug!::Function)
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


# # Assembled augmented function
# function f_augmented(model::Model)
#     f_aug = f_augmented(model.f, model.n, model.m)
#     f(S::Array) = [f_aug(S); zeros(model.m+1,1)]
# end
#
# function f_augmented(f::Function, n::Int, m::Int)
#     f_aug(S::Array) = f(S[1:n], S[n+(1:m)])
# end

"""
$(SIGNATURES)
Converts a separated dynamics function into an augmented dynamics function
"""
function f_augmented!(f!::Function, n::Int, m::Int)
    f_aug!(dS::AbstractArray, S::Array) = f!(dS, S[1:n], S[n+(1:m)])
end
