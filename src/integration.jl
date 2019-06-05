# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Integration schemes
#
#     INTEGRATION METHODS
#         midpoint: midpoint or trapezoidal integration
#         rk4: Runge-Kutta 4
#         rk3: Runge-Kutta 3
#     OTHER METHODS
#         f_augmented!: Create function with augmented state and control input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO: Change S to Z

"""
$(SIGNATURES)
In place Midpoint integration

Defines methods for both separated and augmented forms. Returns a discrete version
of a continuous dynamics function.

# Arguments
* f!: in place dynamics function, i.e. `f!(x?,x,u)`
* dt: time step

"""
function midpoint(f!::Function, dt::Float64)
    fd!(xdot,x,u,dt=dt) = begin
        f!(xdot,x,u)
        xdot .*= dt/2.
        f!(xdot, x + xdot, u)
        copyto!(xdot,x + xdot*dt)
    end
end


"""
$(SIGNATURES)
In place Runge Kutta 4 integration

Defines methods for both separated and augmented forms. Returns a discrete version
of a continuous dynamics function.

# Arguments
* f!: in place dynamics function, i.e. `f!(x?,x,u)` for separate or `f!(S?,S)` for augmented dynamics
* dt: time step
"""
function rk4(f!::Function, dt::Float64)
    # Runge-Kutta 4
    fd!(xdot,x,u,dt=dt) = begin
        k1 = k2 = k3 = k4 = zero(xdot)
        f!(k1, x, u);         k1 *= dt;
        f!(k2, x + k1/2, u); k2 *= dt;
        f!(k3, x + k2/2, u); k3 *= dt;
        f!(k4, x + k3, u);    k4 *= dt;
        copyto!(xdot, x + (k1 + 2*k2 + 2*k3 + k4)/6)
    end
end


"""
$(SIGNATURES)
In place Runge Kutta 3 integration

Defines methods for both separated and augmented forms. Returns a discrete version
of a continuous dynamics function.

# Arguments
* f!: in place dynamics function, i.e. `f!(x?,x,u)` for separate or `f!(S?,S)` for augmented dynamics
* dt: time step
"""
function rk3(f!::Function, dt::Float64)
        # Runge-Kutta 3 (zero order hold)
    fd!(xdot,x,u,dt=dt) = begin
        k1 = k2 = k3 = zero(x)
        f!(k1, x, u);               k1 *= dt;
        f!(k2, x + k1/2, u);       k2 *= dt;
        f!(k3, x - k1 + 2*k2, u);  k3 *= dt;
        copyto!(xdot, x + (k1 + 4*k2 + k3)/6)
    end
end

function rk3(f_aug!::Function)
    # Runge-Kutta 3 augmented (zero order hold)
    fd!(dS,S::Array) = begin
        dt = S[end]^2
        k1 = k2 = k3 = zero(S)
        f_aug!(k1,S);              k1 *= dt;
        f_aug!(k2,S + k1/2);      k2 *= dt;
        f_aug!(k3,S - k1 + 2*k2); k3 *= dt;
        copyto!(dS, S + (k1 + 4*k2 + k3)/6)
    end
end

"""
$(SIGNATURES)
Converts a separated dynamics function into an augmented dynamics function
"""
function f_augmented!(f!::Function, n::Int, m::Int)
    f_aug!(dS::AbstractArray, S::Array) = f!(dS, S[1:n], S[n+1:n+m])
end
