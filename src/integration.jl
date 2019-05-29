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

function midpoint_uncertain(f!::Function, dt::Float64)
    fd!(xdot,x,u,w,dt=dt) = begin
        f!(xdot,x,u,w)
        xdot .*= dt/2.
        f!(xdot, x + xdot, u, w)
        copyto!(xdot,x + xdot*dt)
    end
end

function midpoint(f_aug!::Function)
    fd_aug!(dS, S) = begin
        dt = S[end]^2
        f_aug!(dS, S)
        dS .*= dt/2.
        f_aug!(dS, S + dS)
        copyto!(dS,S + dS*dt)
    end
end

function midpoint_implicit(f::Function,n::Int,m::Int,dt::T) where T
    # get estimate of X[k+1] from explicit midpoint
    f_aug(ẋ,z) = f(ẋ,z[1:n],z[n .+ (1:m)])
    ∇f(x,u) = ForwardDiff.jacobian(f_aug,zero(x),[x;u])

    fd(y,x,u,dt=dt) = begin
        k1 = k2 = kg = zero(x)
        f(k1, x, u);
        k1 *= dt;
        f(k2, x + k1/2, u);
        k2 *= dt;
        copyto!(y, x)

        for i = 1:10
            Xm = 0.5*(x + y)
            f(kg,Xm,u)
            g = y - x - dt*kg

            A = ∇f(Xm,u)[:,1:n]

            ∇g = Diagonal(I,n) - 0.5*dt*A
            δx = -∇g\g

            println(norm(g))
            y += δx
        end
    end
end

function midpoint_implicit_uncertain(f::Function,n::Int,m::Int,r::Int,dt::T) where T
    # get estimate of X[k+1] from explicit midpoint
    f_aug(ẋ,z) = f(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    ∇f(x,u,w) = ForwardDiff.jacobian(f_aug,zeros(eltype(x),n),[x;u;w])
    fd(y,x,u,w,dt=dt) = begin
        k1 = k2 = kg = zero(x)
        f(k1, x, u, w);
        k1 *= dt;
        # f(k2, x + k1/2, u, w);
        # k2 *= dt;
        copyto!(y, x + k1)#+ k2)

        # iterate to solve implicit midpoint step
        g = Inf
        cnt = 0
        for i = 1:10
            # cnt += 1
            # if cnt > 10
            #     error("Integration tolerance failed")
            # end
            Xm = 0.5*(x + y)
            f(kg,Xm,u,w)
            g = y - x - dt*kg

            A = ∇f(Xm,u,w)[:,1:n]

            ∇g = Diagonal(I,n) - 0.5*dt*A
            δx = -∇g\g

            y += δx
        end
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

function rk4_uncertain(f!::Function, dt::Float64)
    # Runge-Kutta 4
    fd!(xdot,x,u,w,dt=dt) = begin
        k1 = k2 = k3 = k4 = zero(xdot)
        f!(k1, x, u, w);         k1 *= dt;
        f!(k2, x + k1/2, u, w); k2 *= dt;
        f!(k3, x + k2/2, u, w); k3 *= dt;
        f!(k4, x + k3, u, w);    k4 *= dt;
        copyto!(xdot, x + (k1 + 2*k2 + 2*k3 + k4)/6)
    end
end


function rk4(f_aug!::Function)
    # Runge-Kutta 4
    fd!(dS,S::Array) = begin
        dt = S[end]^2
        k1 = k2 = k3 = k4 = zero(S)
        f_aug!(k1,S);         k1 *= dt;
        f_aug!(k2,S + k1/2); k2 *= dt;
        f_aug!(k3,S + k2/2); k3 *= dt;
        f_aug!(k4,S + k3);    k4 *= dt;
        copyto!(dS, S + (k1 + 2*k2 + 2*k3 + k4)/6)
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

function rk3_uncertain(f!::Function, dt::Float64)
        # Runge-Kutta 3 (zero order hold)
    fd!(xdot,x,u,w,dt=dt) = begin
        k1 = k2 = k3 = zero(x)
        f!(k1, x, u, w);               k1 *= dt;
        f!(k2, x + k1/2, u, w);       k2 *= dt;
        f!(k3, x - k1 + 2*k2, u, w);  k3 *= dt;
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

function rk3_implicit(f::Function,n::Int,m::Int,dt::T) where T
    f_aug(ẋ,z) = f(ẋ,z[1:n],z[n .+ (1:m)])
    ∇f(x,u) = ForwardDiff.jacobian(f_aug,zero(x),[x;u])

    fd!(xdot,x,u,dt=dt) = begin
        # get estimate of X[k+1] from explicit rk3
        k1 = k2 = k3 = kg1 = kg2 = kg3 = zero(x)
        # f(k1, x, u);
        # k1 *= dt;
        # f(k2, x + k1/2, u);
        # k2 *= dt;
        # f(k3, x - k1 + 2*k2, u);
        # k3 *= dt;
        copyto!(xdot, x)# + (k1 + 4*k2 + k3)/6)

        g = Inf
        cnt = 0
        while norm(g) > 1.0e-12
            cnt += 1
            if cnt > 10
                error("Integration tolerance failed")
            end
            f(kg1,x,u)
            f(kg3,xdot,u)

            Xm = 0.5*(x + xdot) + dt/8*(kg1 - kg3)
            f(kg2,Xm,u)

            g = xdot - x - dt/6*kg1 - 4/6*dt*kg2 - dt/6*kg3

            A1 = ∇f(Xm,u)[:,1:n]
            A2 = ∇f(xdot,u)[:,1:n]


            ∇g = Diagonal(I,n) - 4/6*dt*A1*(0.5*Diagonal(I,n) - dt/8*A2) - dt/6*A2
            δx = -∇g\g

            xdot += δx
        end
    end
end

function rk3_implicit_uncertain(f::Function,n::Int,m::Int,r::Int,dt::T) where T
    f_aug(ẋ,z) = f(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    ∇f(x,u,w) = ForwardDiff.jacobian(f_aug,zero(x),[x;u;w])

    fd!(xdot,x,u,w,dt=dt) = begin
        # get estimate of X[k+1] from explicit rk3
        k1 = k2 = k3 = kg1 = kg2 = kg3 = zero(x)
        # f(k1, x, u, w);
        # k1 *= dt;
        # f(k2, x + k1/2, u, w);
        # k2 *= dt;
        # f(k3, x - k1 + 2*k2, u, w);
        # k3 *= dt;
        copyto!(xdot, x)# + (k1 + 4*k2 + k3)/6)

        g = Inf
        cnt = 0
        while norm(g) > 1.0e-12
            cnt += 1
            if cnt > 10
                error("Integration tolerance failed")
            end
            f(kg1,x,u,w)
            f(kg3,xdot,u,w)

            Xm = 0.5*(x + xdot) + dt/8*(kg1 - kg3)
            f(kg2,Xm,u,w)

            g = xdot - x - dt/6*kg1 - 4/6*dt*kg2 - dt/6*kg3

            A1 = ∇f(Xm,u,w)[:,1:n]
            A2 = ∇f(xdot,u,w)[:,1:n]


            ∇g = Diagonal(I,n) - 4/6*dt*A1*(0.5*Diagonal(I,n) - dt/8*A2) - dt/6*A2
            δx = -∇g\g

            xdot += δx
        end
    end
end

"""
$(SIGNATURES)
Converts a separated dynamics function into an augmented dynamics function
"""
function f_augmented!(f!::Function, n::Int, m::Int)
    f_aug!(dS::AbstractArray, S::Array) = f!(dS, S[1:n], S[n+1:n+m])
end

function f_augmented(f::Function, n::Int, m::Int)
    f_aug(S::Array) = f(S[1:n], S[n+1:n+m])
end

function ZeroOrderHoldInterpolation(t,X)
    itr = interpolate(X,BSpline(Constant()))
    dt = t[2] - t[1]
    function zoh(t2)
        i2 = t2./dt .+ 1
        itr(floor.(i2))
    end
end

function MidpointInterpolation(t,X)
    Xm = [(X[i] + X[i+1])/2 for i = 1:length(X)-1]
    push!(Xm,Xm[end])
    ZeroOrderHoldInterpolation(t,Xm)
end
