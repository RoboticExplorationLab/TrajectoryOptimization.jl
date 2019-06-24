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
        fc = zero(x)

        copyto!(y, x)
        g = Inf
        cnt = 0
        while norm(g) > 1.0e-12
            cnt += 1
            if cnt > 1000
                error("Integration convergence fail")
            end

            Xm = 0.5*(x + y)
            f(fc,Xm,u)
            g = y - x - dt*fc

            A = ∇f(Xm,u)[:,1:n]

            ∇g = Diagonal(I,n) - 0.5*dt*A
            δy = -∇g\g

            y .+= δy
        end
    end
end

function midpoint_implicit_uncertain(f::Function,n::Int,m::Int,r::Int,dt::T) where T
    # get estimate of X[k+1] from explicit midpoint
    f_aug(ẋ,z) = f(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    ∇f(x,u,w) = ForwardDiff.jacobian(f_aug,zeros(eltype(x),n),[x;u;w])
    fd(y,x,u,w,dt=dt) = begin
        fc = zero(x)

        copyto!(y, x )

        # iterate to solve implicit midpoint step
        g = Inf
        cnt = 0
        while norm(g) > 1.0e-12
            cnt += 1
            if cnt > 1000
                error("Integration tolerance failed")
            end
            Xm = 0.5*(x + y)
            f(fc,Xm,u,w)
            g = y - x - dt*fc

            A = ∇f(Xm,u,w)[:,1:n]

            ∇g = Diagonal(I,n) - 0.5*dt*A
            δy = -∇g\g

            y .+= δy
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

    fd!(y,x,u,dt=dt) = begin
        fc1 = fc2 = fc3 = zero(x)

        g = Inf
        cnt = 0
        copyto!(y,x)
        while norm(g) > 1.0e-12
            cnt += 1
            if cnt > 1000
                println("norm: \n $(norm(g))")
                error("Integration convergence failed")
            end
            f(fc1,x,u)
            f(fc3,y,u)

            Xm = 0.5*(x + y) + dt/8*(fc1 - fc3)
            f(fc2,Xm,u)

            g = y - x - dt/6*fc1 - 4/6*dt*fc2 - dt/6*fc3

            A1 = ∇f(Xm,u)[:,1:n]
            A2 = ∇f(y,u)[:,1:n]


            ∇g = Diagonal(I,n) - 4/6*dt*A1*(0.5*Diagonal(I,n) - dt/8*A2) - dt/6*A2
            δy = -∇g\g

            y .+= δy
        end
    end
end

function rk3_implicit_uncertain(f::Function,n::Int,m::Int,r::Int,dt::T) where T
    f_aug(ẋ,z) = f(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    ∇f(x,u,w) = ForwardDiff.jacobian(f_aug,zero(x),[x;u;w])

    fd!(y,x,u,w,dt=dt) = begin
        fc1 = fc2 = fc3 = zero(x)

        g = Inf
        cnt = 0
        copyto!(y,x)

        while norm(g) > 1.0e-12
            cnt += 1
            if cnt > 1000
                println(norm(g))
                error("Integration convergence failed")
            end
            f(fc1,x,u,w)
            f(fc3,y,u,w)

            Xm = 0.5*(x + y) + dt/8*(fc1 - fc3)
            f(fc2,Xm,u,w)

            g = y - x - dt/6*fc1 - 4/6*dt*fc2 - dt/6*fc3

            A1 = ∇f(Xm,u,w)[:,1:n]
            A2 = ∇f(y,u,w)[:,1:n]

            ∇g = Diagonal(I,n) - 4/6*dt*A1*(0.5*Diagonal(I,n) - dt/8*A2) - dt/6*A2
            δy = -∇g\g

            y .+= δy
            # copyto!(y,y+δy)
        end
    end
end

## DifferentialEquations.jl
# NOTE: not fast
function DiffEqIntegrator(f!::Function, dt::Float64, integrator::Symbol, n::Int, m::Int)
    function f_aug(z,p,t)
        ż = zero(z)
        f!(view(ż,1:n),z[1:n],z[n .+ (1:m)])
        ż
    end

    function fd_ode(y,x,u,dt=dt)
        _tf = dt
        _t0 = 0.

        u0=vec([x;u])
        tspan = (_t0,_tf)
        pro = ODEProblem(f_aug,u0,tspan)
        sol = OrdinaryDiffEq.solve(pro,eval(integrator)(),dt=dt,save_everystep=false)#,verbose=false)
        copyto!(y,sol.u[end][1:n])
    end
end

function DiffEqIntegratorUncertain(f!::Function, dt::Float64, integrator::Symbol, n::Int, m::Int, r::Int)
    function f_aug(z,p,t)
        ż = zero(z)
        f!(view(ż,1:n),z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
        ż
    end

    function fd_ode(y,x,u,w,dt=dt)
        _tf = dt
        _t0 = 0.
        u0=vec([x;u;w])
        tspan = (_t0,_tf)
        pro = ODEProblem(f_aug,u0,tspan)
        sol = OrdinaryDiffEq.solve(pro,eval(integrator)(),dt=dt,save_everystep=false)#,verbose=false)
        copyto!(y,sol.u[end][1:n])
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
