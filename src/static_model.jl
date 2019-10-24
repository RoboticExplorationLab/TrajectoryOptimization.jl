
"Generate discrete dynamics function for a dynamics model using RK3 integration"
function rk3_gen(model::AbstractModel)
       # Runge-Kutta 3 (zero order hold)
   @eval begin
       function discrete_dynamics(model::$(typeof(model)), x, u, dt)
           k1 = dynamics(model, x, u)*dt;
           k2 = dynamics(model, x + k1/2, u)*dt;
           k3 = dynamics(model, x - k1 + 2*k2, u)*dt;
           x + (k1 + 4*k2 + k3)/6
       end
   end
end


"""
Generate the continuous dynamics Jacobian for a dynamics model.
The resulting function should be non-allocating if the original dynamics function is non-allocating
"""
function generate_jacobian(model::M) where {M<:AbstractModel}
    n,m = model.n, model.m
    ix,iu = 1:n, n .+ (1:m)
    f_aug(z) = dynamics(model, view(z,ix), view(z,iu))
    ∇f(z) = ForwardDiff.jacobian(f_aug,z)
    ∇f(x::SVector,u::SVector) = ∇f([x;u])
    ∇f(x,u) = begin
        z = zeros(n+m)
        z[ix] = x
        z[iu] = u
        ∇f(z)
    end
    @eval begin
        jacobian(model::$(M), x, u) = $(∇f)(x, u)
        jacobian(model::$(M), z) = $(∇f)(z)
    end
end

"""
Generate the discrete dynamics Jacobian for a dynamics model
"""
function generate_discrete_jacobian(model::M) where {M<:AbstractModel}
    ix,iu,idt = 1:n, n .+ (1:m), n+m+1
    fd_aug(z) = discrete_dynamics(model, view(z,ix), view(z,iu), z[idt])
    ∇fd(z) = ForwardDiff.jacobian(fd_aug, z)
    ∇fd(z,dt) = ForwardDiff.jacobian(fd_aug, [z; @SVector [dt]])
    ∇fd(x,u::SVector,dt) = ∇fd([x;u; @SVector [dt]])
    ∇fd(x,u,dt) = begin
        z = zeros(n+m)
        z[ix] = x
        z[iu] = u
        z[idt] = dt
        ∇fd(z)
    end
    @eval begin
        discrete_jacobian(model::$(M), x, u, dt) = $(∇fd)(x, u, dt)
        discrete_jacobian(model::$(M), z::AbstractVector) = $(∇fd)(z)
        discrete_jacobian(model::$(M), z, dt) = $(∇fd)(z, dt)
    end
end
