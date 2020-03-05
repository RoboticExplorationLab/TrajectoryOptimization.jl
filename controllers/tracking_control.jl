export
    get_control,
    TVLQR,
    LQR,
    MLQR,
    simulate,
    SE3Tracking,
    HFCA,
    test_ICs



import TrajectoryOptimization.Dynamics: trim_controls, build_state
import TrajectoryOptimization: state_diff, state_diff_jacobian
abstract type AbstractController end
abstract type LQRController <: AbstractController end
abstract type TimeVaryingController <: AbstractController end
abstract type TrackingController <: TimeVaryingController end


function get_k(cntrl::TimeVaryingController, t)
    times = get_times(cntrl)
    searchsortedlast(times,t)
end


""" TVLQR
"""
struct TVLQR{L,TK} <: TimeVaryingController
    model::L
    Z::Traj
    K::Vector{TK}
    t::Vector{Float64}
end

""" Assumes the states of the trajectory Z match the states of the model
"""
function TVLQR(model::AbstractModel, Q, R, Z::Traj)
    A,B = linearize(model, Z)
    K = tvlqr(A,B,Q,R)
    t = [z.t for z in Z]
    TVLQR(model,Z,K,t)
end

function TVLQR(model::RigidBody, Q, R, X::Vector{<:RBState}, U::Vector{<:AbstractVector}, dt)
    # Convert the trajectory to the orientation representation of the model
    Z = Traj(model, X, U, dt)

    # Call as normal
    TVLQR(model, Q, R, Z::Traj)
end

@inline get_times(cntrl::TVLQR) = cntrl.t

function get_control(cntrl::TVLQR, x, t)
    k = get_k(cntrl, t)
    zref = cntrl.Z[k]
    dx = state_diff(cntrl.model, x, state(zref))
    return cntrl.K[k]*dx + control(zref)
end

function tvlqr(A, B, Q, R)
    n,m = size(B[1])
    K = [@SMatrix zeros(m,n) for k = 1:length(A)]
    tvlqr!(K, A, B, Q, R)
end

function tvlqr!(K, A, B, Q, R)
    N = length(A)

    # Solve infinite-horizon at goal state
    # Qf = dare(A[N], B[N], Q, R)
    Qf = Q

    P_ = similar_type(A[N])
    P = copy(Qf)
    for k = N-1:-1:1
        P,K[k] = riccati(P,A[k],B[k],Q,R)
    end
    return K
end

function riccati(P,A,B,Q,R)
    P_ = Q + A'P*A - A'P*B*((R+B'P*B)\(B'P*A))
    K = -(R+B'P*B)\(B'P*A)
    return P_,K
end


""" Discrete LQR
"""
struct LQR{T,N,M,TK} <: LQRController
    K::TK
    xref::SVector{N,T}
    uref::SVector{M,T}
end

function LQR(model::AbstractModel, dt::Real, Q, R,
        xeq=zeros(model)[1], ueq=trim_controls(model))
    # Linearize the model
    A,B = linearize(model, xeq, ueq, dt)

    # Calculate the optimal control gain
    K = calc_LQR_gain(A,B,Q,R)
    LQR(K, xeq, ueq)
end

function get_control(cntrl::LQR, x, t)
    dx = x - cntrl.xref
    return cntrl.K*dx + cntrl.uref
end

""" Multiplicative LQR
"""
struct MLQR{T,N,M,TK} <: LQRController
    model::AbstractModel
    K::TK
    xref::SVector{N,T}
    uref::SVector{M,T}
end


function MLQR(model::AbstractModel, dt::Real, Q::AbstractMatrix, R::AbstractMatrix,
        xeq=zeros(model)[1], ueq=trim_controls(model))
    # Linearize the model, accounting for attitude state
    A,B = linearize(model, xeq, ueq, dt)
    G1 = state_diff_jacobian(model, xeq)
    G2 = state_diff_jacobian(model, discrete_dynamics(RK3, model, xeq, ueq, 0.0, dt))
    A = G2'A*G1
    B = G2'B

    # Calculate the optimal control gain
    @assert size(Q) == size(A)
    K = calc_LQR_gain(A,B,Q,R)
    MLQR(model,K,xeq,ueq)
end

function get_control(cntrl::MLQR, x, t)
    dx = state_diff(cntrl.model, x, cntrl.xref)
    return cntrl.K*dx + cntrl.uref
end

""" SE(3) Quadrotor controller
from [Lee 2010]
"""
struct SE3Tracking{T,N} <: TrackingController
    model::AbstractModel
    kx::T
    kv::T
    kR::T
    kΩ::T
    Xref::Vector{SVector{N,T}}
    Xdref::Vector{SVector{N,T}}
    bref::Vector{SVector{3,T}}
    t::Vector{T}
end

function SE3Tracking(model::AbstractModel, Xref, Xdref, bref, t;
        # kx=59.08, kv=24.3, kR=8.81, kO=1.54)
        kx=2.71, kv=1.01, kR=2.26, kO=0.1)
    SE3Tracking(model, kx, kv, kR, kO, Xref, Xdref, bref, t)
end

@inline get_times(cntrl::SE3Tracking) = cntrl.t

function get_control(c::SE3Tracking, x, t)
    # Get model params
    g = c.model.gravity[3]
    e3 = @SVector [0,0,1.]
    mass = c.model.mass
    J = c.model.J

    # Get time step
    k = get_k(c, t)
    b1d = c.bref[k]
    xd = c.Xref[k]

    # Parse the state
    r,q,v,Ω = Dynamics.parse_state(c.model, x)
    rd,qd,vd,Ωd = Dynamics.parse_state(c.model, xd)
    rdd,qdd,vdd,Ωdd = Dynamics.parse_state(c.model, c.Xdref[k])
    R = rotmat(q)

    # Calculate the linear errors
    ex = r-rd
    ev = v-vd
    xdd = rdd
    Ωdotd = Ωdd

    # Calculate desired attitude
    a = -c.kx*ex - c.kv*ev - mass*g*e3 + mass*xdd
    b3 = normalize(a)
    b2 = normalize(b3 × b1d)
    b1 = b2 × b3
    Rd = @SMatrix [b1[1] b2[1] b3[1];
                   b1[2] b2[2] b3[2];
                   b1[3] b2[3] b3[3]]

    # Calculate the attitude errors
    eR = 0.5*vee(Rd'R - R'Rd)
    eΩ = Ω - R'Rd*Ωd

    # Desired Forces and moments
    f = -a'R*e3
    M = -c.kR*eR - c.kΩ*eΩ + Ω × (J*Ω) - J*(skew(Ω)*R'Rd*Ωd - R'Rd*Ωdotd)

    # Convert to motor thrusts
    C = Dynamics.forceMatrix(c.model)
    u = C\(@SVector [f, M[1], M[2], M[3]])
end


""" Hopf Fibration control
"""
struct HFCA{N,T} <: TrackingController
    model::AbstractModel
    kx::T
    kv::T
    kR::Diagonal{T,SVector{3,T}}
    kΩ::Diagonal{T,SVector{3,T}}
    Xref::Vector{SVector{N,T}}
    Xdref::Vector{SVector{N,T}}
    bref::Vector{SVector{3,T}}
    t::Vector{T}
end

function HFCA(model::AbstractModel, Xref, Xdref, bref, t;
        # kx=59.08, kv=24.3, kR=8.81, kO=1.54)
        kx=11.7, kv=4.0,
        kR=3.0*Diagonal(@SVector ones(3)),
        kO=0.7*Diagonal(@SVector ones(3)))
    HFCA(model, kx, kv, kR, kO, Xref, Xdref, bref, t)
end

get_times(c::HFCA) = c.t

function get_control(c::HFCA, x, t)
    # Get model params
    g = c.model.gravity[3]
    e3 = @SVector [0,0,1.]
    mass = c.model.mass
    J = c.model.J

    # Get time step
    k = get_k(c, t)
    b1d = c.bref[k]
    xd = c.Xref[k]
    xdd = c.Xdref[k]

    # Parse the state
    r,q,v,Ω = Dynamics.parse_state(c.model, x)
    rd,qd,vd,Ωd = Dynamics.parse_state(c.model, xd)
    rdd,qdd,vdd,Ωdd = Dynamics.parse_state(c.model, xdd)
    R = rotmat(q)

    # Calculate the linear errors
    ex = r-rd
    ev = v-vd
    xdd = rdd
    Ωdotd = Ωdd

    # Get the desired heading
    ψ0 = atan(c.bref[1][2], c.bref[1][1])
    ψ = atan(b1d[2], b1d[1]) - ψ0

    # Calculate desired angular velocities and quaternion
    ζ = -c.kx*ex - c.kv*ev + mass*rdd - mass*g*e3  # eq (26)
    ζdot = -c.kx*ev + vdd  # is this right?
    abc = normalize(ζ)  # eq (19)
    abc_dot = (ζ'ζ*I - ζ*ζ')*ζdot/(norm(ζ)^3)  # eq (20)
    ψdot = -(abc[2]*abc_dot[1] - abc[1]*abc_dot[2])/(1+abc[3])
    ω = (@SVector [
        sin(ψ)*abc_dot[1] - cos(ψ)*abc_dot[2] - (abc[1]*sin(ψ) - abc[2]*cos(ψ)),
        cos(ψ)*abc_dot[1] + sin(ψ)*abc_dot[2] - (abc[1]*cos(ψ) + abc[2]*sin(ψ)),
        0
    ]) * (abc_dot[3]/(abc[3] + 1))
    q_abc = @SVector [1+abc[3], -abc[2], abc[1], 0]
    q_ψ = @SVector [cos(ψ/2), 0, 0, sin(ψ/2)]
    qd = Lmult(q_abc)*q_ψ
    Rd = rotmat(UnitQuaternion(qd))

    # Calculate angular errors
    eΩ = Ω - R'Rd*Ωd
    eR = 0.5*vee(Rd'R - R'Rd)

    # Calculate controls
    f = -ζ'R*e3
    M = -c.kR*eR - c.kΩ*eΩ

    # Convert to motor thrusts
    C = Dynamics.forceMatrix(c.model)
    u = C\(@SVector [f, M[1], M[2], M[3]])
end


############################################################################################
############################################################################################

function linearize(model::AbstractModel, xeq, ueq, dt)
    # Linearize the system about the given point
    n,m = size(model)
    z = KnotPoint(xeq, ueq, dt)
    ∇f = zeros(n,n+m+1)
    discrete_jacobian!(RK3, ∇f, model, z)
    ix,iu = z._x, z._u
    A = ∇f[ix,ix]
    B = ∇f[ix,iu]
    return A,B
end

# TODO: redo without static matrices
function linearize(model::AbstractModel, Z::Traj)
    N = length(Z)
    D = [TO.SizedDynamicsExpansion(model) for k = 1:N]
    for k = 1:N
        _linearize!(D[k], model, Z[k])
    end
    A = [SMatrix(d.A) for d in D]
    B = [SMatrix(d.B) for d in D]
    return A,B
end

function _linearize!(D::TO.SizedDynamicsExpansion, model::AbstractModel, z::KnotPoint)
    discrete_jacobian!(RK3, D.∇f, model, z)
	D.tmpA .= D.A_  # avoids allocations later
	D.tmpB .= D.B_
    x2 = discrete_dynamics(RK3, model, z)
    G1 = TO.state_diff_jacobian(model, state(z))
    G2 = TO.state_diff_jacobian(model, x2)
    TO.error_expansion!(D,G1,G2)
    return nothing
end

function calc_LQR_gain(A::AbstractMatrix, B::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        max_iters=100, tol=1e-4)
    P_ = similar_type(A)
    P = P_(A)
    for k = 1:max_iters
        P_ = Q + A'P*A - A'P*B*((R + B'P*B)\(B'P*A))
        err = norm(P-P_)
        P = copy(P_)
        if err < tol
            break
        end
    end
    K = -(R + B'P*B)\(B'P*A)
end

function dare(A,B,Q,R,
        max_iters=100, tol=1e-4)
    P_ = similar_type(A)
    P = P_(A)
    for k = 1:max_iters
        P_ = Q + A'P*A - A'P*B*((R + B'P*B)\(B'P*A))
        err = norm(P-P_)
        P = copy(P_)
        if err < tol
            break
        end
    end
    return P
end

simulate(model::RigidBody, cntrl, x0::RBState, tf; kwargs...) =
    simulate(model, cntrl, build_state(model, x0), tf; kwargs...)

function simulate(model::AbstractModel, cntrl, x0, tf;
        dt=1e-4, w=1e-2)
    N = Int(round(tf/dt)) + 1
    dt = tf/(N-1)
    x = copy(x0)
    t = 0.0

    n,m = size(model)

    # Allocate storage for the state trajectory
    X = [@SVector zeros(n) for k = 1:N]

    for k = 1:N
        # Get control from the controller
        u = get_control(cntrl, x, t)

        # Add disturbances
        u += randn(m)*w

        # Simulate the system forward
        x = discrete_dynamics(RK3, model, x, u, t, dt)
        t += dt  # advance time

        # Store info
        X[k] = x
    end
    return X
end

function test_ICs(model::AbstractModel, cntrl::AbstractController, ICs::Vector{<:RBState};
        tf=10., dt=1e-4, xref=(@SVector zeros(N))) where N
    L = length(ICs)
    Xf = deepcopy(ICs)
    data = Dict{Symbol,Vector}(:Xf=>deepcopy(ICs),
        :max_err=>zeros(L), :avg_err=>zeros(L), :term_err=>zeros(L))
    for i = 1:L
        x0 = Dynamics.build_state(model, ICs[i])
        X = simulate(model, cntrl, x0, tf, dt=dt, w=0.0)
        err = map(X) do x
            dx = Dynamics.state_diff(model, x, x0)
            norm(dx)
        end
        data[:max_err][i] = maximum(err)
        data[:avg_err][i] = mean(err)
        data[:term_err][i] = err[end]
        data[:Xf][i] = RBState(model, X[end])
    end
    data
end
