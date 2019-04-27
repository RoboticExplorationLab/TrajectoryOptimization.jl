using ForwardDiff, Plots, LinearAlgebra, BenchmarkTools, MatrixCalculus,
    PartedArrays

## DIRTREL tests
function pendulum_dynamics_stochastic!(ẋ,x,u,w)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    ẋ[1] = x[2]
    ẋ[2] = (u[1] - (m + w[1])*g*lc*sin(x[1]) - b*x[2])/I
end

function rk4_stochastic!(f!::Function, dt)
    # Runge-Kutta 4
    fd!(xdot,dt,x,u,w) = begin
        k1 = zero(xdot)
        k2 = zero(xdot)
        k3 = zero(xdot)
        k4 = zero(xdot)
        f!(k1, x, u, w);         k1 *= dt;
        f!(k2, x + k1/2, u, w); k2 *= dt;
        f!(k3, x + k2/2, u, w); k3 *= dt;
        f!(k4, x + k3, u, w);    k4 *= dt;
        copyto!(xdot, x + (k1 + 2*k2 + 2*k3 + k4)/6)
    end
end


function midpoint_stochastic!(f!::Function, dt::Float64)
    fd!(xdot,dt,x,u,w) = begin
        f!(xdot,x,u,w)
        xdot .*= dt/2.
        f!(xdot, x + xdot, u, w)
        copyto!(xdot,x + xdot*dt)
    end
end

function forward_euler!(f::Function,dt)
    fd(ẋ,dt,x,u,w) = begin
        v = zero(x)
        f(v,x,u,w)
        return copyto!(ẋ,x + v*dt)
    end
end

function f_augmented!(f!::Function, nx::Int, nu::Int, nw::Int)
    f_aug!(dS::AbstractArray, S::Array) = f!(dS, S[1], S[1 .+ (1:nx)], S[(1+nx) .+ (1:nu)], S[(1+nx+nu) .+ (1:nw)])
end

function pack(X,U,dt)
    N = length(X)
    Z̄ = [k != N ? [dt;X[k];U[k]] : X[N] for k = 1:N]
    Z = vcat(Z̄...)
end

function unpack_aug(Z,n,m,N)
    idx = 1+n+m
    X = Vector[]
    U = Vector[]
    dt = Vector[]

    for k = 1:N-1
        z = Z[((k-1)*idx+1):k*idx]
        push!(dt,[z[1]])
        push!(X,z[1 .+ (1:n)])
        push!(U,z[(1+n) .+ (1:m)])
    end
    push!(X,Z[end-n:end])

    return X,U,dt
end

function robust_cost_function(Z,∇f,E1,D,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)

    T = eltype(Z)
    A = zeros(T,nx,nx,N-1)
    B = zeros(T,nx,nu,N-1)
    G = zeros(T,nx,nw,N-1)

    K = zeros(T,nu,nx,N-1)

    E = zeros(T,nx,nx,N)
    H = zeros(T,nx,nw)


    idx = 1+nx+nu
    w = zeros(T,nw)

    for k = 1:N-1

        h = Z[((k-1)*idx+1)]
        x = Z[(1 + (k-1)*idx) .+ (1:nx)]
        u = Z[(1 + nx + (k-1)*idx) .+ (1:nu)]

        F = ∇f([h;x;u;w])

        A[:,:,k] = F[:,1 .+ (1:nx)]
        B[:,:,k] = F[:,(1 + nx) .+ (1:nu)]
        G[:,:,k] = F[:,(1+nx+nu) .+ (1:nw)]

    end

    P = Qf
    for k = N-1:-1:1
        K[:,:,k] = (R + B[:,:,k]'*P*B[:,:,k])\(B[:,:,k]'*P*A[:,:,k])
        P = Q + K[:,:,k]'*R*K[:,:,k] + (A[:,:,k] - B[:,:,k]*K[:,:,k])'*P*(A[:,:,k] - B[:,:,k]*K[:,:,k]);
    end

    for k = 1:N-1
        E[:,:,k+1] = (A[:,:,k] - B[:,:,k]*K[:,:,k])*E[:,:,k]*(A[:,:,k] - B[:,:,k]*K[:,:,k])' + (A[:,:,k] - B[:,:,k]*K[:,:,k])*H*G[:,:,k]' + G[:,:,k]*H'*(A[:,:,k] - B[:,:,k]*K[:,:,k])' + G[:,:,k]*D*G[:,:,k]'
        H = (A[:,:,k] - B[:,:,k]*K[:,:,k])*H + G[:,:,k]*D
    end

    ℓ = 0.
    for k = 1:N-1
        ℓ += tr((Qr + K[:,:,k]'*Rr*K[:,:,k])*E[:,:,k])
    end
    ℓ += tr(Qfr*E[:,:,N])

    return ℓ
end

function gen_rcf(_rcf,_dyn,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)

    dyn = f_augmented!(_dyn,nx,nu,nw)

    _∇f(z,nx) = ForwardDiff.jacobian(dyn,zeros(eltype(z),nx),z)
    function gen_∇f(_∇f,nx)
        ∇f(z) = _∇f(z,nx)
    end
    ∇f = gen_∇f(_∇f,nx)

    function rcf(Z)
        _rcf(Z,∇f,E1,D,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
    end

    return rcf
end

function rcb_port(Z)
    X,U,dt = unpack_aug(Z,nX,nU,N)
    A = zeros(nX,nX,N-1)
    B = zeros(nX,nU,N-1)
    G = zeros(nX,nW,N-1)

    dA = zeros(nX*nX,1+nX+nU,N-1)
    dB = zeros(nX*nU,1+nX+nU,N-1)
    dG = zeros(nX*nW,1+nX+nU,N-1)

    for k = 1:(N-1)
        z = [dt[k];X[k];U[k];0.]
        dx = ∇f(z)
        d2x = ∇²f(z)

        A[:,:,k] = dx[:,1 .+ (1:nX)]
        B[:,:,k] = dx[:,(1+nX) .+ (1:nU)]
        G[:,:,k] = dx[:,(1+nX+nU) .+ (1:nW)]

        dA[:,:,k] = d2x[nX .+ (1:nX*nX),1:(1+nX+nU)]
        dB[:,:,k] = d2x[((1+nX)*nX) .+ (1:nX*nU),1:(1+nX+nU)]
        dG[:,:,k] = d2x[((1+nX+nU)*nX) .+ (1:nX*nW),1:(1+nX+nU)]
    end

    # Solve Riccati Equation
    P = Qf;
    dP = zeros(nX*nX,(N-1)*(1+nX+nU)+nX);
    K = zeros(nU,nX,N-1);
    dK = zeros(nU*nX,(N-1)*(1+nX+nU)+nX,N-1);
    for k = (N-1):-1:1
        K[:,:,k] = (B[:,:,k]'*P*B[:,:,k] + R)\(B[:,:,k]'*P*A[:,:,k])
        dKdA = kron(Matrix(I,nX,nX),(B[:,:,k]'*P*B[:,:,k] + R)\B[:,:,k]'P)
        dKdB = kron(A[:,:,k]'*P, inv(B[:,:,k]'*P*B[:,:,k] + R))*comm(nX,nU) - kron(A[:,:,k]'*P*B[:,:,k], Matrix(I,nU,nU))*kron(inv(B[:,:,k]'*P*B[:,:,k] + R)', inv(B[:,:,k]'*P*B[:,:,k] + R))*(kron(Matrix(I,nU,nU), B[:,:,k]'*P) + kron(B[:,:,k]'*P, Matrix(I,nU,nU))*comm(nX,nU));
        dKdP = kron(A[:,:,k]', (B[:,:,k]'*P*B[:,:,k]+R)\B[:,:,k]') - kron(A[:,:,k]'*P*B[:,:,k], Matrix(I,nU,nU))*kron(inv(B[:,:,k]'*P*B[:,:,k]+R)', inv(B[:,:,k]'*P*B[:,:,k]+R))*kron(B[:,:,k]', B[:,:,k]');
        dK[:,:,k] = dKdP*dP;
        dK[:,((k-1)*(1+nX+nU)) .+ (1:(1+nX+nU)),k] = dK[:,((k-1)*(1+nX+nU)).+(1:(1+nX+nU)),k] + dKdA*dA[:,:,k] + dKdB*dB[:,:,k];

        dPdA = kron(Matrix(I,nX,nX), (A[:,:,k]-B[:,:,k]*K[:,:,k])'*P) + kron((A[:,:,k]-B[:,:,k]*K[:,:,k])'*P, Matrix(I,nX,nX))*comm(nX,nX);
        dPdB = -kron(Matrix(I,nX,nX), (A[:,:,k]-B[:,:,k]*K[:,:,k])'*P)*kron(K[:,:,k]', Matrix(I,nX,nX)) - kron((A[:,:,k]-B[:,:,k]*K[:,:,k])'*P, Matrix(I,nX,nX))*kron(Matrix(I,nX,nX), K[:,:,k]')*comm(nX,nU);
        dPdK = kron(Matrix(I,nX,nX), K[:,:,k]'*R) + kron(K[:,:,k]'*R, Matrix(I,nX,nX))*comm(nU,nX) - kron(Matrix(I,nX,nX), (A[:,:,k]-B[:,:,k]*K[:,:,k])'*P)*kron(Matrix(I,nX,nX), B[:,:,k]) - kron((A[:,:,k]-B[:,:,k]*K[:,:,k])'*P, Matrix(I,nX,nX))*kron(B[:,:,k], Matrix(I,nX,nX))*comm(nU,nX);
        dPdP = kron((A[:,:,k]-B[:,:,k]*K[:,:,k])', (A[:,:,k]-B[:,:,k]*K[:,:,k])');

        P = Q + K[:,:,k]'*R*K[:,:,k] + (A[:,:,k] - B[:,:,k]*K[:,:,k])'*P*(A[:,:,k] - B[:,:,k]*K[:,:,k]);
        dP = dPdP*dP + dPdK*dK[:,:,k];
        dP[:,((k-1)*(1+nX+nU)) .+ (1:(1+nX+nU))] = dP[:,((k-1)*(1+nX+nU)) .+ (1:(1+nX+nU))] + dPdA*dA[:,:,k] + dPdB*dB[:,:,k];
    end


    E = zeros(nX, nX, N);
    E[:,:,1] = E1;
    dE = zeros(nX*nX, (N-1)*(1+nX+nU)+nX, N)
    H = zeros(nX, nW);
    dH = zeros(nX*nW, (N-1)*(1+nX+nU)+nX);

    for k = 1:(N-1)
        dEdA = kron(Matrix(I,nX,nX), A[:,:,k]*E[:,:,k])*comm(nX,nX) + kron(A[:,:,k]*E[:,:,k], Matrix(I,nX,nX)) - kron(B[:,:,k]*K[:,:,k]*E[:,:,k], Matrix(I,nX,nX)) - kron(Matrix(I,nX,nX), B[:,:,k]*K[:,:,k]*E[:,:,k])*comm(nX,nX) + kron(G[:,:,k]*H',Matrix(I,nX,nX)) + kron(Matrix(I,nX,nX),G[:,:,k]*H')*comm(nX,nX);
        dEdB = -kron(Matrix(I,nX,nX), A[:,:,k]*E[:,:,k]*K[:,:,k]')*comm(nX,nU) - kron(A[:,:,k]*E[:,:,k]*K[:,:,k]', Matrix(I,nX,nX)) + kron(Matrix(I,nX,nX), B[:,:,k]*K[:,:,k]*E[:,:,k]*K[:,:,k]')*comm(nX,nU) + kron(B[:,:,k]*K[:,:,k]*E[:,:,k]*K[:,:,k]', Matrix(I,nX,nX)) - kron(G[:,:,k]*H'*K[:,:,k]',Matrix(I,nX,nX)) - kron(Matrix(I,nX,nX),G[:,:,k]*H'*K[:,:,k]')*comm(nX,nU);
        dEdG = kron(Matrix(I,nX,nX),(A[:,:,k]-B[:,:,k]*K[:,:,k])*H)*comm(nX,nW) + kron((A[:,:,k]-B[:,:,k]*K[:,:,k])*H,Matrix(I,nX,nX)) + kron(Matrix(I,nX,nX), G[:,:,k]*D)*comm(nX,nW) + kron(G[:,:,k]*D, Matrix(I,nX,nX));
        dEdK = -kron(B[:,:,k], A[:,:,k]*E[:,:,k])*comm(nU,nX) - kron(A[:,:,k]*E[:,:,k], B[:,:,k]) + kron(B[:,:,k]*K[:,:,k]*E[:,:,k], B[:,:,k]) + kron(B[:,:,k],B[:,:,k]*K[:,:,k]*E[:,:,k])*comm(nU,nX) - kron(G[:,:,k]*H',B[:,:,k]) - kron(B[:,:,k],G[:,:,k]*H')*comm(nU,nX);
        dEdH = kron(G[:,:,k], A[:,:,k]-B[:,:,k]*K[:,:,k]) + kron(A[:,:,k]-B[:,:,k]*K[:,:,k], G[:,:,k])*comm(nX,nW);
        dEdE = kron(A[:,:,k]-B[:,:,k]*K[:,:,k], A[:,:,k]-B[:,:,k]*K[:,:,k]);

        dHdA = kron(H', Matrix(I,nX,nX));
        dHdB = -kron((K[:,:,k]*H)', Matrix(I,nX,nX));
        dHdG = kron(D, Matrix(I,nX,nX));
        dHdK = -kron(H', B[:,:,k]);
        dHdH = kron(Matrix(I,nW,nW), (A[:,:,k]-B[:,:,k]*K[:,:,k]));

        E[:,:,k+1] = (A[:,:,k]-B[:,:,k]*K[:,:,k])*E[:,:,k]*(A[:,:,k]-B[:,:,k]*K[:,:,k])' + (A[:,:,k]-B[:,:,k]*K[:,:,k])*H*G[:,:,k]' + G[:,:,k]*H'*(A[:,:,k]-B[:,:,k]*K[:,:,k])' + G[:,:,k]*D*G[:,:,k]';
        H = (A[:,:,k]-B[:,:,k]*K[:,:,k])*H + G[:,:,k]*D;

        dE[:,:,k+1] = dEdE*dE[:,:,k] + dEdH*dH + dEdK*dK[:,:,k];
        dE[:,((k-1)*(1+nX+nU)) .+ (1:1+nX+nU),k+1] = dE[:,((k-1)*(1+nX+nU)) .+ (1:1+nX+nU),k+1] + dEdA*dA[:,:,k] + dEdB*dB[:,:,k] + dEdG*dG[:,:,k];
        #
        dH = dHdH*dH + dHdK*dK[:,:,k];
        dH[:,((k-1)*(1+nX+nU)) .+ (1:1+nX+nU)] = dH[:,((k-1)*(1+nX+nU)) .+ (1:1+nX+nU)] + dHdA*dA[:,:,k] + dHdB*dB[:,:,k] + dHdG*dG[:,:,k];
    end

    c = 0;
    dc = zeros(1,(N-1)*(1+nX+nU)+nX);
    for k = 1:(N-1)

        c += tr((Qr + K[:,:,k]'*Rr*K[:,:,k])*E[:,:,k]);

        dcdE = vec((Qr + K[:,:,k]'*Rr*K[:,:,k]))';
        dcdK = 2*vec(Rr*K[:,:,k]*E[:,:,k])';
        #
        dc = dc + dcdE*dE[:,:,k] + dcdK*dK[:,:,k];
    end

    c = c + tr(Qfr*E[:,:,N]);
    dcdE = vec(Qfr)';
    dc = dc + dcdE*dE[:,:,N];

    return c, vec(dc), dK, vec(K)
end

N = 5
nx,nu,nw = 2,1,1; nX = nx; nU = nu; nW = nw;

dt = 0.1
x0 = [0.;0.]

# allocate trajectories
X = [zeros(nx) for k = 1:N]
U = [ones(nu) for k = 1:N-1]

# cost functions
R = Rr = [0.1]
Q = Qr = [10. 0.; 0. 1.]
Qf = Qfr = [100. 0.; 0. 100.]

# uncertainty
D = [0.2^2]
E1 = zeros(nx,nx)

# discrete dynamics
pendulum_discrete! = forward_euler!(pendulum_dynamics_stochastic!,dt)

_∇f(z,nx) = ForwardDiff.jacobian(f_augmented!(pendulum_discrete!,nx,nu,nw),zeros(eltype(z),nx),z)
function gen_∇f(_∇f,nx)
    ∇f(z) = _∇f(z,nx)
end
∇f = gen_∇f(_∇f,nx)
∇²f(z) = ForwardDiff.jacobian(∇f,z)

# rollout initial state trajectory
X[1] .= x0
for k = 1:N-1
    pendulum_discrete!(X[k+1],dt,X[k],U[k],zeros(nw))
end

Z_aug = pack(X,U,dt)

rcf = gen_rcf(robust_cost_function,pendulum_discrete!,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)

c_fd = rcf(Z_aug)
dc_fd = ForwardDiff.gradient(rcf,Z_aug)

c1, dc1, dK, K = rcb_port(Z_aug)

norm(c_fd-c1)
norm(dc_fd - vec(dc1))

@benchmark ForwardDiff.gradient(rcf,$Z_aug)

###

x_part = NamedTuple{(:x,:δx)}((1:nx,nx .+ (1:nx^2)))
Xr = [BlockArray(zeros(nx+nx^2),x_part) for k = 1:N]

u_part = NamedTuple{(:u,:δu)}((1:nu,nu .+ (1:nu^2)))
Ur = [BlockArray(zeros(nu+nu^2),u_part) for k = 1:N-1]

Xr[1].δx
Ur[1].δu

function set_controls!(Ur,U0)
    N = length(Us)
    for k = 1:N
        Ur[k].u .= U0[k]
    end
end

U0 = [rand(nu) for k = 1:N-1]
set_controls!(Ur,U0)
U0[1]
Ur[1]

function rollout_robust!(Xr,Ur)
    Xr[1].x .= x0
    for k = 1:N-1
        pendulum_discrete!(Xr[k+1].x,dt,Xr[k].x,Ur[k].u,zeros(nw))
    end
end

rollout_robust!(Xr,Ur)

plot(Xr)
