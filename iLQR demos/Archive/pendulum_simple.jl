module Pendulum
using iLQR
using Plots

n = 2; # dimensions of system
p = 1; # dimensions of control
m = 1; # mass
l = 1; # length
g = 9.8; # gravity
mu = 0.01; # friction coefficient

fc(x,u) = [x[2];
        (g/l*sin(x[1]) - mu/m/(l^2)*x[2] + 1/m/(l^2)*u)];

dynamics_midpoint(x,u, dt) = x + fc(x + fc(x,u)*dt/2,u)*dt;

fx(x,dt) = [1 + g*cos(x[1])*(dt^2)/(2*l) dt - mu*(dt^2)/(2*m*l^2);
            g*cos(x[1] + x[2]*dt/2)*dt/l - mu*g*cos(x[1])*(dt^2)/(2*m*l^3) 1 + g*cos(x[1] + x[2]*dt/2)*(dt^2)/(2*l) - mu*dt/(m*l^2) + (mu^2)*(dt^2)/(2*(m^2)*l^4)];

fu(x,dt) = [(dt^2)/(2*m*l^2);
            (-mu*(dt^2)/(2*(m^2)*l^4) + dt/(m*l^2))];


# initial conditions
x0 = [0; 0];

# goal
xf = [pi; 0]; # (ie, swing up)

# costs
Q = 1e-5*eye(n);
Qf = 25*eye(n);
R = 1e-5*eye(p);

e_dJ = 1e-6;

# simulation
dt = 0.1;
tf = 1;
N = Int(floor(tf/dt));
t = linspace(0,tf,N);
iterations = 4

# initialization
u = zeros(p,N-1);
x = zeros(n,N);
x_prev = zeros(n,N);
x[:,1] = x0;

# first roll-out
for k = 2:N
    x[:, k] = dynamics_midpoint(x[:,k-1], u[:,k-1], dt)
end

# original cost
J = 0
for k = 1:N-1
    J = J + (x[:,k] - xf)'*Q*(x[:,k] - xf) + u[:,k]'*R*u[:,k]
end
J = 0.5*(J + (x[:,N] - xf)'*Qf*(x[:,N] - xf))


## iterations of iLQR using my derivation
# improvement criteria
c1 = 0.25;
c2 = 0.75;

for i = 1:iterations
    S = zeros(n,n,N)
    S[:,:,N] = Qf
    s = zeros(n,N)
    s[:,N] = Qf*(x[:,N] - xf)
    K = zeros(p,n,N)
    lk = zeros(p,N)
    vs1 = 0
    vs2 = 0

    mu_reg = 0;
    k = N-1;
    while k >= 1
        q = Q*(x[:,k] - xf);
        r = R*(u[:,k]);
        A = fx(x[:,k],dt);
        B = fu(x[:,k],dt)
        C1 = q' + s[:,k+1]'*A;  # 1 x n
        C2 = r' + s[:,k+1]'*B;  # 1 x m
        C3 = Q + A'*S[:,:,k+1]*A; # n x n
        C4 = R + B'*(S[:,:,k+1] + mu_reg*eye(n))*B; # m x m
        C5 = Array(B'*(S[:,:,k+1] + mu_reg*eye(n))*A);  # m x n

        # regularization
        if any(eigvals(C4).<0)
            mu_reg = mu_reg + 1;
            k = N-1;
        end

        K[:,:,k] = C4\C5';
        lk[:,k] = C4\C2';
        s[:,k] = C1 - C2*K[:,:,k] + lk[:,k]'*C4*K[:,:,k] - lk[:,k]'*C5;
        S[:,:,k] = C3 + K[:,:,k]'*C4*K[:,:,k] - K[:,:,k]'*C5 - C5'*K[:,:,k];

        vs1 = vs1 + AbstractFloat(lk[:,k]'*C2');
        vs2 = vs2 + AbstractFloat(lk[:,k]'*C4*lk[:,k]);
        k = k - 1;

    end

    # update control, roll out new policy, calculate new cost
    u_ = zeros(p, N)
    x_prev = copy(x);
    J_prev = copy(J);
    J = Inf;
    alpha = 1;
    iter = 0;
    dV = Inf;
    z = 0;

    while J > J_prev
        x = zeros(n,N);
        x[:,1] = x0;
        for k = 2:N
            a = alpha*(lk[:,k-1]);
            delta = (x[:,k-1] - x_prev[:,k-1])

            u_[:, k-1] = u[:, k-1] - K[:,:,k-1]*delta - a;
            x[:,k] = dynamics_midpoint(x[:,k-1], u_[:,k-1], dt);
        end

        # Calcuate cost
        J = 0;
        for k = 1:N-1
            J = J + (x[:,k] - xf)'*Q*(x[:,k] - xf) + u_[:,k]'*R*u_[:,k];
        end
        J = 0.5*(J + (x[:,N] - xf)'*Qf*(x[:,N] - xf));

        alpha = alpha/2;
        iter = iter + 1;

        if iter > 200
            println("max iterations")
            break
        end
    end
    u = copy(u_)
    println("Cost:", J)
end

p = plot(linspace(0,tf,N), x[1,:])
p = plot!(linspace(0,tf,N), x[2,:])
# display(p)

end
