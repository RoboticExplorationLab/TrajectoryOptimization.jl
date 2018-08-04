% Taylor Howell
% Research-Manchester
% June 25, 2018
% iLQR for control of a pendulum (simple) 
    % Midpoint discretization
    % from my derivation - Note that it is slightly different than the
    % iLQR paper but converges to the same cost with less iterations
    
    % regularization and line search are incorporated using Yuval thesis
    % improvements

clear; clc; close all

% system 
n = 2; % dimensions of system
p = 1; % dimensions of control
m = 1; % mass
l = 1; % length
g = 9.8; % gravity
mu = 0.01; % friction coefficient

fc = @(x,u) [x(2);
            (g/l*sin(x(1)) - mu/m/(l^2)*x(2) + 1/m/(l^2)*u)];
        
dynamics_midpoint = @(x,u,dt) x + fc(x + fc(x,u)*dt/2,u)*dt;

fx = @(x,dt) [(1 + g*cos(x(1))*(dt^2)/(2*l)) (dt - mu*(dt^2)/(2*m*l^2));
                      (g*cos(x(1) + x(2)*dt/2)*dt/l - mu*g*cos(x(1))*(dt^2)/(2*m*l^3)) (1 + g*cos(x(1) + x(2)*dt/2)*(dt^2)/(2*l) - mu*dt/(m*l^2) + (mu^2)*(dt^2)/(2*(m^2)*l^4))];

fu = @(x,dt) [(dt^2)/(2*m*l^2); 
                      (-mu*(dt^2)/(2*(m^2)*l^4) + dt/(m*l^2))];

% initial conditions
x0 = [0; 0];

% goal
xf = [pi; 0]; % (ie, swing up)

% costs
Q = 1e-5*eye(n);
Qf = 25*eye(n);
R = 1e-5*eye(p);

e_dJ = 1e-12;

% simulation
dt = 0.01;
tf = 1;
N = floor(tf/dt);
t = linspace(0,tf,N);
iterations = 100;

% initialization
u = zeros(p,N-1);
x = zeros(n,N);
x_prev = zeros(n,N);
x(:,1) = x0;

tic
% first roll-out
for k = 2:N
    x(:,k) = dynamics_midpoint(x(:,k-1),u(:,k-1),dt);
end

% original cost
J = 0;
for k = 1:N-1
    J = J + (x(:,k) - xf)'*Q*(x(:,k) - xf) + u(:,k)'*R*u(:,k);
end
disp('Original cost:')
J = 0.5*(J + (x(:,N) - xf)'*Qf*(x(:,N) - xf))

%% iterations of iLQR using my derivation
% improvement criteria
c1 = 0.25;
c2 = 0.75;

for i = 1:iterations
    S = zeros(n,n,N);
    S(:,:,N) = Qf;
    s = zeros(n,N);
    s(:,N) = Qf*(x(:,N) - xf);
    K = zeros(p,n,N);
    lk = zeros(1,N);
    vs1 = 0;
    vs2 = 0;
    
    mu_reg = 0;
    k = N-1;
    while k >= 1
        q = Q*(x(:,k) - xf);
        r = R*(u(k));
        A = fx(x(:,k),dt);
        B = fu(x(:,k),dt);
        C1 = q' + s(:,k+1)'*A ;
        C2 = r' + s(:,k+1)'*B;
        C3 = Q + A'*S(:,:,k+1)*A;
        C4 = R + B'*(S(:,:,k+1) + mu_reg*eye(n))*B;
        C5 = B'*(S(:,:,k+1) + mu_reg*eye(n))*A;
        
        % regularization
        if any(eig(C4)<0)
            mu_reg = mu_reg + 1;
            k = N-1;
        end
        
        K(:,:,k) = C4\C5';
        lk(k) = C4\C2';
        s(:,k) = C1 - C2*K(:,:,k) + lk(k)'*C4*K(:,:,k) - lk(k)'*C5;
        S(:,:,k) = C3 + K(:,:,k)'*C4*K(:,:,k) - K(:,:,k)'*C5 - C5'*K(:,:,k);
        
        vs1 = vs1 + lk(k)'*C2';
        vs2 = vs2 + lk(k)'*C4*lk(k);
        k = k - 1;
    end
    
    % update control, roll out new policy, calculate new cost
    x_prev = x;
    J_prev = J;
    J = Inf;
    alpha = 1;
    iter = 0;
    dV = Inf;
    z = 0;
    while J > J_prev || z < c1 || z > c2
        x = zeros(n,N);
        x(:,1) = x0;
        dV = 0;
        for k = 2:N
            u_(k-1) = u(k-1) -K(1,:,k-1)*(x(:,k-1) - x_prev(:,k-1)) + alpha*(-lk(k-1));
            x(:,k) = dynamics_midpoint(x(:,k-1),u_(k-1),dt);
        end
        
        J = 0;
        for k = 1:N-1
            J = J + (x(:,k) - xf)'*Q*(x(:,k) - xf) + u_(k)'*R*u_(k);
        end
        J = 0.5*(J + (x(:,N) - xf)'*Qf*(x(:,N) - xf));
        
        dV = (alpha*vs1 + (alpha^2)/2*vs2);
        z = (J_prev - J)/dV;
        
        alpha = alpha/2;
        iter = iter + 1;
    end
    disp('New cost:')
    J
    disp('Expected improvement:')
    vs1 + 0.5*vs2
    disp('z criteria')
    z
    disp('Actual improvement:')
    J - J_prev
    u = u_;
    
    if abs(J - J_prev) < e_dJ
        disp(strcat('eps criteria met at iteration: ',num2str(i)))
        break
    end
end

toc
%% Results

% Animation
r = 1;
figure
U = [u(1) u];
for i = 1:N
    p1 = subplot(1,2,1);
    X = r*cos(x(1,i) - pi/2);
    Y = r*sin(x(1,i) - pi/2);
    plot([0 X],[0 Y],'k-')
    hold on
    plot(X,Y,'ko','MarkerFaceColor', 'k')
    xlabel('pendulum (simple)')
    axis([-1.5*r 1.5*r -1.5*r 1.5*r])
    axis square
    
    p2 = subplot(1,2,2);
    stairs(t(1:i),U(1:i))
    xlabel('t')
    ylabel('u(t)')
    axis([0 tf min(u) max(u)])
    axis square

    drawnow
    %pause(dt)

    if i ~= N
        cla(p1);
        cla(p2);
    end
     
end

figure
hold on
plot(linspace(0,tf,N),x(1,:))
plot(linspace(0,tf,N),x(2,:))
legend('\theta','\theta_{d}')
