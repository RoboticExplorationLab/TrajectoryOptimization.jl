
% Taylor Howell
% Research-Manchester
% June 21, 2018
% iLQR for control of a pendulum (simple) 
    % Midpoint discretization
    % NOTE: the exactr implementation in the UDP paper is incorrect.
    % Implemented is the corrected algorithm
clear; clc; close all

% system 
n = 2; % dimensions of system
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
Q = 1e-5*eye(2);
Qf = 25*eye(2);
R = 1e-5;

e_dJ = 1e-5;

% simulation
dt = 0.01;
tf = 1;
N = floor(tf/dt);
t = linspace(0,tf,N);
iterations = 100;

% initialization
u = zeros(1,N-1);
x = zeros(n,N);
x_prev = zeros(n,N);
x(:,1) = x0;

% first roll-out
for k = 2:N
    x(:,k) = dynamics_midpoint(x(:,k-1),u(k-1),dt);
end

% original cost
J = 0;
for k = 1:N-1
    J = J + (x(:,k) - xf)'*Q*(x(:,k) - xf) + u(k)'*R*u(k);
end
disp('Original cost:')
J = 0.5*(J + (x(:,N) - xf)'*Qf*(x(:,N) - xf))

%% iterations of iLQR 
for i = 1:iterations
    u_ = zeros(1,N-1);
    
    %% Backward pass (DDP derivation):
    H = zeros(n,n,N);
    g = zeros(n,N);
    H(:,:,N) = Qf;
    g(:,N) = Qf*(x(:,N) - xf);
    K = zeros(1,n,N-1);
    lk = zeros(1,N-1);
    r = 0;
    
    for k = N-1:-1:1
        q = Q*(x(:,k) - xf);

        a = Q*x(:,k) + q + fx(x(:,k),dt)'*g(:,k+1);
        b = R*u(k) + r + fu(x(:,k),dt)'*g(:,k+1);
        A = Q + fx(x(:,k),dt)'*H(:,:,k+1)*fx(x(:,k),dt);
        B = R + fu(x(:,k),dt)'*H(:,:,k+1)*fu(x(:,k),dt);
        while B <= 0
            disp('B is not PD')
            B = B + 10;
        end
        C = fu(x(:,k),dt)'*H(:,:,k+1)*fx(x(:,k),dt);
        K(1,:,k) = inv(B)*C;
        lk(k) = inv(B)*b;
        
        H(:,:,k) = A + K(1,:,k)'*B*K(1,:,k) - K(1,:,k)'*C - C'*K(1,:,k);
        g(:,k) = a + (K(1,:,k)'*B - C')*lk(k) - K(1,:,k)'*b;
        
        dV(k) = -lk(k)'*B*lk(k) - b'*lk(k);  
    end
    % update control, roll out new policy, calculate new cost
    x_prev = x;
    J_prev = J;
    J = Inf;
    alpha = 1;
    max_iter = 1000;
    iter = 0;
    while J > J_prev 

        x = zeros(n,N);
        x(:,1) = x0;
        for k = 2:N
            u_(k-1) = u(k-1) - alpha*lk(k-1) - K(1,:,k-1)*(x(:,k-1) - x_prev(:,k-1));
            x(:,k) = dynamics_midpoint(x(:,k-1),u_(k-1),dt);
        end

        J = 0;
        for k = 1:N-1
            J = J + (x(:,k) - xf)'*Q*(x(:,k) - xf) + u_(k)'*R*u_(k);
        end
        J = 0.5*(J + (x(:,N) - xf)'*Qf*(x(:,N) - xf));
        alpha = alpha/2;
        iter = iter + 1;
        x_prev = x;
    end
    disp('New cost:')
    J
    
%     if J > J_prev
%         pause()
%     end
    u = u_;
    
    if abs(J - J_prev) < e_dJ
        disp(strcat('eps criteria met at iteration: ',num2str(i)))
        break
    end
end


%% Results

% Animation
% r = 1;
% figure
% U = [0 u];
% for i = 1:N
%     p1 = subplot(1,2,1);
%     X = r*cos(x(1,i) - pi/2);
%     Y = r*sin(x(1,i) - pi/2);
%     plot([0 X],[0 Y],'k-')
%     hold on
%     plot(X,Y,'ko','MarkerFaceColor', 'k')
%     xlabel('pendulum (simple)')
%     axis([-1.5*r 1.5*r -1.5*r 1.5*r])
%     axis square
%     
%     p2 = subplot(1,2,2);
%     stairs(t(1:i),U(1:i))
%     xlabel('t')
%     ylabel('u(t)')
%     axis([0 tf min(u) max(u)])
%     axis square
% 
%     drawnow
%     %pause(dt)
% 
%     if i ~= N
%         cla(p1);
%         cla(p2);
%     end
%      
% end

figure
hold on
plot(linspace(0,tf,N),x(1,:))
plot(linspace(0,tf,N),x(2,:))
legend('\theta','\theta_{d}')
