% Taylor Howell
% Research-Manchester
% June 19, 2018
% Comparing Euler and Midpoint discretization methods for a free-falling pendulum
% (simple) 

clear; clc; close all

% system parameters
m = 1;
l = 1;
g = 9.8;
mu = 0.01;

% initial conditions
theta0 = pi/2;
thetad0 = 0;

% control
u = 0;

% simulation
tf = 100;
dt = 0.01;

%% Simulink simulation
data = sim('pendulum_simple','StartTime','0','StopTime', num2str(tf),'Solver','ode4','FixedStep',num2str(dt));

%% Euler discretization and Midpoint Discretization
fc = @(x,u) [x(2);
            (g/l*sin(x(1)) - mu/m/(l^2)*x(2) + 1/m/(l^2)*u)];
        
dynamics_euler = @(x,u,dt) x + fc(x,u)*dt;
dynamics_midpoint = @(x,u,dt) x + fc(x + fc(x,u)*dt/2,u)*dt;
dynamics_midpoint_expanded = @(x,u,dt) [x(1) + x(2)*dt + g*sin(x(1))*(dt^2)/2/l - mu*x(2)*(dt^2)/2/(m*l^2) + u*(dt^2)/2/(m*l^2);
                                        x(2) + g*sin(x(1) + x(2)*dt/2)*dt/l - mu*x(2)*dt/(m*l^2) - mu*g*sin(x(1))*(dt^2)/2/(m*l^3) + (mu^2)*x(2)*(dt^2)/2/((m^2)*(l^4)) - mu*u*(dt^2)/2/((m^2)*(l^4)) + u*dt/(m*l^2)];
N = floor(tf/dt);
X_euler = zeros(2,N);
X_euler(:,1) = [theta0; thetad0];
X_midpoint = zeros(2,N);
X_midpoint(:,1) = [theta0; thetad0];
X_midpoint_expanded = zeros(2,N);
X_midpoint_expanded(:,1) = [theta0; thetad0];
T = zeros(1,N);
for k = 2:N
    T(k) = T(k-1) + dt;
    X_euler(:,k) = dynamics_euler(X_euler(:,k-1),0,dt);
    X_midpoint(:,k) = dynamics_midpoint(X_midpoint(:,k-1),0,dt);
    X_midpoint_expanded(:,k) = dynamics_midpoint_expanded(X_midpoint_expanded(:,k-1),0,dt); 
end

%% Plot
hold on
plot(data.get('theta').Time,data.get('theta').Data,'b')
plot(T,X_euler(1,:),'r')
plot(T,X_midpoint(1,:),'g')
%plot(T,X_midpoint_expanded(1,:),'k')
xlabel('Time (s)')
ylabel('\theta')
axis([0 tf -2*pi 2*pi])
legend('Simulink ode4','Euler','Midpoint')%,'Midpoint Exp.')

%% Animation
% r = 1;
% figure
% for i = 1:N
%     p1 = subplot(1,2,1);
%     x = r*sin(X_euler(1,i));
%     y = r*cos(X_euler(1,i));
%     plot([0 x],[0 y],'r-')
%     hold on
%     plot(x,y,'ko','MarkerFaceColor', 'k')
%     xlabel('Euler')
%     axis([-1.5*r 1.5*r -1.5*r 1.5*r])
%     axis square
%     
%     p2 = subplot(1,2,2);
%     x = r*sin(X_midpoint(1,i));
%     y = r*cos(X_midpoint(1,i));
%     plot([0 x],[0 y],'g-')
%     hold on
%     plot(x,y,'ko','MarkerFaceColor', 'k')
%     xlabel('Midpoint')
%     axis([-1.5*r 1.5*r -1.5*r 1.5*r])
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

%% Compare expanded midpoint to function midpoint
% figure
% hold on
% plot(T,X_midpoint(1,:),'g')
% plot(T,X_midpoint_expanded(1,:),'k')
% xlabel('Time (s)')
% ylabel('\theta')

disp(sum(X_midpoint(1,:)-X_midpoint_expanded(1,:)))

%% Linearization
% Euler
A_euler = @(x,dt) [1 dt;
                   g*dt*cos(x(1))/l (1 - mu*dt/(m*l^2))];   
           
B_euler = @(x,dt) [0;
                   dt/(m*l^2)];
               
A_midpoint = @(x,dt) [(1 + g*cos(x(1))*(dt^2)/(2*l)) (dt - mu*(dt^2)/(2*m*l^2));
                      (g*cos(x(1) + x(2)*dt/2)*dt/l - mu*g*cos(x(1))*(dt^2)/(2*m*l^3)) (1 + g*cos(x(1) + x(2)*dt/2)*(dt^2)/(2*l) - mu*dt/(m*l^2) + (mu^2)*(dt^2)/(2*(m^2)*l^4))];

B_midpoint = @(x,dt) [(dt^2)/(2*m*l^2); 
                      (-mu*(dt^2)/(2*(m^2)*l^4) + dt/(m*l^2))];
