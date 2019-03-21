f = Dynamics.dubins_dynamics!
n,m = 3,2
model = Model(f,n,m)
x0 = [0; 0.; 0.];
xf = [1; 0.; 0.]; # (ie, swing up)
u0 = [1.;1.]
Q = 1e-3*Diagonal(I,n)
Qf = 100. *Diagonal(I,n)
R = 1e-3*Diagonal(I,m)
tf = 5.
costfun = LQRCost(Q,R,Qf,xf)
N = 11
dt = 0.1
prob = Problem(model,costfun,x0,rand(m,N-1),N,dt)
prob.X

rollout!(prob)
prob.X
