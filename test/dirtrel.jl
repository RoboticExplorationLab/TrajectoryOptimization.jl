function pendulum_dynamics_uncertain(ẋ,x,u,w)
    ẋ = zero(x)
    m = 1. + w[1]
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    ẋ[1] = x[2]
    ẋ[2] = u[1]/(m*lc*lc) - g*sin(x[1])/lc - b*x[2]/(m*lc*lc)
    return nothing
end

eig_thr = 1.0e-3
n = 2; m = 1; r = 1# states, controls

u_max = 3.
u_min = -3.

h_max = Inf
h_min = 0.0

# Problem
x0 = [0.;0.]
xf = [pi;0.]

E0 = Diagonal(1.0e-6*ones(n))
H0 = zeros(n,r)
D = Diagonal([.2^2])

N = 51 # knot points
NN = (n+m+1)*(N-1) + n # number of decision variables

p = n*N + n + (N-2)# number of equality constraints: dynamics, xf, h_k = h_{k+1}

# p += 2*n^2 # robust terminal constraint n^2
p_ineq = 2*(m^2)*(N-1)*2

Q = Diagonal(zeros(n))
R = Diagonal(zeros(m))
Qf = Diagonal(zeros(n))

Q_lqr = Diagonal([10.;1.])
R_lqr = Diagonal(0.1*ones(m))
Qf_lqr = Diagonal([100.;100.])

Q_r = Q_lqr
R_r = R_lqr
Qf_r = Qf_lqr

tf0 = 2.
dt = tf0/(N-1)

function line_trajectory(x0::Vector,xf::Vector,N::Int)
    t = range(0,stop=N,length=N)
    slope = (xf .- x0)./N
    x_traj = [slope*t[k] for k = 1:N]
    x_traj[1] = x0
    x_traj[end] = xf
    x_traj
end


X = line_trajectory(x0,xf,N)
U = [0.01*rand(m) for k = 1:N-1]
H = [dt*ones(1) for k = 1:N-1]
