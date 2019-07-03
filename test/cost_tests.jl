const TO = TrajectoryOptimization

# Get the model
model = Dynamics.car_model
n,m = model.n, model.m
costfun = Dynamics.car_costfun

# Test Expansion
Q = TO.Expansion{Float64}(n,m)
Q.x .= ones(n)
Q.xx .= ones(n,n)
2*Q
@test Q.x == ones(n)*2
Q/2
@test Q.xx == ones(n,n)

Q2 = copy(Q)
Q2.uu .+= 1
@test Q2.uu == ones(m,m)
@test Q.uu == zeros(m,m)
TO.reset!(Q2)
@test Q2.uu == zeros(m,m)

# Quadratic Cost
Q = Diagonal(1.0I,n)
R = Diagonal(1.0I,m)
H = zeros(m,n)
q = ones(n)
r = zeros(m)
Qf = Diagonal(I,n)*100.
qf = zeros(n)
quadcost = QuadraticCost(Q, R, H, q, r, 0., Qf, qf, 0.)
@test quadcost.c == 0
@test quadcost.Q == Q
@test quadcost.R isa Diagonal

quadcost = QuadraticCost(Q,R)
@test quadcost.R isa Diagonal
@test quadcost.Qf == zeros(n,n)
@test quadcost.Qf isa Diagonal
@test quadcost.H == zeros(m,n)

# LQR Cost
xf = [0,1,0]
lqrcost = LQRCost(Q,R,Qf,xf)
@test lqrcost.Q == Q
@test lqrcost.q == -Q*xf

cost_term = TO.LQRCostTerminal(Qf, xf)
@test cost_term.Q == zeros(0,0)
@test cost_term.Qf == Qf

# Test costs
x,u = rand(n), rand(m)
dt = rand(1)[1]
@test TO.stage_cost(quadcost,x,u,dt) == 0.5*(x'Q*x + u'R*u)*dt
@test TO.stage_cost(quadcost,x) == 0
@test TO.stage_cost(cost_term,x) ≈ 0.5*(x-xf)'Qf*(x-xf)


# Test expansion functions
E = TO.Expansion{Float64}(n,m)
TO.cost_expansion!(E, quadcost, x, u, dt)
@test E.x == Q*x*dt
@test E.xx == Q*dt
@test E.ux == zeros(m,n)*dt
@test u == R*u

TO.reset!(E)
TO.cost_expansion!(E, cost_term, x)
@test E.x ≈ Qf*(x-xf)
@test E.u == zeros(m)
@test E.xx == Qf

grad = PartedVector(zeros(n+m), create_partition((n,m),(:x,:u)))
TO.gradient!(grad, quadcost, x, u, dt)
@test grad.x == dt*Q*x
@test grad.u == dt*R*u

grad = zeros(n)
TO.gradient!(grad, cost_term, x)
@test grad ≈ Qf*(x-xf)

quadcost2 = copy(quadcost)
quadcost2.Q .+= Diagonal(I,n)
@test quadcost2.Q == Diagonal(2I,n)
@test quadcost.Q == Q


# Generic Costs
n,m = 2,1
x,u = rand(n), rand(m)
function mycost(x,u)
    R = Diagonal(0.1I,1)
    Q = 0.1
    return cos(x[1]) + u'R*u + Q*x[2]^2
end
function mycost(xN)
    return cos(xN[1]) + xN[2]^2
end
nlcost = GenericCost(mycost,mycost,n,m)
@test TO.stage_cost(nlcost,x,u) == mycost(x,u)
@test TO.stage_cost(nlcost,x) == mycost(x)

R = Diagonal(0.1I,m)
Q = 0.1
function hess(x,u)
    n,m = length(x),length(u)
    Qexp = Diagonal([-cos(x[1]), 2Q])
    Rexp = 2R
    Hexp = zeros(m,n)
    return Qexp,Rexp,Hexp
end
function hess(x)
    return Diagonal([-cos(x[1]), 2])
end
function gradient(x,u)
    q = [-sin(x[1]), 2Q*x[2]]
    r = 2R*u
    return q,r
end
function gradient(x)
    return [-sin(x[1]), 2*x[2]]
end
nlcost2 = GenericCost(mycost, mycost, gradient, hess, n, m)

E = TO.Expansion{Float64}(n,m)
TO.cost_expansion!(E, nlcost, x, u)
@test E.x == gradient(x,u)[1]
@test E.xx == hess(x,u)[1]
@test E.ux == hess(x,u)[3]
TO.cost_expansion!(E, nlcost2, x, u)
@test E.x == gradient(x,u)[1]
@test E.xx == hess(x,u)[1]
@test E.ux == hess(x,u)[3]

nlcost3 = copy(nlcost)
@test TO.stage_cost(nlcost3,x,u) == TO.stage_cost(nlcost,x,u)
