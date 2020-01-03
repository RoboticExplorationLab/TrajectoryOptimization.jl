# Quadratic Costs
n,m = 3,2
Q = Diagonal(@SVector fill(0.1, n))
R = Diagonal(@SVector fill(0.01, m))
Qf = Diagonal(@SVector fill(10., n))
xf = @SVector ones(n)

quadcost = QuadraticCost(Q,R)
@test quadcost.Q == Q
@test quadcost.q == zeros(n)
@test quadcost.R == R
@test quadcost.r == zeros(m)
@test quadcost.c == 0
@test state_dim(quadcost) == n
@test control_dim(quadcost) == m

quadcost = LQRCost(Q,R,xf)
@test quadcost.Q == Q
@test quadcost.q == -Q*xf
@test quadcost.R == R
@test quadcost.r == zeros(m)
@test quadcost.c == 0.5*xf'Q*xf

quadcost = TO.LQRCostTerminal(Q,xf)
@test quadcost.Q == Q
@test quadcost.q == -Q*xf
@test isempty(quadcost.R)
@test isempty(quadcost.r)
@test quadcost.c == 0.5*xf'Q*xf

q = @SVector rand(n)
quadcost = QuadraticCost(Q,q,0.5)
@test quadcost.Q == Q
@test quadcost.q == q
@test isempty(quadcost.R)
@test isempty(quadcost.r)
@test quadcost.c == 0.5

Q2 = Diagonal(@SVector [-1, 0.1, 0.2])
R2 = Diagonal(@SVector [0, 0.1])
@test_throws ArgumentError QuadraticCost(Q2,R)
@test_logs (:warn, "R is not positive definite") QuadraticCost(Q,R2)
