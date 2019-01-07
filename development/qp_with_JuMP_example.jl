using JuMP, Ipopt, LinearAlgebra

## Example from Nocedal 16.2 (p 452-453)
x̄ = [2;-1;1]
λ = [3;-2]

# set up model
m = Model(solver=IpoptSolver(print_level=0))

# set up primal variables
n = 3
@variable(m, x[1:n])

# set up constraints
G = [6. 2. 1.; 2. 5. 2.; 1. 2. 4.]
c = [-8.; -3.; -3]
A = [1. 0. 1.; 0. 1. 1.]
b = [3.; 0.]

@objective(m, Min, 0.5*x'*G*x + x'*c)

@constraint(m, con, A*x .== b)

print(m)

status = solve(m)

# Solution
println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
println("x = ", getdual(con))
