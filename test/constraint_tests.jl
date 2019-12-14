using Test

function alloc_con(con,z)
    allocs  = @allocated evaluate(con,z)
    allocs += @allocated jacobian(con,z)
end

model = Dynamics.DubinsCar()
x,u = rand(model)
z = KnotPoint(x,u,0.1)
n,m = size(model)

# All static
A = @SMatrix rand(4,3)
b = @SVector rand(4)
con = LinearConstraint{Inequality,State}(n,m,A,b)
evaluate(con, z)
@test evaluate(con, z) == A*x - b
@test jacobian(con, x) == A
@test alloc_con(con,z) == 0

# both dynamic
con = LinearConstraint{Inequality,State}(n,m,Matrix(A),Vector(b))
@test evaluate(con, z) == A*x - b
@test jacobian(con, x) == A
@test alloc_con(con,z) == 0

# mixed
con = LinearConstraint{Inequality,State}(n,m,A,Vector(b))
@test evaluate(con, z) == A*x - b
@test jacobian(con, x) == A
@test alloc_con(con,z) == 0

# wrong input size
@test_throws AssertionError LinearConstraint{Inequality,State}(m,m,A,b)
@test_throws AssertionError LinearConstraint{Inequality,Control}(n,m,A,b)
con = LinearConstraint{Inequality,Control}(n,n,A,b)
@test evaluate(con, z) == A*x - b
@test jacobian(con, z) == A
@test alloc_con(con,z) == 0
