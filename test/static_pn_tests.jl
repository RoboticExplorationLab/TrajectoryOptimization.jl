using StaticArrays, LinearAlgebra, BenchmarkTools, ForwardDiff, SparseArrays

n,m,N = 3,2,11
dt = 0.1
NN = n*N + m*(N-1)
xs = @SVector rand(n)
us = @SVector rand(m)
Hs = @SMatrix rand(n,n)
Xs = [xs for k = 1:N]
Us = [us for k = 1:N-1]
Zs = [KnotPoint(xs,us,dt) for k = 1:N]
Zs[end] = KnotPoint(xs,m)

Z = zeros(NN)
xinds = [SVector{n}((n+m)*(k-1) .+ (1:n)) for k = 1:N]
uinds = [SVector{m}(n + (n+m)*(k-1) .+ (1:m)) for k = 1:N-1]

P = StaticPrimals(n,m,N)
copyto!(P,Zs)
@btime copyto!($P,$Zs)
copyto!(Zs,P)
@btime copyto!($Zs,$P)


# Constraints
xf = @SVector [5.,5,0]

x_min = @SVector [-10,-10,-Inf]
x_max = @SVector [10,10,Inf]
u_min = @SVector [-2.,0]
u_max = @SVector [2.,3]
bnd = StaticBoundConstraint(n,m, x_min=x_min, x_max=x_max,
    u_min=u_min, u_max=u_max)
bnd_con = KnotConstraint(bnd, 1:N-1)
goal = GoalConstraint(xf)
goal_con = KnotConstraint(goal, N:N)
cons = [bnd_con, goal_con]
conSet = ConstraintSets(cons, N)

sum(conSet.p) + n*N
sum(length.(dyn))
sum([sum(length.(con)) for con in cons])
dyn,cons = gen_con_inds(n,m,conSet)
cons isa Vector{Vector{T} where T}

function con_labels(dyn,cons)
    NP = sum(length.(dyn)) + sum([sum(length.(con)) for con in cons])
    labels = Vector{String}(undef,NP)
    for d in dyn
        labels[d] .= "dynamics"
    end
    for i in eachindex(cons)
        for c in cons[i]
            labels[c] .= "constraint $i"
        end
    end
    labels[dyn[1]] .= "initial condition"
    return labels

end
con_labels(dyn,cons)
