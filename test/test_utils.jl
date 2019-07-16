using Plots

@test TO.pos(10) == 10
@test TO.pos(-10) == 0
@test TO.pos(-1e-4) == 0


# Conversion between Matrices and Trajectories
n,m,N = 5,3,21
X = rand(n,N)
X_ = [X[:,k] for k = 1:N]
@test to_array(X_) == X
@test TO.to_trajectory(X) == X_
@test to_dvecs(X) == X_
@test vec(X_) == vec(X)

Z = rand(n,m,N)
Z_ = [Z[:,:,k] for k = 1:N]
@test to_array(Z_) == Z
@test to_dvecs(Z) == Z_

D_ = [Diagonal(rand(n)) for k = 1:N]
D = zeros(n,n,N)
for k = 1:N
    D[:,:,k] = D_[k]
end
@test to_array(D_) == D

# Test copy methods
X_ = [rand(n) for k = 1:N]
X2_ = [rand(n) for k = 1:N]
copyto!(X2_, X_)
@test X2_[1] == X_[1]
X_[1][1] = 100
@test X2_[1][1] != 100

X2_
copyto!(X2_, X)
@test X2_[1] == X[:,1]

copyto!(X, X_)
@test X[1,1] == 100

# Vector of different lengths
x,y,z = rand(n), rand(n), rand(m)
W = [x;y;z]
p = [n,n,m]
@test to_dvecs(W,p) == [x,y,z]


# Others
@test isdir(TO.root_dir())


# Plotting
function test_plot()
    plot()
    plt = TO.plot_circle!([1,1,5], 1, color=:black)

    circles = [[-1,1.5,1], [-1,-1,0.5], [1,-1,0.25]]
    TO.plot_obstacles(circles)

    display(plt)

    time = range(0,10,length=N)
    X = [[sin(t); cos(t); rand(3)] for t in time]
    plot_trajectory!(X)
    plot_trajectory!(to_dvecs(X).*2)

    plot_vertical_lines!(plt, [-1,0,1])

    X_ = to_dvecs(X)
    plot(X_)
    plot!(X_, color=:black)
    plot(X_,1:2)
    return nothing
end
test_plot()


# Constraints
@test TO.circle_constraint([0,0,0], 1, 0, 1) == 0
@test TO.circle_constraint([0,0,0], 1, 0, 0.5) < 0
@test TO.circle_constraint([0.75,0,0], 1, 0, 0.5) > 0
@test TO.circle_constraint([0,0,0], [1, 0], 1) == 0
@test TO.circle_constraint([0,0,0], [1, 0], 0.5) < 0
@test TO.circle_constraint([0.75,0,0], [1, 0], 0.5) > 0

@test TO.sphere_constraint([0,0,0], 1, 0, 0, 1) == 0
@test TO.sphere_constraint([0,0,0], 1, 0, 0, 0.5) < 0
@test TO.sphere_constraint([0.75,0,0], 1, 0, 0, 0.5) > 0
@test TO.sphere_constraint([0,0,0], [1, 0, 0], 1) == 0
@test TO.sphere_constraint([0,0,0], [1, 0, 0], 0.5) < 0
@test TO.sphere_constraint([0.75,0,0], [1, 0, 0], 0.5) > 0
