using Ipopt
using ForwardDiff

model = TrajectoryOptimization.Dynamics.pendulum[1]
model! = TrajectoryOptimization.Dynamics.pendulum![1]
obj = TrajectoryOptimization.Dynamics.pendulum[2]
obj = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-2, u_max=2)
R = obj.R; Q = obj.Q; Qf = obj.Qf; xf = obj.xf
f = model.f
n, m = 2,1
N = 50
dt = 0.1

f_aug! = f_augmented!(model!.f, model.n, model.m)
zdot = zeros(2)
F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)
F([1,2,1])
vec(F([1,2,1]))


weights = ones(N)
weights[[1,end]] = 0.5

function packZ(X,U)
    n, N = size(X)
    m = size(U,1)
    Z = zeros(n+m,N)
    Z[1:n,:] .= X
    Z[n+1:end,1:end] .= U
    Z = reshape(Z,1,(n+m)N)
end

function unpackZ(Z)
    Z = reshape(Z,n+m,N)
    X = Z[1:n,:]
    U = Z[n+1:end,:]
    return X,U
end

function cost(X,U)
    J = 0
    for k = 1:N-1
      J += dt/2*(cost_k(X[:,k],U[:,k]) + cost_k(X[:,k+1],U[:,k+1]))
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
end

function cost_k(x,u)
    0.5*(x-xf)'Q*(x-xf) + 0.5*u'R*u
end

function eval_f(Z)
    X,U = unpackZ(Z)
    cost(X,U)
end

function eval_g(Z, g)
    X,U = unpackZ(Z)
    g = reshape(g,n,N-1)
    @show size(g)
    for k = 1:N-1
        g[:,k] = ( f(X[:,k+1],U[:,k+1]) + f(X[:,k],U[:,k]) )/2 - X[:,k+1] + X[:,k]
    end
    # reshape(g,n*(N-1),1)
    return nothing
end

function eval_grad_f(Z, grad_f)
    X, Z = unpackZ(Z)
    grad_f = reshape(grad_f, n+m, N)
    for k = 1:N
        grad_f[1:n,k] = weights[k]*Q*X[:,k]
        grad_f[n+1:end,k] = weights[k]*R*U[:,k]
    end
    grad_f[1:n,N] += Qf*X[:,N]
    return nothing
end

function eval_jac_g(x, mode, rows, cols, vals)
    z = reshape(x,n+m,N)
    if mode == :Structure
        ind = 1
        for k = 1:N-1 # loop over blocks
            for j = 1:2(n+m) # loop over columns of each block
                for i = 1:n # loop over rows of each block
                    row[ind] = i + (k-1)*n
                    col[ind] = j + (k-1)*(n+m)
                    ind += 1
                end
            end
        end
    else
        n_blk = n*2(n+m)
        n_blk2 = n*(n+m)
        fz = F(z[:,1])
        vals[1:n_blk2] .= vec(fz+eye(n,n+m))
        for k = 2:N-1
            fz = F(z[:,k])
            vals[(1:n_blk2) + (k-1)*n_blk - n_blk2] .= vec(fz+eye(n,n+m))
            vals[(1:n_blk2) + (k-1)*n_blk] .= vec(fz-eye(n,n+m))
        end
        fz = F(z[:,N])
        vals[end-n_blk2+1:end] .= vec(fz-eye(n,n+m))
    end
end

function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
    if mode == :Structure
        ind = 1
        for k = 1:N
            n_blk = (n^2 + m^2)
            for j = 1:n
                for i = 1:n
                    row[ind] = i + (k-1)*n_blk
                    col[ind] = j + (k-1)*n_blk
                    ind += 1
                end
            end
            for j = 1:m
                for i = 1:m
                    row[ind] = i + (k-1)*n_blk + n^2
                    col[ind] = j + (k-1)*n_blk + n^2
                    ind += 1
                end
            end
        end
    end
end

X = rand(n,N)
U = rand(m,N)
Z = packZ(X,U)

X2,U2 = unpackZ(Z)
g = zeros(n*(N-1))
eval_g(Z,g)
g
grad_f = zeros((n+m)*N)
eval_grad_f(Z,grad_f)


nnz_jac_g = (N-1)*n*2(n+m)
row = zeros(Int,nnz_jac_g)
col = zeros(Int,nnz_jac_g)
val = zeros(nnz_jac_g)
eval_jac_g(Z,:Structure, row, col, val)

jac_g0 = zeros(Int,(N-1)*n,N*(n+m))
inds = sub2ind(jac_g0,row,col)
jac_g0[inds] .= 1:nnz_jac_g
jac_g0

eval_jac_g(Z,:None,row,col,val)
val

nnz_h = (n^2 + m^2)*N
h = zeros(N*(n+m), N*(n+m))
row = zeros(Int,nnz_h)
col = zeros(Int,nnz_h)
vals = zeros(nnz_h)
lambda = zeros(n, N-1)
eval_h(Z, :Structure, row, col, 1, lambda, vals)

row

col
inds = CartesianIndex.(row[1:10],col[1:10])
h[inds] .= 1:10
h
