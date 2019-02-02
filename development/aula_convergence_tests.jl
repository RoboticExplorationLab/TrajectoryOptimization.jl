using LinearAlgebra
using Plots

using JuMP, Ipopt

# large random qp test
n = 20
m = 10
p = 5
P = rand(n,n)
P = P'*P
@show isposdef(P)
q = rand(n)
r = rand(1)[1]

A = rand(m,n)
b = rand(m)
C = rand(p,n)
d = rand(p)

# Validate model with JuMP
model = JuMP.Model(solver=IpoptSolver(print_level=0))

@variable(model, x[1:n])
@objective(model, Min, 0.5*x'*P*x + x'*q + r)
@constraint(model, con, A*x .>= b)
@constraint(model, con2, C*x .== d)
print(model)
status = JuMP.solve(model)
println("Objective value: ", getobjectivevalue(model))
println("x = ", getvalue(x))
println("λ = ", getdual(con),getdual(con2))


# Augmented Lagrangian
# n = 2
# P = [2.0 0.0; 0.0 2.0]
# q = [-2.0; -5.0]
# r = 7.25
#
# A = [1.0 -2.0;
#      -1.0 -2.0;
#      -1.0 2.0;
#      1.0 0.0;
#      0.0 1.0]
# b = [-2.0; -6.0; -2.0; 0.0; 0.0]


# n = 3
# P = [6. 2. 1.; 2. 5. 2.; 1. 2. 4.]
# q = [-8.; -3.; -3]
# r = 0
# A = [1. 0. 1.; 0. 1. 1.]
# b = [3.; 0.]

# # Validate model with JuMP
# model = JuMP.Model(solver=IpoptSolver(print_level=0))
#
# @variable(model, x[1:n])
# @objective(model, Min, 0.5*x'*P*x + x'*q + r)
# @constraint(model, con, A*x .>= b)
# # @constraint(model, con2, C*x .== d )
# print(model)
# status = JuMP.solve(model)
# println("Objective value: ", getobjectivevalue(model))
# println("x = ", getvalue(x))
# println("λ = ", getdual(con))#,getdual(con2))

##
b = [b;d]
A = [A;C]

f(x) = 0.5*x'*P*x + q'*x + r
∇f(x) = P*x + q
∇²f(x) = P

L(x,λ,Iμ) = 0.5*x'*P*x + q'*x + r + λ'*(b - A*x) + 0.5*(b - A*x)'*Iμ*(b - A*x)
∇L(x,λ,Iμ) = P*x + q + (-A)'*(λ +Iμ*(b - A*x))
∇²L(x,λ,Iμ) = P + (-A)'*Iμ*(-A)

g(x) = b - A*x
∇g(x) = -A

pI = m
pE = p
p = pI + pE

mutable struct Results
    x::Vector
    c::Vector
    c_prev::Vector
    λ::Vector
    λ_prev::Vector
    μ::Vector
    Iμ::Matrix
    H::Matrix

    val::Array
    grad::Array

    function Results(n::Int,p::Int)
        new(zeros(n),
        zeros(p),zeros(p),
        zeros(p),zeros(p),
        ones(p),
        1.0*Matrix(I,p,p),
        1.0*Matrix(I,p,p),
        Inf*ones(1),
        Inf*ones(1))
    end
end

struct Objective
    n::Int
    p::Int
    pI::Int
    f::Function
    P::Matrix
    q::Vector
    ∇f::Function
    ∇²f::Function
    L::Function
    ∇L::Function
    ∇²L::Function
    g::Function
    A::Matrix
    b::Vector
    ∇g::Function
end

mutable struct Opts
    iterations_outer::Int
    ϵ_gradient_norm::Float64
    ϵ_constraint_max::Float64
    γ::Float64
    τ::Float64
    bfgs_threshold::Float64
end

function solve_AuLa(x0,results,model,opts,p_update=:default,d_update=:first)
    x = results.x
    c = results.c
    c_prev = results.c_prev
    λ = results.λ
    λ_prev = results.λ_prev
    μ = results.μ
    Iμ = results.Iμ
    H = results.H
    val = results.val
    grad = results.grad

    x[:] = copy(x0)
    c[:] = mdl.g(x)
    Iμ_update!(results,mdl)
    c_prev[:] = c

    val[1] = mdl.L(x,λ,Iμ)[1]
    grad[1] = norm(mdl.∇L(x,λ,Iμ))

    val_hist = []
    grad_hist = []
    c_max_hist = []
    λ_hist = []
    c_max = 0
    outer_iter = 0
    for i = 1:opts.iterations_outer
        iter = 0
        while iter < 10 && norm(mdl.∇L(x,λ,Iμ)) > opts.ϵ_gradient_norm
            x[:] = -(mdl.P + mdl.A'*Iμ*A)\(mdl.q - mdl.A'*λ - mdl.A'*Iμ*mdl.b)
            c[:] = mdl.g(x)
            Iμ_update!(results,mdl)
            val[1] = mdl.L(x,λ,Iμ)
            grad[1] = norm(mdl.∇L(x,λ,Iμ))

            c_max = max_violation(results,model)
            push!(val_hist,val[1])
            push!(grad_hist,norm(grad))
            push!(c_max_hist,c_max)

            iter += 1
        end
        outer_iter = i
        if norm(grad) < opts.ϵ_gradient_norm && c_max < opts.ϵ_constraint_max
            println("*Solved*")
            break
        end

        if i == opts.iterations_outer
            println("--Solve Failed--")
        end

        outer_loop_update!(results,mdl,opts,p_update,d_update)
        Iμ_update!(results,mdl)
        push!(λ_hist,λ)
    end
    @show x
    @show c
    @show λ
    @show μ
    @show λ_prev
    @show c_prev
    @show L(x,λ,Iμ)
    @show f(x)
    return val_hist, grad_hist, c_max_hist, λ_hist, outer_iter
end

function Iμ_update!(results,model)
    for j = 1:model.p
        if ((model.g(results.x)[j] > 0.0 || results.λ[j] > 0.0) && j <= model.pI) || j > model.pI
            results.Iμ[j,j] = results.μ[j]
        else
            results.Iμ[j,j] = 0.0
        end
    end
end

function max_violation(results,model)
    maximum(map(x->x.>0, results.Iμ)*model.g(results.x))
end

function outer_loop_update!(results,model,opts,p_update=:default,d_update=:first)
    if p_update == :default
        if d_update == :buys
            active_set = zeros(Bool,p)
            for i = 1:p
                if results.Iμ[i,i] > 0.
                    active_set[i] = true
                end
            end
            results.λ[active_set] += (model.∇g(results.x)[active_set,:]*(model.∇²L(results.x,results.λ,results.Iμ)\model.∇g(results.x)[active_set,:]'))\model.g(results.x)[active_set]
            results.λ[1:model.pI] = max.(0.0,results.λ[1:model.pI])
        elseif d_update == :bfgs && maximum(results.μ) <= opts.bfgs_threshold
            results.λ .+= results.H*model.g(results.x)
            results.λ[1:model.pI] = max.(0.0,results.λ[1:model.pI])
        else # :first
            results.λ .+= results.μ.*results.c
            results.λ[1:pI] = max.(0.0,results.λ[1:pI])
        end
        if d_update != :bfgs || (d_update == :bfgs && maximum(results.μ) <= opts.bfgs_threshold)
            results.μ .= opts.γ.*results.μ
        end
    end

    if p_update == :feedback
        if d_update == :buys
            λ_step = zeros(p)
            active_set = zeros(Bool,p)
            for j = 1:p
                if results.Iμ[j,j] > 0.
                    active_set[j] = true
                end
            end
            λ_step[active_set] = results.λ[active_set] + (model.∇g(results.x)[active_set,:]*(model.∇²L(results.x,results.λ,results.Iμ)\model.∇g(results.x)[active_set,:]'))\model.g(results.x)[active_set]
        end
        if d_update == :bfgs
            λ_step = results.λ + results.H*model.g(results.x)
        end

        for i = 1:model.p
            if i <= model.pI
                if max(0.0,results.c[i]) <= opts.τ*max(0.0,results.c_prev[i])
                    if d_update == :buys
                        results.λ[i] = max.(0.0,λ_step[i])
                    elseif d_update == :bfgs && maximum(results.μ) <= opts.bfgs_threshold
                        results.λ[i] = max.(0.0,λ_step[i])
                    else
                        results.λ[i] += results.μ[i]*results.c[i]
                        results.λ[i] = max(0.0,results.λ[i])
                    end
                else
                    # results.μ[i] = opts.γ*results.μ[i]
                    if d_update != :bfgs || (d_update == :bfgs && maximum(results.μ) <= opts.bfgs_threshold)
                        results.μ[i] = opts.γ*results.μ[i]
                    end
                end
            else
                if abs(results.c[i]) <= opts.τ*abs(results.c_prev[i])
                    if d_update == :buys
                        results.λ[i] = λ_step[i]
                    elseif d_update == :bfgs && maximum(results.μ) <= opts.bfgs_threshold
                        results.λ[i] = λ_step[i]
                    else
                        results.λ[i] += results.μ[i]*results.c[i]
                    end
                else
                    # results.μ[i] *= opts.γ
                    if d_update != :bfgs || (d_update == :bfgs && maximum(results.μ) <= opts.bfgs_threshold)
                        results.μ[i] = opts.γ*results.μ[i]
                    end
                end
            end
        end
    end


    if d_update == :bfgs
        if maximum(results.μ) >= opts.bfgs_threshold
            y = results.c - results.c_prev
            s = results.λ - results.λ_prev

            if y'*s != 0.0
                ρ = 1.0/(y'*s)
            else
                ρ = 0.0
            end
            results.H .= (I - ρ*s*y')*results.H*(I - ρ*y*s') + ρ*s*s'
        end
    end

    results.c_prev .= copy(results.c)
    results.λ_prev .= copy(results.λ)
end

# L(x,λ,μ) = f(x) + λ'*g(x) + 0.5*g(x)'*Iμ(x,λ,μ)*g(x)
# ∇L(x,λ,μ) = ∇f(x) + ∇g(x)'*(λ +Iμ(x,λ,μ)*g(x))
# ∇²L(x,λ,μ) = ∇²f(x) + ∇g(x)'*Iμ(x,λ,μ)*∇g(x)
x0 = zeros(n)
mdl = Objective(n,p,pI,f,P,q,∇f,∇²f,L,∇L,∇²L,g,A,b,∇g)
opts = Opts(20,1e-8,1e-8,10,0.25,10)

results = Results(n,p)
val_hist, grad_hist, c_max_hist, λ_hist, outer_iter = solve_AuLa(x0,results,mdl,opts,:feedback,:buys)
@show outer_iter
# plot(val_hist,xlabel="iteration",ylabel="Cost")
# plot(grad_hist,xlabel="iteration",ylabel="Gradient l2-norm")
plot(c_max_hist,xlabel="iteration",ylabel="max violation")
c_max_hist[end]

val_hist
