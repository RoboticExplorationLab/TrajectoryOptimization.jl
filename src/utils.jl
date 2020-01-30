export
	benchmark_solve!

function interp_rows(N::Int,tf::Float64,X::AbstractMatrix)::Matrix
    n,N1 = size(X)
    t1 = range(0,stop=tf,length=N1)
    t2 = collect(range(0,stop=tf,length=N))
    X2 = zeros(n,N)
    for i = 1:n
        interp_cubic = CubicSplineInterpolation(t1, X[i,:])
        X2[i,:] = interp_cubic(t2)
    end
    return X2
end

function ispossemidef(A)
	eigs = eigvals(A)
	if any(real(eigs) .< 0)
		return false
	else
		return true
	end
end

function convertInf!(A::VecOrMat{Float64},infbnd=1.1e20)
    infs = isinf.(A)
    A[infs] = sign.(A[infs])*infbnd
    return nothing
end

function set_logger()
    if !(global_logger() isa SolverLogger)
        global_logger(default_logger(true))
    end
end

function benchmark_solve!(solver; samples=10, evals=10)
    U0 = deepcopy(controls(solver))
    solver.opts.verbose = false
    b = @benchmark begin
        initial_controls!($solver,$U0)
        solve!($solver)
    end samples=samples evals=evals
    return b
end

function benchmark_solve!(solver, data::Dict; samples=10, evals=10)
	b = benchmark_solve!(solver, samples=samples, evals=evals)

   # Run stats
   push!(data[:time], time(median(b))*1e-6)  # ms
   push!(data[:iterations], solver.stats.iterations)
   push!(data[:cost], solver.stats.cost[end])
   return b
end
