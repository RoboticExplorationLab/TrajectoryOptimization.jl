
# function ispossemidef(A)
# 	eigs = eigvals(A)
# 	if any(real(eigs) .< 0)
# 		return false
# 	else
# 		return true
# 	end
# end

# struct NotImplemented <: Exception
# 	fun::Symbol
# 	type::Symbol
# end

# @inline NotImplemented(fun::Symbol, type::DataType) = NotImplemented(fun, Symbol(DataType))
# @inline NotImplemented(fun::Symbol, type) = NotImplemented(fun, Symbol(typeof(type)))

# Base.showerror(io::IO, ex::NotImplemented) =
# 	print(io, "Not Implemented Error: ", ex.fun, " not implemented for type ", ex.type)

# function gen_zinds(n::Int, m::Int, N::Int, isequal::Bool=false)
# 	Nu = isequal ? N : N-1
# 	zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:Nu]
# 	if !isequal
# 		push!(zinds, (N-1)*(n+m) .+ (1:n))
# 	end
# 	return zinds
# end
