
function ispossemidef(A)
	eigs = eigvals(A)
	if any(real(eigs) .< 0)
		return false
	else
		return true
	end
end

struct NotImplemented <: Exception
	fun::Symbol
	type::Symbol
end

Base.showerror(io::IO, ex::NotImplemented) =
	print(io, "Not Implemented Error: ", ex.fun, " not implemented for type ", ex.type)
