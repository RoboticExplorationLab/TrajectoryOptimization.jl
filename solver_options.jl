import Base.show

mutable struct SolverOptions
    square_root::Bool
    augmented_lagrange::Bool
    verbose::Bool

    function SolverOptions(;square_root=false,al=false,verbose=false)
        new(square_root,al)
    end
end

function show(io::IO, opts::SolverOptions)
    println(io, "SolverOptions:")
    print(io,"  Use Square Root: $(opts.square_root)")
end
