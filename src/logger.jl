using Logging

const OuterLoop = LogLevel(-100)
const InnerLoop = LogLevel(-200)
const InnerIters = LogLevel(-500)

function default_logger(verbose::Bool)
    verbose == false ? min_level = Logging.Warn : min_level = InnerLoop

    logger = SolverLogger(min_level)
    inner_cols = [:iter, :cost, :expected, :z, :α, :info]
    inner_widths = [5,     14,      12,    10, 10,    50]
    outer_cols = [:iter, :total, :c_max, :info]
    outer_widths = [6,          7,        12,        50]
    add_level!(logger, InnerLoop, inner_cols, inner_widths, print_color=:green,indent=4)
    add_level!(logger, OuterLoop, outer_cols, outer_widths, print_color=:yellow,indent=0)
    return logger
end
default_logger(solver::Union{Solver,AbstractSolver}) = default_logger(solver.opts.verbose)

function default_logger(solver::AugmentedLagrangianSolver{T}) where T
    solver.opts.verbose == false ? min_level = Logging.Warn : min_level = InnerLoop

    logger = SolverLogger(min_level)
    outer_cols = [:iter, :total, :c_max, :info]
    outer_widths = [6,          7,        12,        50]
    if solver.opts.opts_uncon.verbose
        freq = 1
    else
        freq = 5
    end
    add_level!(logger, OuterLoop, outer_cols, outer_widths, print_color=:yellow,
        indent=0, header_frequency=freq)
    return logger
end
"""
$(SIGNATURES)
Holds logging information about a particular print level, meant to assemble
a table of output where `cols` gives the order and names of columns, `widths`
are the column widths, and `print` will turn on/off printing the column.

All values can be cached at any moment in time to accumulate a history of the
data.
"""
struct LogData
    cols::Vector{Symbol}
    widths::Vector{Int}
    print::BitArray{1}
    data::Dict{Symbol,Any}
    cache::Dict{Symbol,Vector}
    metadata::NamedTuple
    LogData(cols,widths,print::BitArray{1},data::Dict{Symbol,Any},cache::Dict{Symbol,Vector},
        metadata::NamedTuple) = new(cols,widths,print,data,cache,metadata)
end

Base.getindex(ldata::LogData,index::Symbol) = ldata.data[index]

"$(SIGNATURES) Default (empty) constructor)"
function LogData(metadata::NamedTuple=(color=:default,header_frequency=10, indent=0))
    LogData(Symbol[],Int[],BitArray{1}(),Dict{Symbol,Any}(),Dict{Symbol,Vector}(),metadata)
end

"$(SIGNATURES) Create LogData pre-specifying columns, widths, and optionally printing and variable types (recommended)"
function LogData(cols,widths; do_print=trues(length(cols)), vartypes=fill(Any,length(cols)),
        color=:default, header_frequency=10, indent=0)
    metadata = (color=color,header_frequency=header_frequency,indent=indent)
    ldata = LogData(metadata)
    for (col,width,prnt,vartype) in zip(cols,widths,do_print,vartypes)
        add_col!(ldata,col,width,do_print=prnt,vartype=vartype)
    end
    ldata
end

"$(SIGNATURES) Store the current data in the cache"
function cache_data!(ldata::LogData)
    for (key,val) in ldata.data
        if isempty(val) && eltype(ldata.cache[key]) <: Number
            val = NaN
        end
        push!(ldata.cache[key],val)
    end
end

"$(SIGNATURES) Size of the current cache"
function cache_size(ldata::LogData)
    if isempty(ldata.cols) || isempty(ldata.cache)
        return 0
    else
        return length(ldata.cache[ldata.cols[1]])
    end
end

"$(SIGNATURES) Clear the current data fields (not the cache)"
function clear!(ldata::LogData)
    for key in keys(ldata.data)
        if key == :info
            ldata.data[key] = String[]
        else
            ldata.data[key] = ""
        end
    end
end

function clear_cache!(ldata::LogData)
    for key in keys(ldata.data)
        if key == :info
            ldata.cache[key] = Vector{Vector{String}}()
        else
            ldata.cache[key] = Vector{eltype(ldata.cache[key])}()
        end
    end
end

"""$(SIGNATURES) Add a column to the table
# Arguments
* idx: specify the order in the table. 0 will insert at the end, and any negative number will index from the end
* do_print: specify whether the variable should be printed or just kept for caching purposes. Default=true
* vartype: Type of the variable (recommended). Default=Any
"""
function add_col!(ldata::LogData,name::Symbol,width::Int=10,idx::Int=0; do_print::Bool=true, vartype::Type=Any)
    # Don't add a duplicate column
    if name ∈ ldata.cols; return nothing end

    # Set location
    name == :info ? idx = 0 : nothing
    if idx <= 0
        idx = length(ldata.cols) + 1 + idx
    end

    # Add to ldata
    insert!(ldata.cols,idx,name)
    insert!(ldata.widths,idx,width)
    insert!(ldata.print,idx,do_print)
    if name == :info
        ldata.data[name] = String[]
    else
        ldata.data[name] = ""
    end

    ldata.cache[name] = Vector{vartype}(undef,cache_size(ldata))
    return nothing
end

"$(SIGNATURES) Create the header row (returns a string)"
function create_header(ldata::LogData, delim::String="")
    :indent in keys(ldata.metadata) ? indent = ldata.metadata.indent : indent = 0
    repeat(" ",indent) * join([rpad(col, width) for (col,width,do_print) in zip(ldata.cols,ldata.widths,ldata.print) if do_print],delim) * "\n" * repeat("_",indent) * repeat('-',sum(ldata.widths)) *"\n"
end

"$(SIGNATURES) Create a data row (returns a string)"
function create_row(ldata::LogData)
    :indent in keys(ldata.metadata) ? indent = ldata.metadata.indent : indent = 0
    row = repeat(" ",indent) * join([begin rpad(trim_entry(ldata.data[col],width),width) end for (col,width,do_print) in zip(ldata.cols,ldata.widths,ldata.print) if col != :info && do_print])
    if :info in ldata.cols
        row *= join(ldata.data[:info],". ")
    end
    return row
end

"$(SIGNATURES) Convert a float to a string, keeping the whole thing within a given character width"
function trim_entry(data::Float64,width::Int; pad=true, kwargs...)
    base = log10(abs(data))
    if -ceil(width/2)+1 < base < floor(width / 2) && isfinite(data)
        if base > 0
            prec = width - ceil(Int,base) - 3
        else
            prec = width - 4
        end
        if prec <= 0
            width = width - prec + 1
            prec = 1
        end
        val = format(data,precision=prec,conversion="f",stripzeros=true,positivespace=true; kwargs...)
    elseif !isfinite(data)
        val = string(data)
    else
        width <= 8 ? width = 10 : nothing
        val = format(data,conversion="e",precision=width-8,stripzeros=true,positivespace=true; kwargs...)
    end
    if pad
        val = rpad(val,width)
    end
    return val
end

"$(SIGNATURES) Chops off the end of a string to keep it at width"
function trim_entry(data::String,width; pad=true)
    if length(data) > width-2
        return data[1:width-2]
    end
    return data
end

function trim_entry(data::Int, width::Int; pad=true, kwargs...)
    if pad
        rpad(format(data; kwargs...), width)
    else
        format(data; kwargs...)
    end
end

function trim_entry(data,width; pad=true, kwargs...)
    data
end





"""
$(TYPEDEF)
Logger class for generating output in a tabular format (by iteration)

In general, only levels "registered" with the logger will be used, otherwise
they are passed off to the global logger. Typical use will include setting up
the LogLevels that will be logged as tables, and then using @logmsg to send
information to the logger. When enough data has been gathered, the user can then
print a row for a certain level.
"""
struct SolverLogger <: Logging.AbstractLogger
    io::IO
    min_level::LogLevel
    default_width::Int
    leveldata::Dict{LogLevel,LogData}
    default_logger::ConsoleLogger
end

Base.getindex(logger::SolverLogger, level::LogLevel) = logger.leveldata[level]


function SolverLogger(min_level::LogLevel=Logging.Info; default_width=10, io::IO=stderr)
    SolverLogger(io,min_level,default_width,Dict{LogLevel,LogData}(),ConsoleLogger(stderr,min_level))
end

""" $(SIGNATURES)
"Register" a level with the logger, creating a LogData entry responsible for storing
data generated at that level. Additional keyword arguments (from LogData constructor)

* vartypes = Vector of variable types for each column
* do_print = BitArray specifying whether or now the column should be printed (or just cached and not printed)
"""
function add_level!(logger::SolverLogger, level::LogLevel, cols, widths; print_color=:default,
        indent=0, kwargs...)
    logger.leveldata[level] = LogData(cols, widths, color=print_color, indent=indent; kwargs...)
end

function Base.println(logger::SolverLogger, level::LogLevel)
    ldata = logger[level]
    if cache_size(ldata) % ldata.metadata.header_frequency == 0
        print_header(logger,level)
    end
    print_row(logger,level)
end

"$(SIGNATURES) Print the header row for a given level (in color)"
function print_header(logger::SolverLogger,level::LogLevel)
    if level in keys(logger.leveldata) && level >= logger.min_level
        ldata = logger.leveldata[level]
        printstyled(logger.io,create_header(ldata),
            bold=true,color=ldata.metadata.color)
    end
end

"$(SIGNATURES) Print a row of data and cache it with LogData"
function print_row(logger::SolverLogger,level::LogLevel)
    if  level >= logger.min_level
        println(logger.io, create_row(logger.leveldata[level]))
        cache_data!(logger.leveldata[level])
        clear!(logger.leveldata[level])
    end
end



"""$(SIGNATURES) Send data from log events to LogData columns
The message needs to be a symbol that corresponds to the column name
The value is provided as a keyword argument `value=...`
If the level is not "registered" it is passed on to the global logger

If the message is not in the list of columns, it will be added, in which case
the type is inferred from the value and printing can be specified.

Usage Example:
@info :myvar value=10.2  # Sends a value of 10.2 to the column "myvar"

"""
function Logging.handle_message(logger::SolverLogger, level, message::Symbol, _module, group, id, file, line; value=NaN, print=true, loc=-1, width=logger.default_width)
    if level in keys(logger.leveldata)
        if level >= logger.min_level
            ldata = logger.leveldata[level]
            if !(message in ldata.cols)
                :info in ldata.cols ? idx = loc : idx = 0  # add before last "info" column
                width = max(width,length(string(message))+1)
                add_col!(ldata, message, width, idx, do_print=print, vartype=typeof(value))
            end
            logger.leveldata[level].data[message] = value
        end
    else level >= logger.min_level
        # Pass off to global logger
        Logging.handle_message(logger.default_logger, level, message, _module, group, id, file, line)
    end
end

"$(SIGNATURES) Adds a message to the special :info column, which will join all messages together and print them (like a NOTES column)"
function Logging.handle_message(logger::SolverLogger, level, message::String, _module, group, id, file, line; value=NaN)
    if level in keys(logger.leveldata)
        if level >= logger.min_level
            ldata = logger.leveldata[level]
            if !(:info in ldata.cols)
                add_col!(ldata, :info, 20)
                ldata.data[:info] = String[]
            end
            # Append message to info field
            push!(ldata.data[:info], message)
        end
    else
        # Pass off to global logger
        Logging.handle_message(logger.default_logger, level, message, _module, group, id, file, line)
    end
end

function Logging.shouldlog(logger::SolverLogger, level, _module, group, id)
    true  # Accept everything
end

function Logging.min_enabled_level(logger::SolverLogger)
    return logger.min_level
end

# logger = SolverLogger(InnerLoop)
# inner_cols = [:iter, :cost, :expected, :actual, :z, :α, :c_max, :info]
# inner_widths = [6,     12,      12,         12,  12, 12,   12,      40]
# inner_types = [Int, Float64, Float64, Float64, Float64, Float64, Float64, Any]
# inner_print = [true,  true,   true,      true,   true, true,false,true]
# outer_cols = [:outeriter, :iter, :iterations, :info]
# outer_widths = [4,          4,        4,        40]
#
#
# add_level!(logger, InnerLoop, inner_cols, inner_widths, do_print=inner_print, print_color=:green, indent=4, vartypes=inner_types)
# add_level!(logger, OuterLoop, outer_cols, outer_widths, print_color=:yellow,indent=0)
#
# with_logger(logger) do
#     @logmsg InnerLoop "Hi"
#     @logmsg InnerLoop "A condition"
#     @logmsg InnerLoop :new value=10.1
#     @logmsg InnerLoop :iter value=1
# end
#
# print_header(logger,InnerLoop)
# print_row(logger,InnerLoop)
#
#
# with_logger(logger) do
#     @logmsg InnerLoop :iter value=2
#     @logmsg InnerLoop :new value=10.2
#     @logmsg InnerLoop :z value=1e-3
#     @logmsg InnerLoop :newcost value=100.1 print=false
# end
# print_row(logger,InnerLoop)
#
# ldata = logger.leveldata[InnerLoop]



#
# printstyled(logger.io,create_header(ldata),
#     bold=true,color=ldata.metadata.color)


# print_header(logger,InnerLoop)
#
# # Inner loop
# level = Logging.Info
# col_widths = [20,20,20]
# cols = [:iter, :Cost, :dJ]
# vals = Dict(:iter=>1, :Cost=>10.2, :dJ=>0.02)
#
# ldata = LogData(cols,col_widths)
# add_col!(ldata,:z,10)
#
# logger = SolverLogger()
# add_level!(logger, level, cols, col_widths, print_color=:green,indent=4)
# :dJ in logger.leveldata[level].cols
#
# with_logger(logger) do
#     @info :dJ value=10
#     @warn :something
#     @logmsg inner :dJ value=10
# end
#
# const InnerLevel = LogLevel(500)
#
# with_logger(logger) do
#     @info :iter value=1
#     @info :dJ value=10.2
#     @info :Cost value=100.1
# end
#
# print_header(logger,level)
# print_row(logger,level)
#
# with_logger(logger) do
#     @info :iter value=2
#     @info :dJ value=9.6
#     @info :Cost value=55
#     @info :k value=12.3
# end
#
# print_row(logger,level)
