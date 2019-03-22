using Logging
using Test
import TrajectoryOptimization: LogData, cache_size, add_col!, create_header, create_row, cache_data!, clear!

# Default log data
ld = LogData()
@test isempty(ld.cols)
@test isempty(ld.data)
@test ld.metadata.color == :default

cols = [:iter,:cost]
width = [8,10,10]
ld = LogData(cols,width)
@test ld.cols == cols
@test !(ld.cols === cols)
@test !(ld.widths === width)
@test cache_size(ld) == 0
add_col!(ld,:c_max, 15, vartype=Float64)
@test ld.cols == [:iter,:cost,:c_max]
@test ld.widths[3] == 15
add_col!(ld,:outer, 15, 2, vartype=Float64, do_print=false)
@test ld.cols == [:iter,:outer,:cost,:c_max]
@test ld.print[2] == false
add_col!(ld,:cost)
@test length(ld.cols) == 4

strhead = create_header(ld)
@test occursin("iter",strhead)
@test !occursin("outer",strhead)

# Add entry
ld.data[:iter] = 1
ld.data[:outer] = 10
ld.data[:cost] = 2.5
strrow = create_row(ld)
@test occursin("1",strrow)
@test occursin("2.5",strrow)
@test !occursin("10",strrow)

# Cache row
cache_data!(ld)
@test ld.cache[:iter][1] == 1
@test ld.cache[:outer][1] == 10
@test isnan(ld.cache[:c_max][1])
@test ld.data[:iter] == 1

# Clear data
clear!(ld)
@test ld.data[:iter] == ""


# Logger Tests
import TrajectoryOptimization: SolverLogger, add_level!, InnerLoop, OuterLoop, print_header, print_row
logger = SolverLogger(InnerLoop, default_width=15)
add_level!(logger, InnerLoop, cols, width; print_color=:green, indent=4)
@test logger.leveldata[InnerLoop].metadata.indent == 4
add_level!(logger, InnerLoop, cols, width; print_color=:green, indent=5)
@test logger.leveldata[InnerLoop].metadata.indent == 5
@test length(logger.leveldata) == 1
@test :iter in logger[InnerLoop].cols

# Add some data
global_logger(logger)
@logmsg InnerLoop :iter value=5
@test logger[InnerLoop].data[:iter] == 5
@test logger[InnerLoop].data[:cost] == ""
@logmsg InnerLoop :cost value=10.5
@test logger[InnerLoop].data[:cost] == 10.5

# Printing clears and caches the values
print_row(logger,InnerLoop)
@test logger[InnerLoop].data[:iter] == ""
@test logger[InnerLoop].data[:cost] == ""
@test logger[InnerLoop].cache[:iter][1] == 5
@test logger[InnerLoop].cache[:cost][1] == 10.5
@test cache_size(logger[InnerLoop]) == 1

# Add column
@logmsg InnerLoop :c_max value=3.2
@test logger[InnerLoop].cols[3] == :c_max
@logmsg InnerLoop :iter value=2
@test logger[InnerLoop].data[:iter] == 2
@test logger[InnerLoop].data[:c_max] == 3.2

# Add Info column
@logmsg InnerLoop "Something happened"
@test logger[InnerLoop][:info][1] == "Something happened"
@logmsg InnerLoop "Awesome"
@test logger[InnerLoop][:info][2] == "Awesome"
strrow = create_row(logger[InnerLoop])
@test occursin("Something happened. Awesome", strrow)

# Try a level not in the logger
@test_logs (:warn,:iter) @warn :iter   # Should log
@test_logs @logmsg LogLevel(-500) :hi  # No logs

function test_logging(a,level)
    logger = global_logger()
    for i = 1:length(a)
        @logmsg level :iter value=a[i]
        cache_data!(logger[level])
    end
end
a = 1:100
TrajectoryOptimization.clear_cache!(logger[InnerLoop])
test_logging(a,InnerLoop)
cache_size(logger[InnerLoop])
TrajectoryOptimization.clear_cache!(logger[InnerLoop])
@btime test_logging($a,InnerLoop)

function test_dict(a,d)
    for i = 1:length(a)
        push!(d[:iter],a[i])
    end
end
d = Dict(:iter=>Int[])
test_dict(1:10,d)
@btime test_dict(a,$d)
