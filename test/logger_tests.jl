using Logging
using Test
import TrajectoryOptimization: InnerLoop, print_header, print_row

solver = Solver(Dynamics.pendulum...)
solver.opts.verbose = true
logger = TrajectoryOptimization.default_logger(solver)
@test_logs (:info, "hi") @info "hi"

print_header(logger,InnerLoop)
with_logger(logger) do
    @logmsg InnerLoop :iter value=1
    @logmsg InnerLoop "stuff"
    @logmsg InnerLoop :newcol value=10.2
end
print_row(logger,InnerLoop)

@info "hi"

logger.leveldata[InnerLoop]
