using BenchmarkTools
using RigidBodyDynamics
using Statistics
using Formatting
using Logging
import TrajectoryOptimization: get_num_states, BackwardPassZOH, gradient_todorov, evaluate_convergence, outer_loop_update, update_constraints!, update_jacobians!

function create_benchmarks(solver,X0,U0)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    results = init_results(solver,X0,U0)
    J_prev = cost(solver, results)
    bp = BackwardPassZOH(nn,mm,N)
    update_jacobians!(results, solver)
    Δv = backwardpass!(results, solver, bp)
    J = forwardpass!(results, solver, Δv, J_prev)

    group = BenchmarkGroup()
    group["init_results"] = @benchmarkable init_results($solver,$X0_empty,$U0)
    group["jacobians"] = @benchmarkable update_jacobians!($results, $solver)
    group["backwardpass"] = @benchmarkable backwardpass!($results, $solver, $bp)
    group["forwardpass"] = @benchmarkable forwardpass!($results, $solver, $Δv, $J_prev)
    if solver.state.constrained
        group["update_constraints"] = @benchmarkable update_constraints!($results, $solver)
        group["max_violation"] = @benchmarkable max_violation($results)
        group["outer_loop_update"] = @benchmarkable outer_loop_update($results, $solver, $bp, 1)
    end
    return group
end


function collect_data(r1,r2,f=median)
    j = judge(f(r2),f(r1))
    cols = collect(keys(j))
    header = join([rpad(col,20) for col in cols]," | ")
    times = join.([[prettytime(median(r[col]).time) for r in (r1,r2)] for col in cols]," ← ")
    times .*= " (" .* [prettydiff(ratio(j[col]).time) for col in cols] .* ")"
    mem = join.([[prettymemory(median(r[col]).memory) for r in (r1,r2)] for col in cols], " ← ")
    mem .*= " (" .* [prettydiff(ratio(j[col]).memory) for col in cols] .* ")"
    alloc = join.([[format(median(r1[col]).allocs,autoscale=:metric) for r in (r1,r2)] for col in cols], " ← ")
    alloc .*= " (" .* [prettydiff(ratio(j[col]).allocs) for col in cols] .* ")"
    data = [cols times mem alloc]
    invar = repeat([:invariant],length(cols))
    change = [invar [val(j[col]) for col in cols, val in (time, memory)] invar]
    return data, change
end

function compare_system(res1,res2,name)
    delim = " | "
    r1,r2 = res1[name], res2[name]
    data,change = collect_data(r1,r2)
    col_width = maximum(length.(data),dims=1)
    col_width[1] = max(col_width[1], length(name))
    header = [name,"time","memory","allocs"]
    header = delim * create_row(header,col_width,delim) * delim
    n,m = size(data)
    colormap = Dict(:invariant=>:grey, :regression=>:red, :improvement=>:green)

    printstyled(header,bold=true); println();
    println(repeat("-",sum(col_width) + length(delim)*4 + 2));
    for i = 1:n
        print(delim)
        for j = 1:m
            printstyled(rpad(data[i,j],col_width[j]),color=colormap[change[i,j]])
            print(delim)
        end
        print("\n")
    end
    print("\n")
    collect_data(r1,r2)
end

create_row(data,widths,delim="") = join([rpad(d,width) for (d,width) in zip(data,widths)],delim)

group = BenchmarkGroup()

#####     Kuka arm       #######
# Unconstrained
kuka_group = BenchmarkGroup()
model, obj = Dynamics.kuka
kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)
solver = Solver(model,obj,N=101)
n,m,N = get_sizes(solver)
m̄,mm = get_num_controls(solver)
n̄,nn = get_num_states(solver)

U0 = ones(m,N-1)*10
U0 = TrajectoryOptimization.hold_trajectory(solver, obj.x0[1:7])
X0 = line_trajectory(solver)
X0_empty = Array{Float64,2}(undef,0,0)

kuka_group["unconstrained"] = create_benchmarks(solver,X0_empty,U0)

# Constrained
function create_sphere_constraint(kuka::Mechanism, circles)
    n_obstacles = length(circles)

    state_cache = StateCache(kuka)
    nn = num_positions(kuka)
    ee_body, ee_point = Dynamics.get_kuka_ee(kuka)
    world = root_frame(kuka)

    function cI(c,x::AbstractVector{T},u) where T
        state = state_cache[T]
        set_configuration!(state,x[1:nn])
        ee = transform(state,ee_point,world).v
        for i = 1:n_obstacles
            c[i] = sphere_constraint(ee,circles[i][1],circles[i][2],circles[i][3],circles[i][4])
        end
    end
end

circles = [[0,0,.8,.2],]
cI = create_sphere_constraint(kuka,circles)
obj_con = ConstrainedObjective(obj,u_min=-100,u_max=100,cI=cI)

solver = Solver(model,obj_con,N=101)
kuka_group["constrained"] = create_benchmarks(solver,X0_empty,U0)

#####     Dubins Car      #######
dubin_group = BenchmarkGroup()
model, obj = Dynamics.dubinscar
solver = Solver(model, obj, N=101)
n,m,N = get_sizes(solver)
U0 = ones(m,N-1)
X0 = Array{Float64,2}(undef,0,0)
X0

dubin_group["unconstrained"] = create_benchmarks(solver,X0,U0)

model,obj_con = Dynamics.dubinscar_parallelpark
solver = Solver(model, obj_con, N=101)
dubin_group["constrained"] = create_benchmarks(solver,X0,U0)

dubin_group["constrained"]["solve"] = @benchmarkable solve(solver,U0)

# Create suite
date = today()
SUITE = BenchmarkGroup([date])
SUITE["kuka"] = kuka_group
SUITE["dubinscar"] = dubin_group
suite = SUITE

loadparams!(SUITE, BenchmarkTools.load(paramsfile)[1])

r1 = res
r2 = res_base
compare_system(r1["dubinscar"],r2["dubinscar"],"constrained")

#
# table_data = copy(cols)
# for sys in systems
#     data, change = collect_data(res1[sys],res2[sys])
#     [table_data data]
# end
#
# cols = collect(keys(res1["constrained"]))
#
# systems = collect(keys(res1))
# table = [collect_data(res1[sys],res2[sys]) for sys in systems]
# table_data = hcat([t[1] for t in table]...)
# table_change = hcat([t[2] for t in table]...)
#
#
# col_width = maximum(length.(table_data),dims=1) .+ 2
# name_width = maximum(length.(cols)) + 1
# table_widths = [name_width col_width]
#
# sys_cols = [(1:3) .+ 3(i-1) for i = 1:length(systems)]
# sys_width = [sum(col_width[cols]) for cols in sys_cols]
# header = [rpad(lpad(sys,width÷2),width) for (sys,width) in zip(systems,sys_width)]
# insert!(header,1," ")
# header = create_row(header,[name_width; sys_width]
# subheader = [""; repeat(["time","memory","allcos"],2)]
# subheader = create_row(subheader,table_widths,"|")
