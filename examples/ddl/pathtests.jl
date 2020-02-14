
# include("model.jl")
include("path.jl")
N = 101
s = range(0,10,length=101)
e = sin.(s*10)*0.2

line = StraightPath(10, pi/4)
x,y = localToGlobal(line, s, e)
plot(Vector(x),Vector(y))


arc = ArcPath(-0*pi/4, 10, pi/3)
total_length(arc)
position_change(arc)
s = range(0,total_length(arc), length=N)
e = sin.(s*10)*0.2
x,y = localToGlobal(arc, s, e)
plot(x,y, aspect_ratio=:equal)
rad2deg(final_angle(arc))


line = StraightPath(10, pi/2)
arc = ArcPath(line, 10, pi)
line2 = StraightPath(arc, 12)
paths = [line,arc,line2]
paths = (line,arc,line2)
path = DubinsPath(paths)
plot(path, aspect_ratio=:equal)

s = range(0,total_length(path), length=N)
e = sin.(s*10)*0.2
e = fill(-0.2, N)
x,y = localToGlobal(path, s, e)
plot(x,y, aspect_ratio=:equal)

using BenchmarkTools
s = 1.2
@btime curvature($path, 1.2)
@code_warntype curvature(path, 1.2)
path
