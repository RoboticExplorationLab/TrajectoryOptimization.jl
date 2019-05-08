using LinearAlgebra, MeshCat, StaticArrays, GeometryTypes, Polyhedra, Meshing, Colors, CoordinateTransformations
vis = Visualizer()
open(vis)

geom = Cylinder(Point3f0([0.,0.,0.]),Point3f0(0.,0.,1.),convert(Float32,0.5))
setobject!(vis["cyl"],geom,MeshPhongMaterial(color=RGBA(1.0, 1.0, 1.0, 1.0)))

#linear
pln = x -> x[1] - 0.5001
sdf = SignedDistanceField(pln, HyperRectangle(Vec(-1, -0.5, 0), Vec(2, 1, 1)))
mesh = HomogenousMesh(sdf, MarchingTetrahedra())
setobject!(vis["linear"], mesh, MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5)))
settransform!(vis["/Cameras/default"], compose(Translation(1., -1., 1.),LinearMap(RotX(0.)*RotZ(-pi/4))))

#quad
geom = Cylinder(Point3f0([0.,0.,0.]),Point3f0(0.,0.,1.),convert(Float32,0.5))
setobject!(vis["cyl"],geom,MeshPhongMaterial(color=RGBA(1.0, 1.0, 1.0, 0.8)))
geom = Cylinder(Point3f0([0.,0.,0.]),Point3f0(0.,0.,0.4),convert(Float32,0.5))
setobject!(vis["cyl"]["bottom"],geom,MeshPhongMaterial(color=RGBA(1.0, 1.0, 1.0, 1.0)))
# function quad(x)
#         (-0.5*(x[1:2])'*[3 0; 0 3]*(x[1:2])) - x[3] + 0.875
# end

# sdf = SignedDistanceField(quad, HyperRectangle(Vec(-.5, -.5, 0.5), Vec(1, 1, 2)))
function quad(x)
        (-0.5*(x[1:2])'*[2 0; 0 2]*(x[1:2])) - x[3] + 0.65
end
sdf = SignedDistanceField(quad, HyperRectangle(Vec(-0.75, -0.75, .1), Vec(1.5, 1.5, 2)))

mesh = HomogenousMesh(sdf, MarchingTetrahedra())
setobject!(vis["quad"], mesh, MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5)))
settransform!(vis["/Cameras/default"], compose(Translation(1., -1., 1.),LinearMap(RotX(0.)*RotZ(-pi/4))))

#barrier
geom = Cylinder(Point3f0([0.,0.,0.]),Point3f0(0.,0.,1.),convert(Float32,0.5))
setobject!(vis["cyl"],geom,MeshPhongMaterial(color=RGBA(1.0, 1.0, 1.0, 1.0)))
barrier = x-> -0.2*log(x[2] - 0.4999) - x[3]
sdf = SignedDistanceField(barrier, HyperRectangle(Vec(-.5, 0.5, 0.), Vec(1, 1, 1.25)))
mesh = HomogenousMesh(sdf, MarchingTetrahedra())
setobject!(vis["barrier"], mesh, MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5)))

settransform!(vis["/Cameras/default"], compose(Translation(1., 1., 1.),LinearMap(RotX(0.)*RotZ(pi/4))))
