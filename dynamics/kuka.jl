traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf_kuka_orig = joinpath(urdf_folder, "kuka_iiwa.urdf")
urdf_kuka = joinpath(urdf_folder, "temp/kuka.urdf")

function write_kuka_urdf()
    kuka_mesh_dir = joinpath(TrajectoryOptimization.root_dir(),"dynamics","urdf","kuka_iiwa_mesh")
    temp_dir = joinpath(TrajectoryOptimization.root_dir(),"dynamics","urdf","temp")
    if !isdir(temp_dir)
        mkdir(temp_dir)
    end
    open(urdf_kuka_orig,"r") do f
        open(urdf_kuka, "w") do fnew
            for ln in eachline(f)
                pre = findfirst("<mesh filename=",ln)
                post = findlast("/>",ln)
                if !(pre isa Nothing) && !(post isa Nothing)
                    inds = pre[end]+2:post[1]-2
                    pathstr = ln[inds]
                    file = splitdir(pathstr)[2]
                    ln = ln[1:pre[end]+1] * joinpath(kuka_mesh_dir,file) * ln[post[1]-1:end]
                end
                println(fnew,ln)
            end
        end
    end
end

# Write new urdf file with correct absolute paths
write_kuka_urdf()

model = Model(urdf_kuka)
n,m = model.n, model.m

# initial and goal states
x0 = zeros(n)
xf = zeros(n)
xf[1] = pi/2
xf[2] = pi/2

# costs
Q = 1e-4*Diagonal(I,n)
Qf = 250.0*Diagonal(I,n)
R = 1e-4*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

kuka = [model, obj_uncon]
