include("model.jl")

# Model
car = ContingencyCar()
n,m = size(car)
p_c = 0.1 # probability of contingency plan
N = 30


# Objective
Qd = @SVector [0,      # δ
               4e-10,  # fx
               0,      # r
               0,      # Uy
               0.0156, # Ux
               0,      # dpsi
               4,      # e
               0]      # t

Rd   = @SVector [0.1013,  # δ_dot
                 4e-8]    # fx_dot
Rd_c = @SVector [0.3282,  # δ_dot
                 4e-8]    # fx_dot

Q = Diagonal([(1-p_c)*Qd; p_c*Qd])
R = Diagonal([Rd; Rd_c])

obj = LQRObjective(Q,R,Q*N,xf)
