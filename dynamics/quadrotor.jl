function quadrotor_dynamics!(xdot,X,u)
      #TODO change concatentations to make faster!
      # Modified from D. Mellinger, N. Michael, and V. Kumar,
      # "Trajectory generation and control for precise aggressive maneuvers with quadrotors",
      # In Proceedings of the 12th International Symposium on Experimental Robotics (ISER 2010), 2010.

      ## States: X ∈ R^13; q = [v;s]
      # x
      # y
      # z
      # q1
      # q2
      # q3
      # q4
      # xdot
      # ydot
      # zdot
      # omega1
      # omega2
      # omega3

      x = X[1:3]
      q = X[4:7]
      v = X[8:10]
      omega = X[11:13]

      # Parameters
      m = .5 # mass
      I = diagm([0.0023,0.0023,0.004]) # inertia matrix
      invI = diagm(1./[0.0023,0.0023,0.004]) # inverted inertia matrix
      g = 9.81 # gravity
      L = 0.1750 # distance between motors

      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      kf = 1; # 6.11*10^-8;

      F1 = kf*w1;
      F2 = kf*w2;
      F3 = kf*w3;
      F4 = kf*w4;

      km = 0.0245;

      M1 = km*w1;
      M2 = km*w2;
      M3 = km*w3;
      M4 = km*w4;

      tmp = hamilton_product(q,hamilton_product([0;0;F1+F2+F3+F4;0],quaternion_congugate(q))) #TODO does the quaternion need to be unit when we do this rotation? or is unit quaternion only required when we convert quaterion to rotation matrix
      a = (1/m)*([0;0;-m*g] + tmp[1:3]);

      xdot[1:3] = v # velocity
      xdot[4:7] = 0.5*hamilton_product(q,[omega;0]) # ̇q TODO is this concatenation slow?
      xdot[8:10] = a # acceleration
      xdot[11:13] = invI*([L*(F2-F4);L*(F3-F1);(M1-M2+M3-M4)] - cross(omega,I*omega)) # ̇ω; Euler's equation: I(̇ω) + ω x Iω = τ
end

function quadrotor_dynamics(X,u)
      xdot = zeros(13,1)
      quadrotor_dynamics!(xdot,X,u)
      xdot
end

function hamilton_product(q1,q2)
      # perform quaternion multiplication
      Q1 = [q1[1] -q1[2] -q1[3] -q1[4];
            q1[2]  q1[1] -q1[4]  q1[3];
            q1[3]  q1[4]  q1[1] -q1[2];
            q1[4] -q1[3]  q1[2]  q1[1]]

      Q1*q2
end

function quaternion_congugate(q)
      # calculate the congugate of a quaternion: q^+; q = [v;s] -> q^+ = [-v;s]
      q_ = zeros(4)
      q_[1] = -q[1]
      q_[2] = -q[2]
      q_[3] = -q[3]
      q_[4] = q[4]

      q_
end

xdot = ones(13,1)

quadrotor_dynamics!(xdot,ones(13,1),ones(4,1))
xdot
quadrotor_dynamics(ones(13,1),ones(4,1))
