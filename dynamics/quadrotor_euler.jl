# untested
function quadrotor_dynamics_euler!(xdot,x,u)
    # States
    # x
    # y
    # z
    # phi (roll)
    # theta (pitch)
    # psi (yaw)
    # xdot
    # ydot
    # zdot
    # phidot
    # thetadot
    # psidot

    #Parameters
    m = .5 # mass
    IM = Matrix(Diagonal([0.0023,0.0023,0.004])) # inertia matrix
    invI = Matrix(Diagonal(1 ./[0.0023,0.0023,0.004])) # inverted inertia matrix
    g = 9.81 # gravity
    L = 0.1750 # distance between motors

    # states
    phi = x[4];
    theta = x[5];
    psi = x[6];

    phidot = x[10];
    thetadot = x[11];
    psidot = x[12];

    w1 = u[1];
    w2 = u[2];
    w3 = u[3];
    w4 = u[4];

    # Rotation matrix from body to world frames
    R = rpy2rotmat([phi;theta;psi]);

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


    xyz_ddot = (1/m)*([0;0;-m*g] + R*[0;0;F1+F2+F3+F4]);

    pqr = rpydot2angularvel([phi;theta;psi],[phidot;thetadot;psidot]);
    pqr = R'*pqr;

    pqr_dot = invI*([L*(F2-F4);L*(F3-F1);(M1-M2+M3-M4)] - cross(pqr,IM*pqr));

    # Now, convert pqr_dot to rpy_ddot
    Phi, dPhi = angularvel2rpydotMatrix([phi;theta;psi]);

    Rdot =  [0 (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)) (cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta));
             0 (cos(phi)*sin(psi)*sin(theta) -cos(psi)*sin(phi)) (-cos(phi)*cos(psi) -sin(phi)*sin(psi)*sin(theta));
             0 cos(phi)*cos(theta) -cos(theta)*sin(phi)]*phidot

    Rdot += [-cos(psi)*sin(theta) cos(psi)*cos(theta)*sin(phi) cos(phi)*cos(psi)*cos(theta);
                -sin(psi)*sin(theta) cos(theta)*sin(phi)*sin(psi) cos(phi)*cos(theta)*sin(psi);
                -cos(theta) -sin(phi)*sin(theta) -cos(phi)*sin(theta)]*thetadot

    Rdot += [-cos(theta)*sin(psi) (-cos(phi)*cos(psi) -sin(phi)*sin(psi)*sin(theta)) (cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta));
             cos(psi)*cos(theta)  (cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi)) (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta));
             0 0 0]*psidot

    rpy_ddot = Phi*R*pqr_dot + reshape((dPhi*[phidot;thetadot;psidot]),3,3)*R*pqr + Phi*Rdot*pqr;

    # xdot = [x[7:12];xyz_ddot;rpy_ddot];
    xdot[1:6] = x[7:12]
    xdot[7:9] = xyz_ddot
    xdot[10:12] = rpy_ddot

end

# Utilities
function rpy2rotmat(rpy)
    # equivalent to rotz(rpy(3))*roty(rpy(2))*rotx(rpy(1))

    cos_r = cos(rpy[1]);
    sin_r = sin(rpy[1]);
    cos_p = cos(rpy[2]);
    sin_p = sin(rpy[2]);
    cos_y = cos(rpy[3]);
    sin_y = sin(rpy[3]);

    rotMat = [(cos_y*cos_p) (cos_y*sin_p*sin_r-sin_y*cos_r) (cos_y*sin_p*cos_r+sin_y*sin_r);
              (sin_y*cos_p) (sin_y*sin_p*sin_r+cos_y*cos_r) (sin_y*sin_p*cos_r-cos_y*sin_r);
              -sin_p cos_p*sin_r cos_p*cos_r]
end

function rpydot2angularvel(rpy,rpydot)
    # Converts time derivatives of rpy (rolldot, pitchdot, yawdot) into the
    # angular velocity vector in base frame.  See eq. (5.41) in Craig05.
    #
    # @param rpy [roll; pitch; yaw]
    # @param rpydot time derivative of rpy
    #
    # @retval omega angular velocity vector in base frame
    # @retval domega. A 4 x 6 matrix. The gradient of omega w.r.t [rpy;rpydot]

    E = rpydot2angularvelMatrix(rpy);
    omega = E*rpydot;
end

function angularvel2rpydotMatrix(rpy)
    # Computes the matrix that transforms the angular velocity vector to the
    # time derivatives of rpy (rolldot, pitchdot, yawdot).
    # See eq. (5.41) in Craig05. Derivation in rpydot2angularvel.
    #
    # @param rpy [roll; pitch; yaw]
    # @retval Phi matrix such that Phi * omega = rpyd, where omega is the
    # angular velocity in world frame, and rpyd is the time derivative of
    # [roll; pitch; yaw]
    #
    # @retval dPhi gradient of Phi with respect to rpy
    # @retval ddPhi gradient of dPhi with respect to rpy

    # compute_gradient = nargout > 1;
    # compute_second_deriv = nargout > 2;

    p = rpy[2];
    y = rpy[3];

    sy = sin(y);
    cy = cos(y);
    sp = sin(p);
    cp = cos(p);
    tp = sp / cp;

    # warning: note the singularities!
    Phi = [cy/cp sy/cp 0;
           -sy cy 0;
           cy*tp tp*sy 1]

    sp2 = sp^2;
    cp2 = cp^2;
    dPhi = [
            0 (cy*sp)/cp2 -sy/cp;
            0 0 -cy;
            0 (cy + (cy*sp2)/cp2) -(sp*sy)/cp;
            0 (sp*sy)/cp2 cy/cp;
            0 0 -sy;
            0 (sy + (sp2*sy)/cp2)  (cy*sp)/cp;
            0 0 0;
            0 0 0;
            0 0 0];

    Phi, dPhi
end

function rpydot2angularvelMatrix(rpy)
    # Computes matrix that converts time derivatives of rpy
    # (rolldot, pitchdot, yawdot) into the angular velocity vector expressed in
    # base frame.  See eq. (5.41) in Craig05.
    #
    # @param rpy [roll; pitch; yaw]
    #
    # @retval E matrix such that omega = E * rpyd, where omega is the angular
    # velocity vector in base frame and rpyd is the time derivative of rpy
    # @retval dE. A 9 x 3 matrix. The gradient of E w.r.t rpy

    # % Derived using:
    # % syms r p y real; rpy=[r p y];
    # % R = rpy2rotmat(rpy);
    # % E(1,:) = jacobian(R(3,1),rpy)*R(2,1) + jacobian(R(3,2),rpy)*R(2,2) + jacobian(R(3,3),rpy)*R(2,3);
    # % E(2,:) = jacobian(R(1,1),rpy)*R(3,1) + jacobian(R(1,2),rpy)*R(3,2) + jacobian(R(1,3),rpy)*R(3,3);
    # % E(3,:) = jacobian(R(2,1),rpy)*R(1,1) + jacobian(R(2,2),rpy)*R(1,2) + jacobian(R(2,3),rpy)*R(1,3);
    # % simplify(E)
    # % Note: I confirmed that the same recipe yields (5.42)

    # % r=rpy(1);
    p=rpy[2];
    y=rpy[3];

    cos_p = cos(p);
    sin_p = sin(p);
    cos_y = cos(y);
    sin_y = sin(y);

    E = [ cos_p*cos_y -sin_y 0;
          cos_p*sin_y cos_y 0;
    	   -sin_p 0 1];
end

# Model
n = 12
m = 4

quadrotor_euler = Model(quadrotor_dynamics_euler!,n,m)
