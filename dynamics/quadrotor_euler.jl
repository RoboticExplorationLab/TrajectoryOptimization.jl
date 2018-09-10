function quadrotor_euler(x,u)
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
    m = .5;
    I = diagm([0.0023;0.0023;0.004]);
    invI = diagm(1./[0.0023;0.0023;0.004]);
    g = 9.81;
    L = 0.1750;

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

    pqr_dot = invI*([L*(F2-F4);L*(F3-F1);(M1-M2+M3-M4)] - cross(pqr,I*pqr));

    # Now, convert pqr_dot to rpy_ddot
    Phi, dPhi = angularvel2rpydotMatrix([phi;theta;psi]);

    # Rdot =  [ 0, sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta),   cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta);
    #           0, cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi), - cos(phi)*cos(psi) - sin(phi)*sin(psi)*sin(theta);
    #           0,                              cos(phi)*cos(theta),                               -cos(theta)*sin(phi)]*phidot + ...
    #         [ -cos(psi)*sin(theta), cos(psi)*cos(theta)*sin(phi), cos(phi)*cos(psi)*cos(theta);
    #           -sin(psi)*sin(theta), cos(theta)*sin(phi)*sin(psi), cos(phi)*cos(theta)*sin(psi);
    #                   -cos(theta),         -sin(phi)*sin(theta),         -cos(phi)*sin(theta)]*thetadot +
    #                  [ -cos(theta)*sin(psi), - cos(phi)*cos(psi) - sin(phi)*sin(psi)*sin(theta), cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta);
    #                    cos(psi)*cos(theta),   cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi), sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta);
    #                                      0,                                                  0,                                                0]*psidot;
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

    [xyz_ddot;rpy_ddot]

end

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

    # if(nargout>1)
    #     drotmat = zeros(9,3);
    #
    #     drotmat[1,2] = cos_y*-sin_p;
    #
    #     drotmat[1,3] = -sin_y*cos_p;
    #
    #     drotmat[2,2] = sin_y*-sin_p;
    #
    #     drotmat[2,3] = cos_y*cos_p;
    #
    #     drotmat[3,2] = -cos_p;
    #
    #     drotmat[4,1] = cos_y*sin_p*cos_r+sin_y*sin_r;
    #
    #     drotmat[4,2] = cos_y*cos_p*sin_r;
    #
    #     drotmat[4,3] = -sin_y*sin_p*sin_r-cos_y*cos_r;
    #
    #     drotmat[5,1] = sin_y*sin_p*cos_r-cos_y*sin_r;
    #
    #     drotmat[5,2] = sin_y*cos_p*sin_r;
    #
    #     drotmat[5,3] = cos_y*sin_p*sin_r-sin_y*cos_r;
    #
    #     drotmat[6,1] = cos_p*cos_r;
    #
    #     drotmat[6,2] = -sin_p*sin_r;
    #
    #     drotmat[7,1] = cos_y*sin_p*-sin_r+sin_y*cos_r;
    #
    #     drotmat[7,2] = cos_y*cos_p*cos_r;
    #
    #     drotmat[7,3] = -sin_y*sin_p*cos_r+cos_y*sin_r;
    #
    #     drotmat[8,1] = sin_y*sin_p*-sin_r-cos_y*cos_r;
    #
    #     drotmat[8,2] = sin_y*cos_p*cos_r;
    #
    #     drotmat[8,3] = cos_y*sin_p*cos_r+sin_y*sin_r;
    #
    #     drotmat[9,1] = -cos_p*sin_r;
    #
    #     drotmat[9,2] = -sin_p*cos_r;
    # end

    # if(nargout>2)
    #     rr_idx = 1;
    #     rp_idx = 2;
    #     ry_idx = 3;
    #     pr_idx = 4;
    #     pp_idx = 5;
    #     py_idx = 6;
    #     yr_idx = 7;
    #     yp_idx = 8;
    #     yy_idx = 9;
    #
    #     #ddrotmat = sparse(9,9);
    #     ddrotmat = zeros(9,9);
    #
    #     ddrotmat[1,yy_idx] = -cos_y*cos_p;
    #
    #     ddrotmat[1,yp_idx] = sin_y*sin_p;
    #
    #     ddrotmat[1,pp_idx] = -cos_y*cos_p;
    #
    #     ddrotmat[1,py_idx] = sin_y*sin_p;
    #
    #     ddrotmat[2,yy_idx] = -sin_y*cos_p;
    #
    #     ddrotmat[2,yp_idx] = -cos_y*sin_p;
    #
    #     ddrotmat[2,py_idx] = -cos_y*sin_p;
    #
    #     ddrotmat[2,pp_idx] = -sin_y*cos_p;
    #
    #     ddrotmat[3,pp_idx] = sin_p;
    #
    #     ddrotmat[4,rr_idx] = -cos_y*sin_p*sin_r+sin_y*cos_r;
    #
    #     ddrotmat[4,rp_idx] = cos_y*cos_p*cos_r;
    #
    #     ddrotmat[4,ry_idx] = -sin_y*sin_p*cos_r+cos_y*sin_r;
    #
    #     ddrotmat[4,pr_idx] = cos_y*cos_p*cos_r;
    #
    #     ddrotmat[4,pp_idx] = -cos_y*sin_p*sin_r;
    #
    #     ddrotmat[4,py_idx] = -sin_y*cos_p*sin_r;
    #
    #     ddrotmat[4,yr_idx] = -sin_y*sin_p*cos_r+cos_y*sin_r;
    #
    #     ddrotmat[4,yp_idx] = -sin_y*cos_p*sin_r;
    #
    #     ddrotmat[4,yy_idx] = -cos_y*sin_p*sin_r+sin_y*cos_r;
    #
    #     ddrotmat[5,rr_idx] = -sin_y*sin_p*sin_r-cos_y*cos_r;
    #
    #     ddrotmat[5,rp_idx] = sin_y*cos_p*cos_r;
    #
    #     ddrotmat(5,ry_idx) = cos_y*sin_p*cos_r+sin_y*sin_r;
    #
    #     ddrotmat(5,pr_idx) = sin_y*cos_p*cos_r;
    #
    #     ddrotmat(5,pp_idx) = -sin_y*sin_p*sin_r;
    #
    #     ddrotmat(5,py_idx) = cos_y*cos_p*sin_r;
    #
    #     ddrotmat(5,yr_idx) = cos_y*sin_p*cos_r+sin_y*sin_r;
    #
    #     ddrotmat(5,yp_idx) = cos_y*cos_p*sin_r;
    #
    #     ddrotmat(5,yy_idx) = -sin_y*sin_p*sin_r-cos_y*cos_r;
    #
    #     ddrotmat(6,rr_idx) = -cos_p*sin_r;
    #
    #     ddrotmat(6,rp_idx) = -sin_p*cos_r;
    #
    #     ddrotmat(6,pr_idx) = -sin_p*cos_r;
    #
    #     ddrotmat(6,pp_idx) = -cos_p*sin_r;
    #
    #     ddrotmat(7,rr_idx) = -cos_y*sin_p*cos_r-sin_y*sin_r;
    #
    #     ddrotmat(7,rp_idx) = -cos_y*cos_p*sin_r;
    #
    #     ddrotmat(7,ry_idx) = sin_y*sin_p*sin_r+cos_y*cos_r;
    #
    #     ddrotmat(7,pr_idx) = -cos_y*cos_p*sin_r;
    #
    #     ddrotmat(7,pp_idx) =  -cos_y*sin_p*cos_r;
    #
    #     ddrotmat(7,py_idx) =  -sin_y*cos_p*cos_r;
    #
    #     ddrotmat(7,yr_idx) = sin_y*sin_p*sin_r+cos_y*cos_r;
    #
    #     ddrotmat(7,yp_idx) =  -sin_y*cos_p*cos_r;
    #
    #     ddrotmat(7,yy_idx) = -cos_y*sin_p*cos_r-sin_y*sin_r;
    #
    #     ddrotmat(8,rr_idx) = -sin_y*sin_p*cos_r+cos_y*sin_r;
    #
    #     ddrotmat(8,rp_idx) = -sin_y*cos_p*sin_r;
    #
    #     ddrotmat(8,ry_idx) = -cos_y*sin_p*sin_r+sin_y*cos_r;
    #
    #     ddrotmat(8,pr_idx) = -sin_y*cos_p*sin_r;
    #
    #     ddrotmat(8,pp_idx) =  -sin_y*sin_p*cos_r;
    #
    #     ddrotmat(8,py_idx) = cos_y*cos_p*cos_r;
    #
    #     ddrotmat(8,yr_idx) =  -cos_y*sin_p*sin_r+sin_y*cos_r;
    #
    #     ddrotmat(8,yp_idx) = cos_y*cos_p*cos_r;
    #
    #     ddrotmat(8,yy_idx) = -sin_y*sin_p*cos_r+cos_y*sin_r;
    #
    #     ddrotmat(9,rr_idx) =  -cos_p*cos_r;
    #
    #     ddrotmat(9,rp_idx) = sin_p*sin_r;
    #
    #     ddrotmat(9,pr_idx) = sin_p*sin_r;
    #
    #     ddrotmat(9,pp_idx) = -cos_p*cos_r;
    # end
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

    # if(nargout <= 1)
    E = rpydot2angularvelMatrix(rpy);
    omega = E*rpydot;
    # else
    #     [E,dE] = rpydot2angularvelMatrix(rpy);
    #     omega = E*rpydot;
    # end
    # domega = [matGradMult(dE,rpydot) E];
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

    # if compute_gradient
      sp2 = sp^2;
      cp2 = cp^2;
      dPhi = [
        0 (cy*sp)/cp2 -sy/cp;
        0 0 -cy;
        0 cy + (cy*sp2)/cp2 -(sp*sy)/cp;
        0 (sp*sy)/cp2 cy/cp;
        0 0 -sy;
        0 (sy + (sp2*sy)/cp2)  (cy*sp)/cp;
        0 0 0;
        0 0 0;
        0 0 0];
    # end

    # if compute_second_deriv
    #   cp3 = cp2 * cp;
    #   ddPhi = [
    #     0 0 0;
    #     0 0 0;
    #     0 0 0;
    #     0 0 0;
    #     0 0 0;
    #     0 0 0;
    #     0 0 0;
    #     0 0 0;
    #     0 0 0;
    #
    #     0 -(cy*(cp2 - 2))/cp3 (sp*sy)/(sp2 - 1);
    #     0 0 0;
    #     0 (2*cy*sp)/cp3 sy/(sp2 - 1);
    #     0 (2*sy - cp2*sy)/cp3 (cy*sp)/cp2;
    #     0 0 0;
    #     0 (2*sp*sy)/cp3 cy/cp2;
    #     0 0 0;
    #     0 0 0;
    #     0 0 0;
    #     0 (sp*sy)/(sp2 - 1) -cy/cp;
    #     0 0 sy;
    #     0 sy/(sp2 - 1) -(cy*sp)/cp;
    #     0 (cy*sp)/cp2 -sy/cp;
    #     0 0 -cy;
    #     0 cy/cp2 -(sp*sy)/cp;
    #     0 0 0;
    #     0 0 0;
    #     0 0 0]
    #
    #     # ddPhi = reshape(ddPhi, numel(Phi), []); # to match geval output
    # end
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
# if(nargout>1)
#   rows = zeros(7,1);
#   cols = zeros(7,1);
#   vals = zeros(7,1);
#   rows(1) = 1; cols(1) = 2; vals(1) = -sin_p*cos_y;
#   rows(2) = 1; cols(2) = 3; vals(2) = -cos_p*sin_y;
#   rows(3) = 2; cols(3) = 2; vals(3) = -sin_p*sin_y;
#   rows(4) = 2; cols(4) = 3; vals(4) = cos_p*cos_y;
#   rows(5) = 3; cols(5) = 2; vals(5) = -cos_p;
#   rows(6) = 4; cols(6) = 3; vals(6) = -cos_y;
#   rows(7) = 5; cols(7) = 3; vals(7) = -sin_y;
#   dE = sparse(rows,cols,vals,9,3,7);
end
