%% Robot Modelling

% Link length UR5
L1 = 0.0892;
L2 = 0.1359;
L3 = 0.4250;
L4 = 0.1197;
L5 = 0.3923;
L6 = 0.0930;
L7 = 0.0947;
L8 = 0.0823;

% d
vd1 = L1;
vd2 = 0;
vd3 = 0;
vd4 = L2-L4+L6;
vd5 = L7;
vd6 = L8;

% a
va1 = 0;
va2 = -L3;
va3 = -L5;
va4 = 0;
va5 = 0;
va6 = 0;

% alpha
valpha1 = pi/2;
valpha2 = 0;
valpha3 = 0;
valpha4 = pi/2;
valpha5 = -pi/2;
valpha6 = 0;

% L = Link(theta, d, a, alpha); - DH parameters
L(1) = Link([0  vd1  va1  valpha1]); % L(1) = Link([0        L1   0  pi/2]);
L(2) = Link([0  vd2  va2  valpha2]); % L(2) = Link([0         0 -L3     0]); offset=-pi/2
L(3) = Link([0  vd3  va3  valpha3]); % L(3) = Link([0         0 -L5     0]);
L(4) = Link([0  vd4  va4  valpha4]); % L(4) = Link([0  L2-L4+L6   0  pi/2]); offset=-pi/2
L(5) = Link([0  vd5  va5  valpha5]); % L(5) = Link([0        L7   0 -pi/2]);
L(6) = Link([0  vd6  va6  valpha6]); % L(6) = Link([0        L8   0     0]);
UR5 = SerialLink(L, 'name', 'UR5')

% % Plot configuration
% UR5.plot([0 0 0 0 0 0])

%% Forward Kinematics

% joint variables
syms qi
syms q1
syms q2
syms q3
syms q4
syms q5
syms q6
assume(qi,'real')
assume(q1,'real')
assume(q2,'real')
assume(q3,'real')
assume(q4,'real')
assume(q5,'real')
assume(q6,'real')

% link lengths
syms ai
syms a1
syms a2
syms a3
syms a4
syms a5
syms a6
assume(ai,'real')
assume(a1,'real')
assume(a2,'real')
assume(a3,'real')
assume(a4,'real')
assume(a5,'real')
assume(a6,'real')

% link offsets
syms di
syms d1
syms d2
syms d3
syms d4
syms d5
syms d6
assume(di,'real')
assume(d1,'real')
assume(d2,'real')
assume(d3,'real')
assume(d4,'real')
assume(d5,'real')
assume(d6,'real')

% link twists
syms alphai
syms alpha1
syms alpha2
syms alpha3
syms alpha4
syms alpha5
syms alpha6
assume(alphai,'real')
assume(alpha1,'real')
assume(alpha2,'real')
assume(alpha3,'real')
assume(alpha4,'real')
assume(alpha5,'real')
assume(alpha6,'real')

% joint offsets
syms offseti
syms offset1
syms offset2
syms offset3
syms offset4
syms offset5
syms offset6
assume(offseti,'real')
assume(offset1,'real')
assume(offset2,'real')
assume(offset3,'real')
assume(offset4,'real')
assume(offset5,'real')
assume(offset6,'real')

% Generic formula using DH parameters
A01i = [cos(qi+offseti) -sin(qi+offseti)*cos(alphai)  sin(qi+offseti)*sin(alphai) ai*cos(qi+offseti);...
       sin(qi+offseti)  cos(qi+offseti)*cos(alphai) -cos(qi+offseti)*sin(alphai) ai*sin(qi+offseti);...
          0     sin(alphai)          cos(alphai)         di;...
          0    0     0    1]

% theta
vtheta1 = 0;
vtheta2 = 0; % -pi/2
vtheta3 = 0;
vtheta4 = 0; % -pi/2
vtheta5 = 0;
vtheta6 = 0;

% A matrices (numerical expression)
A01 = subs(A01i,{ai,di,alphai,offseti,qi},{ va1,   vd1,  valpha1, vtheta1,  q1});
A12 = subs(A01i,{ai,di,alphai,offseti,qi},{ va2,   vd2,  valpha2, vtheta2,  q2});
A23 = subs(A01i,{ai,di,alphai,offseti,qi},{ va3,   vd3,  valpha3, vtheta3,  q3});
A34 = subs(A01i,{ai,di,alphai,offseti,qi},{ va4,   vd4,  valpha4, vtheta4,  q4});
A45 = subs(A01i,{ai,di,alphai,offseti,qi},{ va5,   vd5,  valpha5, vtheta5,  q5});
A56 = subs(A01i,{ai,di,alphai,offseti,qi},{ va6,   vd6,  valpha6, vtheta6,  q6});

% Forward Kinematics
A06 = simplify(A01*A12*A23*A34*A45*A56)

%% Inverse Kinematics - Closed form computation

qa = [pi/4,pi/3,pi/2,pi/4,pi/3,pi/2];
T06 = double(subs(A06,{q1,q2,q3,q4,q5,q6},{qa})) % tfs_ee

% Previous determinations
px = T06(1,4);
py = T06(2,4);
pz = T06(3,4);
r11 = T06(1,1); r12 = T06(1,2); r13 = T06(1,3);
r21 = T06(2,1); r22 = T06(2,2); r23 = T06(2,3);
r31 = T06(3,1); r32 = T06(3,2); r33 = T06(3,3);

% theta 1
ikA = py-vd6*r23;
ikB = px-vd6*r13;
ikq1 = atan2(sqrt(ikB^2+(-ikA)^2-vd4^2),vd4)+atan2(ikB,-ikA); % +- (1)
% theta 5
ikC = sin(ikq1)*r11-cos(ikq1)*r21;
ikD = cos(ikq1)*r22-sin(ikq1)*r12;
ikq5 = atan2(sqrt(ikC^2+ikD^2),sin(ikq1)*r13-cos(ikq1)*r23); % +-
% theta 6
ikq6 = atan2(ikD/sin(ikq5),ikC/sin(ikq5));
% theta 3
ikE = cos(ikq1)*r11+sin(ikq1)*r21;
ikF = cos(ikq5)*cos(ikq6);
ikqaux = atan2(r31*ikF-sin(ikq6)*ikE,ikF*ikE+sin(ikq6)*r31);
ikPC = cos(ikq1)*px+sin(ikq1)*py-sin(ikqaux)*vd5+cos(ikqaux)*sin(ikq5)*vd6;
ikPS = pz-vd1+cos(ikqaux)*vd5+sin(ikqaux)*sin(ikq5)*vd6;
ikq3 = atan2(sqrt(1-((ikPS^2+ikPC^2-va2^2-va3^2)/(2*va2*va3))^2),(ikPS^2+ikPC^2-va2^2-va3^2)/(2*va2*va3)); % +-
% theta 2
ikq2 = atan2(ikPS,ikPC)-atan2(sin(ikq3)*va3,cos(ikq3)*va3+va2);
% theta 4
ikq4 = ikqaux-ikq2-ikq3;

% Joint values
ikq = [ikq1 ikq2 ikq3 ikq4 ikq5 ikq6]

%% Inverse Kinematics - Closed form Algorithm 1

% Solution sets
sol = NaN(8,6);     % rows: number of solutions, cols: joint values

% Step 1 - Both q1 solutions are computed, and complex angles are discarded.
A = py-vd6*r23;
B = px-vd6*r13;
q1 = [];    % final size = 2 (number of diferent values)
q1_1 = atan2(sqrt(B^2+(-A)^2-vd4^2),vd4)+atan2(B,-A);
q1_2 = -atan2(sqrt(B^2+(-A)^2-vd4^2),vd4)+atan2(B,-A);
% Checking valid values of q1 (result inside sqrt != imginary)
if isreal(q1_1)
    q1 = [q1, q1_1];
    sol(1,1) = q1_1;
    sol(2,1) = q1_1;
    sol(3,1) = q1_1;
    sol(4,1) = q1_1;
else
    q1 = [q1, NaN];
end
if isreal(q1_2)
    q1 = [q1, q1_2];
    sol(5,1) = q1_2;
    sol(6,1) = q1_2;
    sol(7,1) = q1_2;
    sol(8,1) = q1_2;
else
    q1 = [q1, NaN];
end

% Step 2 - Compute q5. The sets containing values of q5 that are not considered valid are rejected.
q5 = [];    % final size = 4 (number of diferent values)
for i = size(q1,2)
    q1_i = q1(i);
    if isnan(q1_i)
        q5 = [q5,NaN(1,2)];
    else
        C = sin(q1_i)*r11-cos(q1_i)*r21;
        D = cos(q1_i)*r22-sin(q1_i)*r12;
        q5_1 = atan2(sqrt(C^2+D^2),sin(q1_i)*r13-cos(q1_I)*r23);
        q5_2 = -atan2(sqrt(C^2+D^2),sin(q1_i)*r13-cos(q1_I)*r23);
        % Checking valid values of q5 (real and |·|>1e-12)
        if isreal(q5_1) && (q5_1>1e-12)
            q5 = [q5, q5_1];
            sol(1+4*(i-1),5) = q5_1;
            sol(2+4*(i-1),5) = q5_1;
        else
            q5 = [q5, NaN];
        end
        if isreal(q5_2) && (q5_2>1e-12)
            q5 = [q5, q5_2];
            sol(3+4*(i-1),5) = q5_2;
            sol(4+4*(i-1),5) = q5_2;
        else
            q5 = [q5, NaN];
        end
    end
end

% Step 3 - q6 is computed for the remaining sets.
q6 = [];    % final size = 4 (number of diferent values)
for i = 1:size(q5,2)
    q5_i = q5(i);
    if isnan(q5_i)
        q6 = [q6,NaN];
    else
        q6_i = atan2(D/sin(q5_i),C/sin(q5_i));
        sol(1+2*(i-1),5) = q6_i;
        sol(2+2*(i-1),5) = q6_i;
    end
end


% Step 4 - q3 computed and verified. Again, the solutions with angles that are not acceptable are discarded.



% Step 5 - q2 and q4 computed, and the sets of angles that are not valid are rejected.



%  Step 6 - Solution with the minimal difference with respect to the current joint positions.


