%% ARX model identification benchmark data
% Fits an ARX model to system identificaton data

pkg load control

% Data from GEL-7017 course HW4
data = load('data/TP04_Q1a.mat');
nT = size(data.u, 1);
assert(size(data.y, 1) == nT)

% Sampling period
Ts = 2;
t = Ts * (0:nT-1)';

% Model dimensions
na = 2;
nb = 2;
nk = 1;

% Identify ARX transfer function model
id_data = iddata(data.y, data.u, Ts);
sys = arx(id_data, 'na', na, 'nb', nb, 'nk', nk);

% Simulate model
e = zeros(size(data.u));
[y, t, x] = lsim(sys, [data.u e], t);

display(sys)

% Save data and results to CSV file
fid = fopen('data/TP04_Q1a.csv', 'w');
fprintf(fid, 't,u_data,x1,x2,x3,y,y_data\n');
fclose(fid);
dlmwrite('data/TP04_Q1a.csv', [t data.u x y data.y], '-append');

figure()
stairs(t, data.u, '-'); hold on
plot(t, data.y, 'o'); hold on
plot(t, y, '.-')
xlabel('Time, t (mins)')
ylabel('u(t), y(t)')
title('Transfer Function Model')
legend('u(t)', 'y_m(t)', 'y(t)')
grid on

% Print with more decimal places
fprintf("\nTransfer Function Parameters\n")
fprintf("Numerator:\n")
fprintf("%12.8f %12.8f %12.8f %12.8f\n", sys.num{1})
fprintf("Denomenator:\n")
fprintf("%12.8f %12.8f %12.8f %12.8f\n", sys.den{1})

% When x0 is returned, a state-space model is identified
% (observer canonical form?)
[sys_ss, x0] = arx(id_data, 'na', na, 'nb', nb, 'nk', nk);

fprintf("Initial state: [%f %f %f]", x0)
display(sys_ss)

% Simulate state space model
t = Ts * (0:nT-1)';
[y, t, x] = lsim(sys_ss, data.u, t);

figure()
stairs(t, data.u, '-'); hold on
plot(t, data.y, 'o'); hold on
plot(t, y, '.-')
xlabel('Time, t (mins)')
ylabel('u(t), y(t)')
title('State Space Model')
legend('u(t)', 'y_m(t)', 'y(t)')
grid on

% Save data and results to CSV file
fid = fopen('data/TP04_Q1a_ss.csv', 'w');
fprintf(fid, 't,u_data,x1,x2,x3,y,y_data\n');
fclose(fid);
dlmwrite('data/TP04_Q1a_ss.csv', [t data.u x y data.y], '-append');

% Print with more decimal places
fprintf("\nState Space Model Matrices\n")
fprintf("A:\n")
fprintf("%12.8f %12.8f %12.8f\n", sys_ss.a)
fprintf("B:\n")
fprintf("%12.8f\n", sys_ss.b)
fprintf("C:\n")
fprintf("%12.8f %12.8f %12.8f\n", sys_ss.c)
fprintf("D:\n")
fprintf("%12.8f\n", sys_ss.d)