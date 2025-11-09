%% ARX model identification benchmark data
% Fits an ARX model to system identificaton data

pkg load control

% Data from GEL-7017 course HW4
data = load('data/TP04_Q1a.mat');

% Convert to iddata set
Ts = 2;
data = iddata(data.y, data.u, Ts);

% Model dimensions
na = 2;
nb = 2;
nk = 1;

% Identify ARX transfer function model
sys = arx(data, 'na', na, 'nb', nb, 'nk', nk);

% Simulate model
t = Ts * (0:nT-1)';
u = data.u{1};
e = zeros(size(u));
[y, t, x] = lsim(sys, [u e], t);

display(sys)

figure()
stairs(t, u, '-'); hold on
plot(t, y, 'o-')
xlabel('Time, t (mins)')
ylabel('u(t), y(t)')
title('Transfer Function Model')
legend('u(t)', 'y(t)')
grid on

% When x0 is returned, a state-space model is returned
% (observer canonical form?)
[sys_ss, x0] = arx(data, 'na', na, 'nb', nb, 'nk', nk);

display(sys_ss)

% Simulate state space model
t = Ts * (0:nT-1)';
u = data.u{1};
[y, t, x] = lsim(sys_ss, u, t);

figure()
stairs(t, u, '-'); hold on
plot(t, y, 'o-')
xlabel('Time, t (mins)')
ylabel('u(t), y(t)')
title('State Space Model')
legend('u(t)', 'y(t)')
grid on
