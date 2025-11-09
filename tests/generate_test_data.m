%% Generates input-output data for unit tests

pkg load control

% Results will be saved to file
if ~exist('data', 'dir')
    mkdir('data');
end

% Simulation time
nT = 12;
Ts = 0.5;
t = Ts * (0:nT)';

% Input signal
u = zeros(nT+1, 1);
u(t >= 1) = 1;

G1 = tf(1, [1 1]);
G2 = tf(2, [2.5 1]);

% Underdamped second order system
zeta = 0.5;
omega = 1.6;
num = [omega^2];
den = [1 2*zeta*omega omega^2];
G3 = tf(num, den);

% PI controller
Kc = 0.5;
Ti = 2.5;
Gc = tf([Kc*Ti Kc], [Ti 0]);

systems = {
    G1,
    G2,
    G3,
    G1*G2,
    G1 * G2 + G3,
    feedback(Gc * G2, 1)
};

results = zeros(nT+1, numel(systems));
for i = 1:numel(systems)
    [y, t, x] = lsim(systems{i}, u, t);
    results(:, i) = y;
end

% Save results to CSV file
headers = {'t', 'u', 'y_G1', 'y_G2', 'y_G3', 'y_G1 * G2', 'G1 * G2 + G3', 'y_H2'};
fid = fopen('data/results_ct.csv', 'w');
fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s\n', headers{:});
fclose(fid);
dlmwrite('data/results_ct.csv', [t u results], '-append');
