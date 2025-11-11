% Three examples of tf2ss conversion using Octave

pkg load control

% Sampling period
Ts = 0.5;

% System 1
num = [0.2 0.1];
den = [1 -1.4 0.49];
sys1 = ss(tf(num,den,Ts));

abs_tol = 0.0001;

% Check matrices
assert(all(all(
    abs(sys1.a -
       [0.0000    0.4900;
       -1.0000    1.4000]
    ) < abs_tol
)))
assert(all(
    abs(sys1.b -
      [-0.1000;
        0.2000]
    ) < abs_tol
))
assert(all(
    abs(sys1.c -
        [0.0000    1.0000]
    ) < abs_tol
))
assert(abs(sys1.d) < abs_tol)

% System 2
num = [0.05 0.1 0.05];
den = [1 -1.5 0.7 -0.1];
sys2 = ss(tf(num,den,Ts));

% Check matrices
assert(all(all(
    abs(sys2.a -
       [0.0000  -0.0000  -0.1000;
       -1.0000  -0.0000  -0.7000;
             0   1.0000   1.5000]
    ) < abs_tol
)))
assert(all(
    abs(sys2.b -
      [-0.050000;
        0.100000;
        0.050000]
    ) < abs_tol
))
assert(all(
    abs(sys2.c -
       [0.0000    0.0000    1.0000]
    ) < abs_tol
))
assert(abs(sys2.d) < abs_tol)


% System 3
num = [0.01 0.03 0.03 0.02 0.01];
den = [1 -2.3 2.6 -1.8 0.72 -0.12];
sys3 = ss(tf(num,den,Ts));

% Check matrices
assert(all(all(
    abs(sys3.a -
       [0.0000  -0.0000   0.0000  -0.0000  -0.1200;
        1.0000  -0.0000  -0.0000  -0.0000   0.7200;
             0  -1.0000  -0.0000  -0.0000   1.8000;
             0        0   1.0000  -0.0000  -2.6000;
             0        0        0   1.0000   2.3000]
    ) < abs_tol
)))
assert(all(
    abs(sys3.b -
      [-0.010000;
       -0.020000;
        0.030000;
        0.030000;
        0.010000]
    ) < abs_tol
))
assert(all(
    abs(sys3.c -
       [0.0000    0.0000    0.0000    0.0000    1.0000]
   ) < abs_tol
))
assert(abs(sys3.d) < abs_tol)

% System 4
num = [1];
den = [1 -1];
sys4 = ss(tf(num, den, Ts));

% Check matrices
assert(abs(sys4.a - 1) < abs_tol)
assert(abs(sys4.b - 1) < abs_tol)
assert(abs(sys4.c - 1) < abs_tol)
assert(abs(sys4.d) < abs_tol)


% System 5
z = [];   % no finite zeros
p = [1, 1, 0.6, 0.5, ...
     exp(1j*pi/4), exp(-1j*pi/4)];
k = 0.1;   % gain
Gd = zpk(z, p, k, Ts);

% Coefficients
assert(all(
    abs(Gd.num{1} -
        [0        0        0        0        0        0   0.1000]
    ) < abs_tol
))
assert(all(
    abs(Gd.den{1} -
        [1.000000  -4.514214   8.884062  -9.749747   6.204163  -2.124264   0.300000]
    ) < abs_tol
))

sys5 = ss(Gd);

% Check matrices
assert(all(all(
    abs(sys5.a - [
    0.000000    0.000000    0.000000    0.000000    0.000000   -0.300000;
   -0.100000    0.000000    0.000000    0.000000    0.000000   -0.212426;
    0.000000   -1.000000    0.000000    0.000000    0.000000   -0.620416;
    0.000000    0.000000    1.000000    0.000000    0.000000    0.974975;
    0.000000    0.000000    0.000000   -1.000000    0.000000    0.888406;
    0.000000    0.000000    0.000000    0.000000  -10.000000    4.514214;
    ]) < abs_tol
)))
assert(all(
    abs(sys5.b -
      [-0.1000
        0
        0
        0
        0
        0]
    ) < abs_tol
))
assert(all(
    abs(sys5.c -
       [0.0000    0.0000    0.0000    0.0000   0.0000   -1.0000]
    ) < abs_tol
))
assert(abs(sys5.d) < abs_tol)

systems = {sys1, sys2, sys3, sys4, sys5};


% Simulate all systems
nT = 30;
t = Ts * (0:nT)';
u = zeros(nT+1, 1);
u(t >= 1) = 1;

% Initialize combined results with time and input
combined_results = [t u];
headers = {'t', 'u'};

% Loop through each system
for i = 1:length(systems)
    sys = systems{i};

    % Simulate system
    [y, t_out, x] = lsim(sys, u, t);

    % Add states and output to combined results
    combined_results = [combined_results x y];

    % Generate headers for this system
    n_states = size(x, 2);
    for j = 1:n_states
        headers{end+1} = sprintf('sys%d_x%d', i, j);
    end
    headers{end+1} = sprintf('sys%d_y', i);
end

% Save results to CSV file
fid = fopen('data/results_tf2ss.csv', 'w');
fprintf(fid, '%s', headers{1});
for i = 2:length(headers)
    fprintf(fid, ',%s', headers{i});
end
fprintf(fid, '\n');
fclose(fid);
dlmwrite('data/results_tf2ss.csv', combined_results, '-append');
