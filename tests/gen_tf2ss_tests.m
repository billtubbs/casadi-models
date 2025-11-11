% Three examples of tf2ss conversion using Octave

% System 1
num = [0.2 0.1];
den = [1 -1.4 0.49];
Ts = 0.5;
sys = ss(tf(num,den,Ts));

abs_tol = 0.0001;

% Check matrices
assert(all(
    abs(sys.a -
       [0.0000    0.4900;  
       -1.0000    1.4000]
    ) < abs_tol
))
assert(all(
    abs(sys.b -
      [-0.1000;
        0.2000]
    ) < abs_tol
))
assert(all(
    abs(sys.c - 
        [0.0000    1.0000]
    ) < abs_tol
))
assert(abs(sys.d) < abs_tol)


% System 2
num = [0.05 0.1 0.05];
den = [1 -1.5 0.7 -0.1];
Ts = 0.5;
sys = ss(tf(num,den,Ts));

% Check matrices
assert(all(
    abs(sys.a -
       [0.0000  -0.0000  -0.1000;
       -1.0000  -0.0000  -0.7000;
             0   1.0000   1.5000]
    ) < abs_tol
))
assert(all(
    abs(sys.b -
      [-0.050000;
        0.100000;
        0.050000]
    ) < abs_tol
))
assert(all(
    abs(sys.c -
       [0.0000    0.0000    1.0000]
    ) < abs_tol
))
assert(abs(sys.d) < abs_tol)


% System 3
num = [0.01 0.03 0.03 0.02 0.01];
den = [1 -2.3 2.6 -1.8 0.72 -0.12];
Ts = 0.5;
sys = ss(tf(num,den,Ts));

% Check matrices
assert(all(
    abs(sys.a -
       [0.0000  -0.0000   0.0000  -0.0000  -0.1200;
        1.0000  -0.0000  -0.0000  -0.0000   0.7200;
             0  -1.0000  -0.0000  -0.0000   1.8000;
             0        0   1.0000  -0.0000  -2.6000;
             0        0        0   1.0000   2.3000]
    ) < abs_tol
))
assert(all(
    abs(sys.b -
      [-0.010000;
       -0.020000;
        0.030000;
        0.030000;
        0.010000]
    ) < abs_tol
))
assert(all(
    abs(sys.c -
       [0.0000    0.0000    0.0000    0.0000    1.0000]
   ) < abs_tol
))
assert(abs(sys.d) < abs_tol)

% System 4
num = [1];
den = [1 -1];
Ts = 0.5;
sys = ss(tf(num, den, Ts));

% Check matrices
assert(abs(sys.a - 1) < abs_tol)
assert(abs(sys.b - 1) < abs_tol)
assert(abs(sys.c - 1) < abs_tol)
assert(abs(sys.d) < abs_tol)


% System 5
z = [];   % no finite zeros
p = [1, 1, 0.6, 0.5, ...
     exp(1j*pi/4), exp(-1j*pi/4)];
k = 0.1;   % gain
Ts = 1;
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

sys = ss(Gd);

% Check matrices
assert(all(
    abs(sys.a - [
    0.000000    0.000000    0.000000    0.000000    0.000000   -0.300000; 
   -0.100000    0.000000    0.000000    0.000000    0.000000   -0.212426;
    0.000000   -1.000000    0.000000    0.000000    0.000000   -0.620416;
    0.000000    0.000000    1.000000    0.000000    0.000000    0.974975; 
    0.000000    0.000000    0.000000   -1.000000    0.000000    0.888406; 
    0.000000    0.000000    0.000000    0.000000  -10.000000    4.514214;
    ]) < abs_tol
))
assert(all(
    abs(sys.b -
      [-0.1000
        0
        0
        0
        0
        0]
    ) < abs_tol
))
assert(all(
    abs(sys.c -
       [0.0000    0.0000    0.0000    0.0000   0.0000   -1.0000]
    ) < abs_tol
))
assert(abs(sys.d) < abs_tol)
