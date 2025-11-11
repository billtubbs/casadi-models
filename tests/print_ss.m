function print_ss(sys)
  %PRINT_SS Print state-space matrices (A, B, C, D) with fixed formatting
  %
  %   sys must be a struct or object with fields A, B, C, D 
  %   like models from ss(), arx(), etc.

  matrices = {'A', 'B', 'C', 'D'};

  for i = 1:numel(matrices)
    name = matrices{i};
    M = sys.(lower(name));
    fprintf('\n%s =\n', name);
    for r = 1:size(M, 1)
      fprintf('  ');
      fprintf('%10.6f  ', M(r, :));
      fprintf('\n');
    end
  end
end