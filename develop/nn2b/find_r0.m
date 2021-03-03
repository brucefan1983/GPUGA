function [r, box] = find_r0()
r0 = [0.0, 0.0, 0.5, 0.5; ...
      0.0, 0.5, 0.0, 0.5; ...
      0.0, 0.5, 0.5, 0.0].';
a = 5.284;
box = [1, 0, 0;
       0, 1, 0;
       0, 0, 1] * a * 4;

r = zeros(256, 3);
n = 0;
for nx = 0 : 4 - 1
    for ny = 0 : 4 - 1
        for nz = 0 : 4 - 1
            for m = 1 : 4
                n = n + 1;
                r(n, :) = a .* ([nx,ny,nz] + r0(m, :));   
            end
        end
    end
end

