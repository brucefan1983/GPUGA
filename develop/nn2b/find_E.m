function [energy, virial, force] = find_E(r, box)
N = size(r, 1);
energy = 0;
virial = zeros(6, 1);
force = zeros(N, 3);
for n1=1:N-1
    for n2=(n1+1):N
        r12=r(n2,:)-r(n1,:);                  % position difference vector
        r12=r12.';                            % column vector
        r12=box\r12;                          % transform to cubic box
        r12=r12-round(r12);                   % mininum image convention
        r12=box*r12;                          % transform back
        d12=sqrt(sum(r12.*r12));              % distance
        if d12 > 5
            continue;
        end
        energy = 0.5*(d12-2)^2 + 1/6*(d12-2)^3 - 0.25;
        f2 = ((d12-2) + 0.5*(d12-2)^2)/d12;
        force(n1, :) = force(n1, :) + f2 * r12.';
        force(n2, :) = force(n2, :) - f2 * r12.';
        virial(1) = virial(1) - r12(1) * r12(1) * f2;
        virial(2) = virial(2) - r12(2) * r12(2) * f2;
        virial(3) = virial(3) - r12(3) * r12(3) * f2;
        virial(4) = virial(4) - r12(1) * r12(2) * f2;
        virial(5) = virial(5) - r12(2) * r12(3) * f2;
        virial(6) = virial(6) - r12(3) * r12(1) * f2;
    end
end

