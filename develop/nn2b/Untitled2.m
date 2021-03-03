clear; close all;
epsilon = 1.032e-2;
epsilon4 = epsilon * 4;
sigma = 3.405;
sigma6 = sigma^6;
sigma12 = sigma6 * sigma6;
rc = 8;
energy_cut = epsilon4 * (sigma12 / rc^12 - sigma6 / rc^6);
d_energy_cut = epsilon4 * (6 * sigma6 / rc^7 - 12 * sigma12 / rc^13);

d12=3:0.1:8;
N=length(d12);
p2=zeros(N,1);
p2lj=zeros(N,1);
f2=zeros(N,1);
for n = 1:N
    [p2(n), f2(n)] = find_p2_and_f2(d12(n));
    
    if d12(n) > 7
        p2(n) = p2(n) * (0.5 * cos(pi * (d12(n) - 7)) + 0.5);
    end
    
    d12inv2 = 1 / (d12(n) * d12(n));
    d12inv6 = d12inv2 * d12inv2 * d12inv2;
    d12inv8 = d12inv2 * d12inv6;
    d12inv12 = d12inv6 * d12inv6;
    d12inv14 = d12inv2 * d12inv12;
    p2lj(n) = epsilon4 * (sigma12 * d12inv12 - sigma6 * d12inv6) ...
        - energy_cut - d_energy_cut * (d12(n) - rc);
end

figure;
plot(d12, p2,'x');
hold on;
plot(d12, p2lj,'o');
