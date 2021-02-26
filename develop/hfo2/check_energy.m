clear; close all;
load data_mono0.txt; load a.txt; factor=1+fliplr(a);
type = data_mono0(:,1);
q = zeros(768,1);
for n =1:768
    if type(n)==0
        q(n)=2.4;
    else
        q(n)=-1.2;
    end
end
type=type+1;
r0 = data_mono0(:, 2:4);
box0=[20.564200    0.000000     -0.089970    
0.000000     20.781044    0.000000     
-3.487025    0.000000     21.019901 ].'  ;

energy_matlab = zeros(length(factor), 1);
for n = 1:length(factor)
    box = box0*factor(n);
    r = r0 * factor(n);
    energy_matlab(n) = find_E(box, r, type, q);
    disp(factor(n));
    disp(energy_matlab(n));
end

load energy.out;

figure;
plot(factor, energy_matlab, 'o');
hold on;
plot(factor, energy(1:33,1)/768,'x');
plot(factor, energy(1:33,2)/768,'s');
legend('matlab', 'GPUGA', 'DFT')
