clear; close all;
load ga.out;

figure;
loglog(ga(:,1),ga(:,2));
xlabel('generation');
ylabel('cost');
set(gca,'fontsize',12);

figure;
semilogx(ga(:,1),ga(:,3:end));
xlabel('generation');
ylabel('solution');
set(gca,'fontsize',12);