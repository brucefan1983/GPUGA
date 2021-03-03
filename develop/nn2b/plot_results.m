clear; %close all;
load energy.out;
load virial.out;
load force.out;
N=2;

figure;
subplot(1,2,1);
plot(force(:,4),force(:,1),'bo');hold on;
hold on;
plot(-1.5:0.01:1.5,-1.5:0.01:1.5,'r--','linewidth',1.5);
xlabel('Training force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NN2B force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

strain = -0.5:0.005:0.5;
subplot(1,2,2);
plot(energy(:,2),energy(:,1),'bo','linewidth',1);hold on;
plot(-.5:0.01:.5,-.5:0.01:.5,'r--','linewidth',1.5);
xlabel('Training Energy (eV)','fontsize',12,'interpreter','latex');
ylabel('NN2B Energy (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);



