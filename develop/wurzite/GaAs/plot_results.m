clear; close all;

% get the outputs from GPUGA:
load energy.out;
load virial.out;
load force.out;

% number of atoms
num_atoms=ones(size(energy,1),1)*192;
num_atoms(end-2:end)=96; % last three polytypes

figure;

% force
subplot(2,2,1); 
plot(force(1:end,4),force(1:end,1),'d');
hold on;
plot(force(1:end,5),force(1:end,2),'s');
plot(force(1:end,6),force(1:end,3),'o');
plot(-1:0.01:1,-1:0.01:1,'--');
xlim([-1.1,1.1]);
ylim([-1.1,1.1]);
xlabel('Force (DFT) (eV/\AA)','fontsize',12,'interpreter','latex');
ylabel('Force (Potential) (eV/\AA)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('x','y','z');


% energy
subplot(2,2,2);
plot(energy(:,2)./num_atoms,'o');
hold on;
plot(energy(:,1)./num_atoms,'x','linewidth',2);
%ylim([-4.63,-3.8]);
xlabel('Configuration ID','fontsize',12,'interpreter','latex');
ylabel('Energy (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

% virial-xx
subplot(2,2,3);
n=1;
plot(virial((n-1)*end/6+1:n*end/6,2)./num_atoms,'o','linewidth',1);
hold on;
plot(virial((n-1)*end/6+1:n*end/6,1)./num_atoms,'x','linewidth',1);
ylim([-3,6]);
xlabel('Configuration ID','fontsize',12,'interpreter','latex');
ylabel('Virial-xx (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

% virial-xy
subplot(2,2,4);
n=4;
plot(virial((n-1)*end/6+1:n*end/6,2)./num_atoms,'o','linewidth',1);
hold on;
plot(virial((n-1)*end/6+1:n*end/6,1)./num_atoms,'x','linewidth',1);
ylim([-3,6]);
xlabel('Configuration ID','fontsize',12,'interpreter','latex');
ylabel('Virial-xy (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('DFT','Potential');

