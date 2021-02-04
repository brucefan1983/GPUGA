clear; close all;
load a.txt; a=(1-a);
load energy.out;
load virial.out;
load force.out;
energy=energy/768; % eV/atom
virial=virial/768; % eV/atom
virial_xx=virial(0*end/6+1:1*end/6,:);
virial_yy=virial(1*end/6+1:2*end/6,:);
virial_zz=virial(2*end/6+1:3*end/6,:);
figure;
subplot(2,2,1)
plot(a,energy(:,2),'ro','linewidth',1);
hold on;
plot(a,energy(:,1),'b-','linewidth',2);
xlabel('$a/a_0$','fontsize',12,'interpreter','latex');
ylabel('E (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
xlim([0.94,1.06]);
ylim([-11,-10.6]);

subplot(2,2,2) 
plot(a,virial_xx(:,2),'ro','linewidth',1);
hold on;
plot(a,virial_xx(:,1),'b-','linewidth',2);
xlabel('$a/a_0$','fontsize',12,'interpreter','latex');
ylabel('$\sigma_{xx}$ (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
xlim([0.94,1.06]);
ylim([-4,4]);

subplot(2,2,3) 
plot(a,virial_yy(:,2),'ro','linewidth',1);
hold on;
plot(a,virial_yy(:,1),'b-','linewidth',2);
xlabel('$a/a_0$','fontsize',12,'interpreter','latex');
ylabel('$\sigma_{yy}$ (eV)','fontsize',12,'interpreter','latex');
xlim([0.94,1.06]);
ylim([-4,4]);

subplot(2,2,4) 
plot(a,virial_zz(:,2),'ro','linewidth',1);
hold on;
plot(a,virial_zz(:,1),'b-','linewidth',2);
xlabel('$a/a_0$','fontsize',12,'interpreter','latex');
ylabel('$\sigma_{zz}$ (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
xlim([0.94,1.06]);
ylim([-4,4]);