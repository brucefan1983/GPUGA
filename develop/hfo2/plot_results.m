clear; 
load a.txt; a=(1-a);
load energy.out;
load virial.out;
load force.out;

figure;

subplot(1,3,1);
plot(reshape(force(:,4:6),768*3,1),reshape(force(:,1:3),768*3,1),'o');
xlim([-2,2]);
ylim([-2,2]);
xlabel('DFT force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('MBKS force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
title('(a)');

subplot(1,3,2);
plot(a,energy(1:33,2)/768,'rd','linewidth',1);
hold on;
plot(a,energy(34:66,2)/768,'bs','linewidth',1);
plot(a,energy(67:99,2)/864,'go','linewidth',1);
plot(a,energy(1:33,1)/768,'r-','linewidth',2);
plot(a,energy(34:66,1)/768,'b--','linewidth',2);
plot(a,energy(67:99,1)/864,'g:','linewidth',2);
xlabel('$a/a_0$','fontsize',12,'interpreter','latex');
ylabel('Energy (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
xlim([0.94,1.06]);
title('(b)');

subplot(1,3,3);
plot(a,virial(198*0+1:198*0+33,2)/768,'rd','linewidth',1);
hold on;
plot(a,virial(198*0+34:198*0+66,2)/768,'bs','linewidth',1);
plot(a,virial(198*0+67:198*0+99,2)/864,'go','linewidth',1);
plot(a,virial(198*0+1:198*0+33,1)/768,'r-','linewidth',2);
plot(a,virial(198*0+34:198*0+66,1)/768,'b--','linewidth',2);
plot(a,virial(198*0+67:198*0+99,1)/864,'g:','linewidth',2);
xlabel('$a/a_0$','fontsize',12,'interpreter','latex');
ylabel('Virial (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
xlim([0.94,1.06]);
legend('DFT-monclinic','DFT-cubic','DFT-tetragonal','MBKS-monclinic','MBKS-cubic','MBKS-tetragonal')
title('(c)');
