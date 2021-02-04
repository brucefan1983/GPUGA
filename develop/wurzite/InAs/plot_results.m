clear; close all;

% get the outputs from GPUGA:
load energy.out;
load virial.out;
load force.out;

% number of atoms
num_atoms=ones(size(energy,1),1)*192;
num_atoms(end-2:end)=96; % last three polytypes

figure;
plot(force(1:end,4),force(1:end,1),'d');
hold on;
plot(force(1:end,5),force(1:end,2),'s');
plot(force(1:end,6),force(1:end,3),'o');
xlim([-1.1,1.1]);
ylim([-1.1,1.1]);
xlabel('Force (DFT) (eV/\AA)','fontsize',12,'interpreter','latex');
ylabel('Force (Potential) (eV/\AA)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('x','y','z');

figure;

for n=1:3
subplot(1,3,n);
plot(force(1:192,n)-force(1:192,3+n),'o');
ylim([-1.1,1.1]);
xlabel('Atom ID','fontsize',12,'interpreter','latex');
ylabel('Force Error (eV/\AA)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
    if n==1
        title('(a) x-direction')
    elseif n==2
        title('(b) y-direction')
    elseif n==3
        title('(c) z-direction')
    end
end


figure;
plot(energy(:,2)./num_atoms,'o');
hold on;
plot(energy(:,1)./num_atoms,'x','linewidth',2);
ylim([-4.63,-3.8]);
xlabel('Configuration ID','fontsize',12,'interpreter','latex');
ylabel('Energy (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

% virial
label_text=['Virial-xx (eV)';'Virial-yy (eV)';'Virial-zz (eV)';'Virial-xy (eV)'];
for n=1:4
    figure;
    plot(virial((n-1)*end/6+1:n*end/6,2)./num_atoms,'o','linewidth',1);
    hold on;
    plot(virial((n-1)*end/6+1:n*end/6,1)./num_atoms,'x','linewidth',1);
    ylim([-3,6]);
    xlabel('Configuration ID','fontsize',12,'interpreter','latex');
    if n==1
        ylabel('Virial-xx (eV)','fontsize',12,'interpreter','latex');
    elseif n==2
        ylabel('Virial-yy (eV)','fontsize',12,'interpreter','latex');
    elseif n==3
        ylabel('Virial-zz (eV)','fontsize',12,'interpreter','latex');
    elseif n==4
        ylabel('Virial-xy (eV)','fontsize',12,'interpreter','latex');  
        legend('DFT','Potential');
    end
    set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
end

