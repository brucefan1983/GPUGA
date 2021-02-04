clear; close all;

% get the outputs from GPUGA:
num_atoms=ones(210,1)*192;
e=cell(6,1);
load Si/energy.out;e{1}=energy(1:210,:)./num_atoms;
load GaP/energy.out;e{2}=energy(1:210,:)./num_atoms;
load GaAs/energy.out;e{3}=energy(1:210,:)./num_atoms;
load InP/energy.out;e{4}=energy(1:210,:)./num_atoms;
load InAs/energy.out;e{5}=energy(1:210,:)./num_atoms;
load ZnSe/energy.out;e{6}=energy(1:210,:)./num_atoms;


figure;
for n=1:6
    subplot(2,3,n);
    plot(e{n}(1:end,2),'s');
    hold on;
    plot(e{n}(1:end,1),'x');
    xlim([0,211]);
    xlabel('Configurations ID','fontsize',10,'interpreter','latex');
    ylabel('$E$ (eV)','fontsize',10,'interpreter','latex');
    set(gca,'fontsize',10,'ticklength',get(gca,'ticklength')*2);
    if n==1
        legend('DFT','Potential');
        title('(a) Si');
    elseif n==2
        title('(b) GaP');
    elseif n==3
        title('(c) GaAs');
    elseif n==4
        title('(d) InP');
    elseif n==5
        title('(e) InAs');
    elseif n==6
        title('(f) ZnSe');
    end
end
