clear; close all;

% get the outputs from GPUGA:
num_atoms=ones(210,1)*192;
v=cell(6,1);
load Si/virial.out;v{1}=virial;
load GaP/virial.out;v{2}=virial;
load GaAs/virial.out;v{3}=virial;
load InP/virial.out;v{4}=virial;
load InAs/virial.out;v{5}=virial;
load ZnSe/virial.out;v{6}=virial;

for m=1:6
    offset=(m-1)*213;
    figure;
    for n=1:6
        subplot(2,3,n);
        plot(v{n}(offset+1:offset+210,2)./num_atoms,'s');
        hold on;
        plot(v{n}(offset+1:offset+210,1)./num_atoms,'x');
        xlim([0,211]);
        xlabel('Configurations ID','fontsize',10,'interpreter','latex');
        if m==1
            ylabel('$W_{xx}$ (eV)','fontsize',10,'interpreter','latex');
        elseif m==2
            ylabel('$W_{yy}$ (eV)','fontsize',10,'interpreter','latex');
        elseif m==3
            ylabel('$W_{zz}$ (eV)','fontsize',10,'interpreter','latex');
        elseif m==4
            ylabel('$W_{xy}$ (eV)','fontsize',10,'interpreter','latex');
        elseif m==5
            ylabel('$W_{yz}$ (eV)','fontsize',10,'interpreter','latex');
        elseif m==6
            ylabel('$W_{zx}$ (eV)','fontsize',10,'interpreter','latex');
        end
        
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
end


