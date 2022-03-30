%##########################################################################
% ==================== EMG force & Torque estimation ======================
% =================== Presented by: Reza Saadatyar ========================
% ============================== 2017 =====================================
clc;clear;close all;
%% Load data
data=load('50.mat');
Torq=data.Raw_Torque';%% data Torque
BB=data.TAB_RMS.BB; %% data BB
BR=data.TAB_RMS.BR; %% data BR
TM=data.TAB_RMS.TM; %% data TBM
TL=data.TAB_RMS.TL; %% data TBL
%% Butterworth Filter (10-750Hz) used for data 
fc = 1; % Cut off frequency
fs = 2048; % Sampling rate
[b,a] = butter(2,fc/(fs/2)); % Butterworth filter of order 6
TorqF = filter(b,a,Torq);
%% Downsample
desired=decimate(TorqF,1025);
%% Estimate
s=BB+BR+TM+TL;
mu=0.000001;%step
change=1; %% AR Model
if change==0
    w=[0 0 0 0]'; %initional tap-weights
    tit='Torque estimation using AR model';
else 
    w=[0 0 0 0 0]';  %initional tap-weights
    tit=' Torque estimation using ARX model';
end
y=zeros(1,numel(s)-numel(w));
figure()
subplot(211)
for n=1:1:numel(s)-numel(w) %Number epockss
    %% AR model
    if change==0
        u=[s(n) s(n+1) s(n+2) s(n+3)]';
        title(['\fontsize{10}Coefficients Model AR Using least mean square algorithm; \mu=' num2str(mu)]);
    else 
        %% ARX model
        u=[desired(n) s(n+3) s(n+2) s(n+1) s(n)]'; 
        title(['\fontsize{10}Coefficients Model ARX Using least mean square algorithm; \mu=' num2str(mu)]);
   end
    if n >= 170
        mu=0.000001;%step
    end
    %% update
    e=desired(n)-w'*u;
    w=w+mu*u*e;  %update the tap-weights
    y(n)=w'*u; %output
    plot(n,w,'.');%show tap-weights
    hold on;grid on;
end
ylabel({'Amp'},'FontSize',10,'FontWeight','bold')
xlabel({'Sample'},'FontSize',10,'FontWeight','bold')
%% Plot Results
subplot(212)
plot(desired,'LineWidth',2);
hold on;
plot(y,'color','r','LineWidth',2);grid on;
ylabel({'Amp'},'FontSize',10,'FontWeight','bold')
xlabel({'Sample'},'FontSize',10,'FontWeight','bold')
legend({'Torq','Torq Estimate'},'FontSize',8);
title([tit,'; \mu=' num2str(mu)],'fontsize',10);