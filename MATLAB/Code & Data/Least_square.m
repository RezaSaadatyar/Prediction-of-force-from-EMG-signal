%##########################################################################
% ==================== EMG force & Torque estimation ======================
% =================== Presented by: Reza Saadatyar ========================
% ==================== Reza.Saadatyar92@gmail.com =========================
% ============================== 2017 =====================================
clc;clear;close all;
%% Data load
data=load('30.mat');
Torq=data.Raw_Torque';%% data Torque
BB=(data.TAB_RMS.BB)'; %% data BB
BR=(data.TAB_RMS.BR)'; %% data BR
TM=(data.TAB_RMS.TM)'; %% data TM
% nosie TM
TM(TM>50)=mean(TM);
TL=(data.TAB_RMS.TL)'; %% data TL
%% Butterworth Filter
fc = 1; % Cut off frequency Hz
fs = 2048; % Sampling rate Hz
[b,a] = butter(2,fc/(fs/2)); % Butterworth filter of order 2
TorqF = filter(b,a,Torq);
%% Downsample
TorqFD=decimate(TorqF,1025);
%% Estimate 
t=numel(TorqFD);
r=input('Enter 30, 70 or 100:');
switch r
    case 30
        N=ceil(t*0.30); 
    case 50
        N=ceil(t*0.50); 
     case 70
        N=ceil(t*0.70); 
     case 100
        N=numel(TorqFD); 
end
y = zeros(N,1); %yhat
a=rand(4,1); % coefficient random 
PHI=zeros(N,numel(a));% K
for i = 1:N
    y(i)=a(1)*BB(i)+a(2)*BR(i)+a(3)*TM(i)+a(4)*TL(i);
end
for i =  1:N
    PHI(i,:) =[BB(i),BR(i),TM(i),TL(i)];
end
%% EStim ##################################################################
Torq=TorqFD(1:N,:);
tetah=PHI\Torq;  %% (PHI' * PHI)^-1 * PHI' *Y
TorqEst=tetah(1)*BB(1:N,:)+tetah(2)*BR(1:N,:)+tetah(3)*TM(1:N,:)+tetah(4)*TL(1:N,:);
%% R2 & VAF & Pea
R2 = 1 - sum((Torq - TorqEst).^2)/sum((Torq - mean(Torq)).^2);
VAF=(1-sum((Torq- TorqEst).^2)/sum((Torq).^2))*100;
% Pea
num=sum((Torq- mean(Torq)).*(TorqEst - mean(TorqEst)));
den=sqrt(sum((Torq- mean(Torq)).^2)).*sqrt(sum((TorqEst - mean(TorqEst)).^2));
Pea=num/den;
%##########################################################################
%% Vaildation @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
TorqFDnt=TorqFD(N:end,:);
TorqEstnt=tetah(1)*BB(N:end,:)+tetah(2)*BR(N:end,:)+tetah(3)*TM(N:end,:)+tetah(4)*TL(N:end,:);
%% R2 & VAF & Pea
R2nt = 1 - sum((TorqFDnt - TorqEstnt).^2)/sum((TorqFDnt - mean(TorqFDnt)).^2);
VAFnt=(1-sum((TorqFDnt - TorqEstnt).^2)/sum((TorqFDnt).^2))*100;
% Pea
num=sum((TorqFDnt - mean(TorqFDnt)).*(TorqEstnt - mean(TorqEstnt)));
den=sqrt(sum((TorqFDnt - mean(TorqFDnt)).^2)).*sqrt(sum((TorqEstnt - mean(TorqEstnt)).^2));
Peant=num/den;
% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%% Plot Results
figure();
subplot(211)
plot(TorqFD,'LineWidth',2');hold on;
plot(BB,'r:','LineWidth',2');grid on;
plot(BR,'y','LineWidth',2');grid on;
plot(TM,'k','LineWidth',2');grid on;
plot(TL,'g--','LineWidth',2');grid on;
xlabel('Time(sec)','FontSize',12,'FontWeight','bold')
ylabel('Amp','FontSize',12,'FontWeight','bold')
legend({'TorqF','Biceps Brachii','Brachioradialis','Triceps Medial Brachii','Triceps Lateral Brachii'},'FontSize',10,'FontWeight','bold')

subplot(212)
TorqE=zeros(t,1);TorqE(1:N,:)=TorqEst;
TorqEtn=zeros(t,1);TorqEtn(N:end,:)=TorqEstnt;
plot(TorqFD,'LineWidth',2);hold on;
plot(TorqE,'k-.','LineWidth',2);
plot(TorqEtn,'r:','LineWidth',2);
grid on
ax=gca;
ax.FontSize=12;
ax.FontWeight='bold';
xlabel('Time','FontWeight','bold','FontSize',12,'FontName','Times New Roman') ; 
ylabel('Amp','FontWeight','bold','FontSize',12,'FontName','Times New Roman') ; 
legend({'Torq','TorqTrain','TorqTest'},'FontWeight','bold','FontSize',10,'FontName','Times New Roman')
title(['LS algorithm ;  Num sample for training:%' num2str(r),' of the Data',...
    ';  Num sample for test:%' num2str(100-r),' of the Data'],'FontWeight','bold','FontSize',12,'FontName','Times New Roman')
