%% This code is used for FFT, Data collection -- Zhi Zhang 2/16/2020, MD Ferdous, Antony
 clc; clear all; close all

% This version with FFT and Transfer fn data saving

A = csvread('DOE_avg_reference_signal_timedomain.csv'); % Reference signal
B = csvread('DOE_Ref_SampleA_300V_0dB_50avg_freq-2250kHz_location-7_Block-1_Jan-19-2020_11-18.csv');  % Actual signal
d = 0;
a = 0;
SigRangeA_lower = 550; % signal range in time domain; choose this value by ploting the time domain data.
SigRangeA_upper = 1050;

SigRangeB_lower = 800;
% SigRangeB_upper = 1740;
SigRangeB_upper = 2040; % this is new limit 03/17/2020
AmpA(:,1) = A(:,1);
AmpB(:,1) = B(:,1);
SR = 50e6;                     % Sampling Frequency (MHz): 50; 25; 12.5; 6.25
n = length(AmpA);
T = n/SR;
dt = T/n;
t = dt*(0:n-1)'*10^6;
% figure()
% subplot(2,1,1)
% plot(t,AmpA,'b','LineWidth',1);
% hold on;
% grid on;
% xlabel('Time [\mus]','fontweight','bold','fontsize',12);
% ylabel('Amplitude','fontweight','bold','fontsize',12);
% set(gca,'FontSize',12,'FontWeight','Bold');
% title('Time Domain','fontweight','bold','fontsize',12);
% legend({'Reference Signal'},'fontweight','bold','fontsize',10);
% subplot(2,1,2)
% plot(t,AmpB,'r','LineWidth',1);
% grid on;
% xlabel('Time [\mus]','fontweight','bold','fontsize',12);
% ylabel('Amplitude','fontweight','bold','fontsize',12);
% set(gca,'FontSize',12,'FontWeight','Bold');
% title('Time Domain','fontweight','bold','fontsize',12);

% legendTitle = sprintf('Sample \n(d=%0.2fmm, a=%0.2fmm)', d, a); 
% legend({legendTitle},'fontweight','bold','fontsize',10); %change specimen name

% saveas(gcf,strcat(Filename,'_TimeDomain','.jpeg'));
%%
%Calculating FFT
df = 1/dt;% = SR
AmpNA = zeros(size(AmpA));
MA = mean(AmpA);
AmpNA = AmpNA + MA;
AmpNB = zeros(size(AmpB));
MB = mean(AmpB);
AmpNB = AmpNB + MB;
AmpNA(SigRangeA_lower:SigRangeA_upper) = AmpA(SigRangeA_lower:SigRangeA_upper);        % check reference signal data range 
AmpfA = abs(fft(AmpNA));                 % FFT Reference Data 
AmpNB(SigRangeB_lower:SigRangeB_upper) = AmpB(SigRangeB_lower:SigRangeB_upper);        % 1753
AmpfB = abs(fft(AmpNB));                 % FFT Actual Data 
lenA = length(AmpfA);
FA = df*(0:lenA/2)/lenA;
AmpfA = AmpfA(1:lenA/2+1);
AmpfAN =AmpfA./max(AmpfA(2:length(AmpfA)));
for i=2:1:length(AmpfAN)
   if(AmpfAN(i) >= 0.06)
       fANindex_B = i;
       break;
   end
end
for i=length(AmpfAN):-1:2
   if(AmpfAN(i) >= 0.06)
       fANindex_E = i;
       break;
   end
end
lenB = length(AmpfB);
FB = df*(0:lenB/2)/lenB;
AmpfB = AmpfB(1:lenB/2+1);              
AmpfBN =AmpfB./max(AmpfB(2:length(AmpfB)));
%Plotting FFT
figure();
plot(FA*10^-6,abs(AmpfA),'b','LineWidth',1);
grid on;
xlabel('Frequency [MHz]','fontweight','bold','fontsize',12);
ylabel('Amplitude [au]','fontweight','bold','fontsize',12);
set(gca,'FontSize',12,'FontWeight','Bold');
title('Frequency Domain','fontweight','bold','fontsize',12);
xlim([0.2 5.0]);
hold on;
plot(FB*10^-6,abs(AmpfB),'r','LineWidth',1);
legendTitle = sprintf('Sample \n(d=%0.2fmm, a=%0.2fmm)', d, a); 
legend({'Reference Signal',legendTitle},'fontweight','bold','fontsize',10); %change specimen name

% saveas(gcf,strcat(Filename,'_FreqDomain','.jpeg'));

% %% store FFT data
% ref_signal_FFT = [FA'*10^-6 abs(AmpfAN)]; 
% act_signal_FFT = [FB'*10^-6 abs(AmpfBN)];
% % save signal FFT data as csv files f=2250 kHz
% csvwrite(strcat(Filename,'_Ref_signal_FFT.csv'), ref_signal_FFT)
% csvwrite(strcat(Filename,'_Act_signal_FFT.csv'), act_signal_FFT)

%% Calculating Amplitudes in Decibels
dbA = mag2db(AmpfAN);
mA = length(dbA);
dbB = mag2db(AmpfBN);
mB = length(dbB);
dbNA = dbA(2:mA);
Mx = max(dbNA);
My = Mx - 6;
for i=1:mA
    
     if(dbA(i)==Mx)
        Nmax = i;
     end
end
Rmax = Nmax;
for i=1:mA
    
    if(dbA(i)<My && i<Nmax)
        N1 = i;
    end
    if(dbA(i)>My && i>Nmax)
        N2 = i;
    end
    
end
L1A = FA(N1)*10^-6
L2A = FA(N2)*10^-6
CfA = (L1A+L2A)/2
PkA = FA(Rmax) *10^-6
dbNB = dbB(2:mB);
Mx = max(dbNB);
My = Mx - 6;
for i=1:mB
    
     if(dbB(i)==Mx)
        Nmax = i;
     end
end
 for i=1:mB
    
    if(dbB(i)<=My && i<Nmax)
        N1 = i;
    end
    if(dbB(i)>=My && i>Nmax)
        N2 = i;
        if(dbB(i+1)<My)
            break;
        end        
    end        
end
L1B = FB(N1)*10^-6
L2B = FB(N2)*10^-6
CfB = (L1B+L2B)/2
PkB = FB(Nmax)*10^-6
%Calculating Transfer Function
FAf = FA(2:mA);
num = AmpfA(2:mA);
den = AmpfB(2:mB)/100;
TNF = den./num;
dbTNF = mag2db(TNF);
k1 = 1;
k2 = 1;
while(FAf(k1)<500000) %Lower Limit 0.5 MHz
    k1= k1 + 1;
end
while(FAf(k2)<=3500000) %Upper Limit 3.5 MHz
    k2 = k2 + 1;
end
% 
%%Calulating Peak & Central Frequency of the Transfer Function Plots
dbTNF_lim = dbTNF(k1:k2);
FAf_lim = FAf(k1:k2);
mC = length(dbTNF_lim);
Mx = max(dbTNF_lim);
My = Mx - 6;
for i=1:mC
    if(dbTNF_lim(i)==Mx)
        Nmax = i;
     end
end
 for i=1:mC
    
    if(dbTNF_lim(i)<My && i<Nmax)
        N1 = i;
    end
    if(dbTNF_lim(i)>My && i>Nmax)
        N2 = i;
    end
        
end
L1TF = FAf_lim(N1)*10^-6;
L2TF = FAf_lim(N2)*10^-6;
CfTF = (L1TF+L2TF)/2;
PkTF = FAf_lim(Nmax)*10^-6;
% %%Plotting Transfer Function PLots
% fig4 = figure(4);
% subplot(2,1,1)
% plot(FAf*10^-6,TNF,'k','LineWidth',1);
% xlim([0.2 5.0]);
% grid on;
% xlabel('Frequency [MHz]','fontweight','bold','fontsize',12);
% ylabel('Amplitude [au]','fontweight','bold','fontsize',12);
% set(gca,'FontSize',12,'FontWeight','Bold');
% title('Transfer Func = Sample Amp / Ref Amp','fontweight','bold','fontsize',12);
% subplot(2,1,2)
% plot(FAf*10^-6,TNF,'k','LineWidth',1);
% xlim([1 3]);
% grid on;
% xlabel('Frequency [MHz]','fontweight','bold','fontsize',12);
% ylabel('Amplitude [au]','fontweight','bold','fontsize',12);
% set(gca,'FontSize',12,'FontWeight','Bold');
% title('Transfer Func = Sample Amp / Ref Amp','fontweight','bold','fontsize',12);
% 
% saveas(gcf,strcat(Filename,'_TransferFunc','.jpeg'));

% dbTNF = smooth(dbTNF,3);             % Transmission data 
% dblim(1,1:2000) = -40; 

figure()
plot(FAf*10^-6,dbTNF,'m','LineWidth',1); % plot the transmission plot 
% hold on;
% plot(FAf*10^-6,dblim,'b','LineWidth',1);

xlim([0.5 4.5]);
ylim([-70 20]);
grid on;
xlabel('Frequency [MHz]','fontweight','bold','fontsize',12);
ylabel('Amplitude [dB]','fontweight','bold','fontsize',12);
set(gca,'FontSize',12,'FontWeight','Bold');
title('Transfer Func (Decibel Scale)','fontweight','bold','fontsize',12);


% saveas(gcf,strcat(Filename,'_TransferFunc_dB','.jpeg'));

% %% store transfer function data 
% a = []; b = []; c = [];         
% transmission_data = [dbTNF dblim' FAf'*10^-6]; 
% for ii = 1:length(transmission_data(:, 1))
%     if transmission_data(ii, 3) >= 0.2 && transmission_data(ii, 3) <= 5
%         a(ii) = transmission_data(ii, 1);
%         b(ii) = transmission_data(ii, 2);
%         c(ii) = transmission_data(ii, 3); 
%     end
% end
% a = nonzeros(a); 
% b = nonzeros(b); 
% c = nonzeros(c); 
% transmission_data = [a b c];
% 
% csvwrite(strcat(Filename,'_Transmission_data.csv'), transmission_data)



