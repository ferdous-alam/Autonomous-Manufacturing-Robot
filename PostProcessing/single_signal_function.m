%% This code is used for FFT, Data collection -- Zhi Zhang 2/16/2020, MD Ferdous, Antony
 clc; clear all; close all
data=TimeToFreqDomSing('PnC-signal-TD_d350_lxy700_Sample1_freq1250kHz.csv');
 
 function [frequency_domain] = TimeToFreqDomSing(filename)
 % This version with FFT and Transfer fn data saving
 filepath = '/Users/christinad/Desktop/ML/code/TD_data/'
 A = csvread(strcat(filepath, filename)); %  signal
 d = 0;      % used for legend label
 a = 0;      % used for legend label
 
 % signal range in time domain. This is lower and upper values are determined by ploting the time domain data to determine where the pulse starts and ends.
 
 SigRangeA_lower = 649;
 SigRangeA_upper = 2150;
 
 
 AmpA(:,1) = A(:,1);     % first column vector of csv file A
 Sampling_frequency = 50e6;              % Sampling Frequency (MHz): 50; 25; 12.5; 6.25
 n = length(AmpA);       % number of data points
 Total_time = n/Sampling_frequency;               % total time
 dt = Total_time/n;               % difference in time for plotting
 t = dt*(0:n-1)'*10^6;   % time for measurements of A and B csv files, goes to n-1 bc at n it repeats (MHz conversion)!check this
 
 %Calculating FFT
 
 % set mean value before and after pulse, eliminates noise and reflections
 df = 1/dt;                  % difference between frequency
 
 AmpNA = zeros(size(AmpA));  % matrix of zeros
 MA = mean(AmpA);            % mean of A
 AmpNA = AmpNA + MA;         % the "zero" point (mean of A)
 AmpNA(SigRangeA_lower:SigRangeA_upper) = AmpA(SigRangeA_lower:SigRangeA_upper);        % check reference signal data range
 
 
 
 %take fft and normalize coefficents
 AmpfA = abs(fft(AmpNA));                 % coefficents of the FFT
 
 lenA = length(AmpfA);       % length of A
 
 
 FA = df*(0:lenA/2)/lenA;    % create frequency depth points for plotting
 AmpfA = AmpfA(1:lenA/2+1);  % amplitudes of FFT STARTING FROM FIRST VALUE
 
 
 AmpfAN =AmpfA./max(AmpfA(2:length(AmpfA))); % ! create dimensionless value (compare all values to max values)normalizes
 
 %% !im not sure what this for loop is for /ignoring noise
 for i=2:1:length(AmpfAN) % for loop from 2 to length of AmpfAN, incrementing by 1 (starts at 2 bc before it is mean)
     if(AmpfAN(i) >= 0.06) % if the value is greater than .06 the index of B is =i (has influence) and exits
         fANindex_B = i;
         break;
     end
 end
 for i=length(AmpfAN):-1:2   % for loop from length of AmpfAN to 2, decrementing by 1
     if(AmpfAN(i) >= 0.06)    % if the value is greater than .06 the index of E is =i  & exits
         fANindex_E = i;
         break;
     end
 end
 
 
 
 %% Plotting FFT
 figure();
 plot(FA*10^-6,abs(AmpfA),'b','LineWidth',1);
 grid on;
 xlabel('Frequency [MHz]','fontweight','bold','fontsize',12);
 ylabel('Amplitude [au]','fontweight','bold','fontsize',12);
 set(gca,'FontSize',12,'FontWeight','Bold');
 title('Frequency Domain','fontweight','bold','fontsize',12);
 xlim([0.2 5.0]);
 hold on;
 legendTitle = sprintf('Sample \n(d=%0.2fmm, a=%0.2fmm)', d, a);
 legend({'Reference Signal',legendTitle},'fontweight','bold','fontsize',10); %change specimen name
 
 frequency_domain= abs(AmpfA)
 filepath_fft = strcat('/Users/christinad/Desktop/ML/code/data/single_data/fft_single_', filename);
 filepath_fft_figure =strcat('/Users/christinad/Desktop/ML/code/data/single_data/figures/fftplot_', filename);
 saveas(gcf,strcat(filepath_fft_figure,'.jpeg'));
 writematrix(frequency_domain, filepath_fft);
 end
 
 
