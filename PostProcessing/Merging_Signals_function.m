% This code is used to generate merged PnC signal from multiple experimental data and calculate transmission spectra for loss value calcuation.                                                          %

clc; clear; close all
data= TimeToFreqDomMulti('400','700','Sample2');

function [frequency_domain] = TimeToFreqDomMulti(d, lxy, sample)
    set(0, 'DefaultAxesFontName', 'Times');
    set(gca,'FontSize',20)
    %% preparation 
    % log-based sequence
    freq_full_range = 2.^([log2(1e6)-4*(log2(4e6)-log2(1e6))/4:...
        (log2(4e6)-log2(1e6))/4:log2(4e6)+(log2(4e6)-log2(1e6))/4]);
    % freq_breaks for freq localizing basis functions 
    freq_breaks = (3/5)*2.^([log2(1e6)-4*(log2(4e6)-log2(1e6))/4:...
        (log2(4e6)-log2(1e6))/4:log2(4e6)+2*(log2(4e6)-log2(1e6))/4]);          
    
    % raw waveform for merging
    Data_freq_range = [1, 1.5, 2, 2.75, 4]*1e6;    
    
    num_freq = length(Data_freq_range);   
    
    Fs = 50e6;                                   % sampling frequency: 50 MHz
    % choose the interested segment waveform 
    % lB = 540;                                    % ref lowerBoundary       
    % uB = 1101;                                   % ref upperboundary
     lB = 649;                                    % pnc lowerBoundary       
     uB = 2150;  
    L_master = 4000;                             % final length after FFT 
    t = linspace(0,80,4000);
    f = Fs*(0:(floor(L_master/2)))/L_master;     % frequency axis
    
    sig_cum = zeros(L_master/2+1, 1);            % cumulative signal 

    %% read and merge data with filtering
    for freq_data = 1000:250:4000   
        % location of the input files
        TD_file = strcat('PnC-signal-TD_d', d, '_lxy', lxy, '_', sample, '_freq', string(freq_data), 'kHz.csv');
        % TD_file= 'PnC-signal-TD_d350_lxy700_Sample1_freq1000kHz.csv'
        file_path =strcat('/Users/christinad/Desktop/ML/code/TD_data/', TD_file);
        TD_data=csvread(file_path);
            %% data check; plot time domain 
            figure(1)
            hold on 
            plot(t, TD_data)
            title('Time Domain')
            xlabel('time [micro second]')
            ylabel('Amplitude')

         for ii = 1: num_freq

            % remove basis 
            % sample A
            avgA = mean(TD_data(lB:uB,1));
            %% check after this line 63
            trace_dataA(5).freq(ii).data(:,1)=TD_data(lB:uB,1)-avgA;
         end

        %% merging with filtering 
        [act_cumA] = Filtering(trace_dataA, Fs, num_freq, Data_freq_range, ...
            freq_breaks, L_master);
        sig_cum = sig_cum + (act_cumA);
        

    end
    %% merged average PnC signal  
    avg_sig = 1/12*sig_cum;
    % remove the first point
    freq = f(2:end);     % the freq for plotting
    Merged_Avg_sig = avg_sig(2:end);  % the fft of the merged signal
    c = date;
    
    % folder name and file name of the output location
    foldername= strcat('/Users/christinad/Desktop/ML/code/data/merged_data/',sample,'_', c); 
    mkdir(foldername);
    filename = strcat(foldername, '/freq.domain.d',d,'.lxy',lxy,'.csv');
    writematrix(Merged_Avg_sig, filename)
    
    % returned value
    frequency_domain= Merged_Avg_sig;
% average signal with merged datasets plot
    figure(5)
    plot(freq,Merged_Avg_sig,'k')
    title('Averged Signal with Merged Multiple Datasets')
    leg=strcat(sample,'   d: ', d,'   lxy: ', lxy);
    legend(leg)
    xlabel('Frequency [MHz]')
    ylabel('Amplitude')
    xlim([0 5e6])
 %   ylim([0 0.5])
    
    
    end
% Filtering freq localizing basis functions 
function [act_cum] = Filtering(trace_data, Fs, num_freq, Data_freq_range, freq_breaks, L_master)
freq_full_range = 2.^([log2(1e6)-4*(log2(4e6)-log2(1e6))/4:...
    (log2(4e6)-log2(1e6))/4:log2(4e6)+(log2(4e6)-log2(1e6))/4]);
% Frequency axis
f = Fs*(0:(floor(L_master/2)))/L_master;
    %% frequency localizing basis functions (Equation 35 of Welsh 2007)
    
    q =1;
    %for q = 1:1 
    for i = 1: length(freq_full_range)
        % F(i+1) filter is for F(i)
        % left part of the equation
        F(i).order(q) = tf([(freq_breaks(i+1)*2*pi)^q, zeros(1,q*(i))],[1]);         
        for k = 1:i+1
            den = [1 freq_breaks(k)*2*pi];   % coefficient of (s+Pi)
            for qq = 1:q-1
                % conv() to get the coefficient of den 1/(s+Pi)^q
                den = conv(den, [1 freq_breaks(k)*2*pi]);                           
            end
            F(i).order(q) = F(i).order(q)*tf([1],den);
        end
            [mag phas] = bode(F(i).order(q),f*2*pi);
            

        %end 
 
    end 


    %% merging data with selected frequency localizing basis functions
    act_cum = zeros(L_master/2+1, 1);
    for i = 1:num_freq
        L_orig = length(trace_data(5).freq(i).data(:,1));
        act(1:L_orig) = trace_data(5).freq(i).data(:,1);
        L_zero_padding = L_master - L_orig;                            
        
        %% signal and windowing 
        wind(1:L_orig) = blackman(L_orig);
        act_winded = wind.*act;
        
        % FFT
        act_winded = [act_winded, zeros(1, L_zero_padding)]; % zero-padding

        L = length(act_winded); 
        ACT = fft(act_winded);                  % window processed
        ACT2 = abs(ACT/L_orig);                 % normalization by interested segment signal length
        SSB = ACT2(1:floor(L/2)+1);             % Single Side Band (SSB)
        SSB(2:end-1) = 2*SSB(2:end-1);                      
        pro_data(5).freq(i).data(:,1) = SSB;              
     
    
        [mag phas] = bode(F(i+4).order(q),f*2*pi); 
                
 
        %% filter data with frequency localizing basis functions
        for k = 1:length(pro_data(5).freq(i).data)
            pro_data(5).freq(i).data(k,1) = ...
            pro_data(5).freq(i).data(k,1)*mag(:,:,k)...
            *exp(j*phas(:,:,k)*pi/180);
        end
        %% merging multiple experiments from different testing frequencies 
        
        act_cum = act_cum + abs(pro_data(5).freq(i).data(:,1));   
    end
    
end 
