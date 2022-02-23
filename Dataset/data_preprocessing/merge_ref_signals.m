
% Filtering freq localizing basis functions 
function [merged_signal, x_freqs] = merge_signals(data)

% -----------------------------------------------
% Merge all reference signals
% Md Ferdous Alam, 07/05/2021 

%{ 
Docementation: 
call this function like this --> [merged_signal, x_freqs] = merge_signals(data) 
where, 
input ---> data 
output --->  merged_signal and corresponding frequency for x-axis
please give the input like this ---> 
build input dataset
data = {};
data{1} = freq1; % load csv file for freq 1 
data{2} = freq2; % load csv file for freq 1
data{3} = freq3; % load csv file for freq 1
data{4} = freq4; % load csv file for freq 1
data{5} = freq5;  % load csv file for freq 1
as many frequecny as you want to merge
 etc. 
% -------- input: data -----------------
% innput data should in cell array  
% for example; 
% data = {}; 
% data{1} = [1, 2, 3, 4, 5];
% data{2} = [5, 3, -2, 6, 7]; 
% ------------------------------------
%}

num_freq = length(data);
% log-based sequence
freq_full_range = 2.^([log2(1e6)-4*(log2(4e6)-log2(1e6))/4:...
    (log2(4e6)-log2(1e6))/4:log2(4e6)+(log2(4e6)-log2(1e6))/4]);
% freq_breaks for freq localizing basis functions 
freq_breaks = (3/5)*2.^([log2(1e6)-4*(log2(4e6)-log2(1e6))/4:...
    (log2(4e6)-log2(1e6))/4:log2(4e6)+2*(log2(4e6)-log2(1e6))/4]);   

Fs = 50e6;                                   % sampling frequency: 50 MHz
% choose the interested segment waveform 
lB = 540;                                    % lowerBoundary       
uB = 1101;                                   % lowerBoundary

L_master = 4000;                             % final length after FFT 
% Frequency axis
f = Fs*(0:(floor(L_master/2)))/L_master;

%% select interested segment waveform and remove basis
trace_data = {};
for ii = 1: num_freq 
    % remove basis 
    avg_signal = mean(data{ii}(lB:uB,1));
    trace_data{ii} = data{ii}(lB:uB,1)-avg_signal;      
end

%% frequency localizing basis functions (Equation 35 of Welsh 2007)
    for q = 1:1 
        for i = 1: length(freq_full_range)
            % F(i+1) filter is for F(i)
            % left part of the equation
            F(i).order(q) = tf([(freq_breaks(i+1)*2*pi)^q zeros(1,q*(i))],[1]);         
            for k = 1:i+1
                den = [1 freq_breaks(k)*2*pi];   % coefficient of (s+Pi)
                for qq = 1:q-1
                    % conv() to get the coefficient of den 1/(s+Pi)^q
                    den = conv(den, [1 freq_breaks(k)*2*pi]);                           
                end
                F(i).order(q) = F(i).order(q)*tf([1],den);
            end
            
            [mag phas] = bode(F(i).order(q),f*2*pi);
            
        end 
    end 


    %% merging data with selected frequency localizing basis functions
act_cumulative = zeros(L_master/2+1, 1);
    for i = 1:num_freq
        L_orig = length(trace_data{i}(:, 1));
        act(1:L_orig) = trace_data{i}(:, 1);
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
        pro_data{i}(:, 1) = SSB;              
     
        [mag phas] = bode(F(i+4).order(q),f*2*pi); 
                
        %% filter data with frequency localizing basis functions
        for k = 1:length(pro_data{i})
            pro_data{i}(k, 1) = pro_data{i}(k, 1)*mag(:,:,k)*exp(j*phas(:,:,k)*pi/180);
        end
        %% merging multiple experiments from different testing frequencies 
        act_cumulative = act_cumulative + abs(pro_data{i}(:, 1));   
    end
    
x_freqs = f; 
merged_signal = act_cumulative; 
end 














