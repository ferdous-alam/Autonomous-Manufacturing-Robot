
 function td2fft(td_signal, SigRange_lower, SigRange_upper, PnC_dim, sample_no, freq_no)

 Amp(:,1) = td_signal(:,1);  % first column vector of csv file of time domain signal
 Sampling_frequency = 50e6;  % Sampling Frequency (MHz): 50; 25; 12.5; 6.25

 n = length(Amp);       % number of data points
 Total_time = n/Sampling_frequency;               % total time
 dt = Total_time/n;               % difference in time for plotting
 t = dt*(0:n-1)'*10^6;   % time for measurements of A and B csv files, goes to n-1 bc at n it repeats (MHz conversion)!check this

 %%
 %Calculating FFT

 % set mean value before and after pulse, eliminates noise and reflections
 df = 1/dt;                  % difference between frequency
 AmpN = zeros(size(Amp));  % matrix of zeros
 MA = mean(Amp);            % mean of A
 AmpN = AmpN + MA;         % the "zero" point (mean of the signal)
 AmpN(SigRange_lower:SigRange_upper) = Amp(SigRange_lower:SigRange_upper);        % check reference signal data range 
 %take fft and normalize coefficents
 Ampf = abs(fft(AmpN));   % coefficents of the FFT   
 length_td_signal = length(Ampf);       % length of td signal

 FA = df*(0:length_td_signal/2)/length_td_signal;    % create frequency depth points for plotting
 AmpfA = Ampf(1:length_td_signal/2+1);  % amplitudes of FFT STARTING FROM FIRST VALUE

 % store FFT data
 PnC_signal_FFT = [FA'*10^-6 abs(AmpfA)]; 
%  file_title_fft = sprintf(('FFT_data/PnC-signal-FFT_%s_%s_%s.csv'), PnC_dim, sample_no, freq_no);
%  writematrix(PnC_signal_FFT, file_title_fft);
  
 end
 
 
