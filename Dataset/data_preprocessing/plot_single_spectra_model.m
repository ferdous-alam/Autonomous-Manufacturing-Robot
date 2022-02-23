%%
% single plot of a model against the true experimental spectra
%%


lxy = 750; 
dia = 550; 
k = 3;   % sample number
spectra = zeros(2000, 3); 
for k = 1:3
    path_exp = 'N:\Research_files\Experimental_data\DOE_data_Zhi\Zhi_paper_data\spectra_data\spectra';
    spectra_exp = load(sprintf('%s_d%d_lxy%d_Sample%d.csv', path_exp, dia, lxy, k));
    spectra(:, k) = spectra_exp; 
end    
frequency_exp = load('N:\Research_files\Experimental_data\DOE_data_Zhi\Zhi_paper_data\spectra_data\frequency.csv');
avg_spectra = sum(spectra, 2);   % along axis 2
avg_spectra = avg_spectra(1:240); 
frequency_exp = frequency_exp(1:240); 

[model, PeakFreq, Bandwidth] = calculate_bandgap_properties_exp(frequency_exp, avg_spectra); 
plot(model, frequency_exp, avg_spectra); 
xline(PeakFreq + Bandwidth, 'LineWidth', 1.5);
xline(PeakFreq - Bandwidth, 'LineWidth', 1.5);
title(sprintf('fc: %0.4f, w: %0.4f', PeakFreq/1e6, Bandwidth/1e6)); 
fprintf('fc: %0.4f, w: %0.4f', PeakFreq/1e6, Bandwidth/1e6);



function [model, PeakFreq, Bandwidth] = calculate_bandgap_properties_exp(frequency, spectra)

% Pcenter_desired range: [0.40, 1.30] MHz        --> (40, 130)e4  Hz
% BandWidth_desired range: [0.07, 0.30] MHz      --> (7, 30)e4  Hz

freqRange = [40, 130]; 
freq_random = randsample(freqRange(1):freqRange(2), 1);
bandwidthRange = [7, 30];
bandwidth_random = randsample(bandwidthRange(1):bandwidthRange(2),1);


% weight of loss value 
loss1_freq_weight = 1;                       % peak frequency
loss2_bandwidth_weight = 1;                  % bandwidth

%%
cutoff = -3;             % in dB scale          

%% Two-term Gaussian equation 
gaussEqn = 'a1*exp(-((x-b1)/(sqrt(2)*c1))^2)+a2*exp(-((x-b2)/(sqrt(2)*c2))^2)';
% [a1 a2 b1 b2 c1 c2]
startPoints = [0.1 0.1 0.5e6 1e6 1e5 1e5];
lowerPoints = [0 0 0.2e6 0.2e6 0.03e6 0.03e6]; 
upperPoints = [Inf Inf 2e6 2e6 0.8e6 0.8e6]; 

Position = reshape(1:48, 8,6)';
f = frequency; 
        
% magnitude threshold of the response
Threshold = 0.02*mean(spectra);
% frequency axis 
%% Gaussian fitting 
model = fit(f, spectra, gaussEqn,...
            'Start', startPoints, 'Lower', lowerPoints, 'Upper', upperPoints, 'Exclude', f > 2e6, 'Exclude', spectra < Threshold);

        
[PeakFreq, Bandwidth] = FindBandWidth(model, cutoff); 

end

        
       
   
%% functions 
% Find the peak frequency and bandwidth of the signal (closed-form)

function [PeakFreq, Bandwidth] = FindBandWidth(model, cutoff)
% model is the fitted two-term Gaussian model  
% a1*exp(-((x-b1)/(sqrt(2)*c1))^2)+a2*exp(-((x-b2)/(sqrt(2)*c2))^2
% cutoff is threshold in dB scale (-3dB cutoff) 

    B = 1/db2mag(cutoff); 
    coeff = coeffvalues(model);     
    a1 = coeff(1);
    a2 = coeff(2);
    b1 = coeff(3);
    b2 = coeff(4);
    c1 = coeff(5);
    c2 = coeff(6);
    
    if a1>= a2                     % determine the dominating Gaussian eqn
        PeakFreq = b1; 
        Bandwidth = 2*c1*sqrt(2*log(B));
    else 
        PeakFreq = b2; 
        Bandwidth = 2*c2*sqrt(2*log(B));
    end 
end 


