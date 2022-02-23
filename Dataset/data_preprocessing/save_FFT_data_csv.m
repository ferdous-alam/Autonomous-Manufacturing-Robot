
%% -------------------------------------------------------- 
% Md Ferdous Alam, 06/16/2021
% HRL, OSU 
% Title: Saving experiment data by Zhi
% ---------------------------------------------------------


%% -------------------- time domain data ---------------------------------
time_domain_data = load('Zhi_paper_data\Paper_NEW_DOE_All_PnC_RawData_multi-freq.mat');
all_PnC_dim = fieldnames(time_domain_data.Ref_RawData_TimeDomain); 
for i = 1:length(all_PnC_dim)
    all_samples = fieldnames(time_domain_data.Ref_RawData_TimeDomain.(all_PnC_dim{i})); 
    for s = 1:length(all_samples)
        all_tested_freqs = fieldnames(time_domain_data.PnC_RawData_TimeDomain.(all_Red_dim{i}).(all_samples{s})); 
        for k =1:length(all_tested_freqs)    
            td_signal = time_domain_data.PnC_RawData_TimeDomain.(all_PnC_dim{i}).(all_samples{s}).(all_tested_freqs{k});
            title = sprintf(('Reference_TD_data/Ref-signal-TD_%s_%s_%s.csv'), all_PnC_dim{i}, all_samples{s}, all_tested_freqs{k});
            % save data 
            writematrix(td_signal, title);
            
            sample_no = all_samples{s}; 
            freq_no = all_tested_freqs{k}; 
            PnC_dim = all_PnC_dim{i}; 
            td2fft(td_signal, 649, 2150, PnC_dim, sample_no, freq_no); 
            fprintf(title); 
            fprintf('\n'); 
        end
    end
   
end
fprintf('################################# complete! ####################\n');








 