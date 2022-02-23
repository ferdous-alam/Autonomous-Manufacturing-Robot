
%% -------------------------------------------------------- 
% Md Ferdous Alam, 06/16/2021
% HRL, OSU 
% Title: Saving experiment data by Zhi
% ---------------------------------------------------------


%% -------------------- time domain data ---------------------------------
time_domain_data = load('Zhi_paper_data\Paper_NEW_DOE_All_Ref_RawData_multi-freq.mat');

% ------------------------------------------------------------
% there are two samples for reference, extract data from both by changing 
% Sample number in all_tests; 
% example: ----> 
% all_tests = time_domain_data.Ref_RawData_TimeDomain.SampleA; 
% all_tests = time_domain_data.Ref_RawData_TimeDomain.SampleB;
% ------------------------------------------------------------

% waitbar
s = 0; 
wait_bar = waitbar(0,'Progress', 'Name','progress...',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(wait_bar,'canceling',0);


all_tests = time_domain_data.Ref_RawData_TimeDomain.SampleA; 
all_test_fields = fieldnames(all_tests); 
for i = 1: length(all_test_fields)
    test_freqs = all_tests.(all_test_fields{i}); 
    all_tested_freqs = fieldnames(test_freqs); 
    for j = 1: length(all_tested_freqs)
       Ref_TD_signal =  test_freqs.(all_tested_freqs{j}); 
       title = sprintf(('Reference_FFT_data/Ref-signal-FFT_%s_%s_%s.csv'), 'sample-A', all_test_fields{i}, all_tested_freqs{j});
       % save data 
       Ref_FFT_data = td2fft_ref(Ref_TD_signal, 540, 1101); 
       writematrix(Ref_FFT_data, title);
       
       % update waitbar
       s = s+1; 
       progress = s/(length(all_test_fields) * length(all_tested_freqs)); 
       waitbar(progress, wait_bar); 
    end
    
    
end
delete(wait_bar)  % delete wait bar 

fprintf('################################# complete! ####################\n');






 