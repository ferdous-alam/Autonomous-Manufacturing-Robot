%% -------------------------------------------------------- 
% Md Ferdous Alam, 06/16/2021
% HRL, OSU 
% Title: Saving experiment data by Zhi
% ---------------------------------------------------------

% ------------ waitbar -----------
f = waitbar(0,'1','Name','Approximating pi...',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(f,'canceling',0);
% --------------------------------

freq_vals = [1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000]; 
samples = [1 2 3]; 
d = 350:50:400;
lxy = 700:50:750;

step = 0; 

for i=1:length(d)
    for j=1:length(lxy)
        for s=1:length(samples)
            for freq=1:length(freq_vals)
                path_td = 'N:\Research_files\Experimental_data\DOE_data_Zhi\TD_data\PnC-signal-TD';
                path_fft = 'N:\Research_files\Experimental_data\DOE_data_Zhi\FFT_data\PnC-signal-FFT';
                td_signal = load(sprintf('%s_d%d_lxy%d_Sample%d_freq%dkHz.csv', path, d(i), lxy(j),  samples(s), freq_vals(freq))); 
%                 fft_signal = load(sprintf('%s_d%d_lxy%d_Sample%d_freq%dkHz.csv', path, d(i), lxy(j),  samples(s), freq_vals(freq))); 
%                 plot_signals(td_signal, fft_signal, PnC_dim, sample_no, freq_no) 
                
                % --- waitbar -------
                  % Update waitbar and message
                step = step + 1; 
                msg = 'plotting';
                steps = length(d)*length(lxy)*length(samples)*length(freq_vals);
                waitbar(step/steps,f,sprintf('completed: %f%%',step/steps*100));
                % -------------------
            end 
        end 
    end
end
delete(f); 
        


fprintf('################################# complete! ####################\n');



function plot_signals(td_signal, fft_signal, PnC_dim, sample_no, freq_no) 
 set(0,'DefaultFigureVisible','off'); 

 %% --------- plot figure ------------
 % ----- set latex font everywhere -------------
 set(groot,'defaultAxesTickLabelInterpreter','latex');   
 set(groot,'defaulttextinterpreter','latex');
 set(groot,'defaultLegendInterpreter','latex');
 % ----------------------------------------------
 figure(); 
 ax = gca; 
 % plot time domain signal
 subplot(2, 1, 1); 
 plot(td_signal); 
 xlabel('frequency (MHz)', 'FontSize', 14); 
 ylabel('amplitude', 'FontSize', 14); 
 plot_title_td = sprintf(('Time domain PnC signal %s'), PnC_dim);
 title(plot_title_td, 'FontSize', 14); 

 % plot fft of the time domain signal
 subplot(2, 1, 2); 
 plot(FA*10^-6,abs(AmpfA),'b','LineWidth',1);
 xlabel('frequency (MHz)', 'fontsize', 14);
 ylabel('amplitude (au)', 'fontsize', 14);
 plot_title_fft = sprintf(('FFT PnC signal %s'), PnC_dim);
 title(plot_title_fft, 'FontSize', 14); 
 xlim([0.2 4.0]);
 % ------ save plot ------
 file_title = sprintf(('Figures/PnC-signal_%s_%s_%s'), PnC_dim, sample_no, freq_no);  
 print(file_title, '-dpng', '-r300'); 
 end