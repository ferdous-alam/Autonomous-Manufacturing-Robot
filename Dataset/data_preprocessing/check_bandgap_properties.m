% FEM bandgap properties: uncomment the following section if FEM bandgap properties are needed
% path for FEM data
path_fem = 'N:\Research_files\Experimental_data\DOE_data_Zhi\FEM_data\FEM_spectra';

% memory allocation
d_FEM = 300:25:635;
lxy_FEM = 700:25:1035;
% create data for saving, format --> [lxy, dia, peak_frequency, band_width]
bandgap_properties_FEM = zeros(length(d_FEM)*length(lxy_FEM), 4);   
index_FEM = 0;

% main loop
fprintf('calculating . . .');
for i = 1:length(d_FEM)
    for j = 1:length(lxy_FEM)
          index_FEM = index_FEM + 1; 
          % uncomment the following section if experimental bandgap
          % properties are needed
          spectra_FEM = load(sprintf('%s_d%d_lxy%d.csv', path_fem, d_FEM(i), lxy_FEM(j)));
          spectra_FEM = reshape(spectra_FEM, [length(spectra_FEM), 1]); 
          frequency_FEM = load('N:\Research_files\Experimental_data\DOE_data_Zhi\FEM_data\FEM_frequency.csv');
          frequency_FEM = reshape(frequency_FEM, [length(frequency_FEM), 1]); 
          [peak_freq_FEM, band_width_FEM] = calculate_bandgap_properties_FEM(frequency_FEM, spectra_FEM);
          bandgap_properties_FEM(index_FEM, :) = [lxy_FEM(j), d_FEM(i), peak_freq_FEM, band_width_FEM]; 
    end
    fprintf('progress: %0.4f%%\n', i/length(d_FEM));
end
fprintf('finished!'); 
save('bandgap_properties_FEM_shortened.mat', 'bandgap_properties_FEM');
% plot features
plot_features(bandgap_properties_FEM, 'Source');

%% Experimental bandgap properties: uncomment the following section if experimental bandgap properties are needed

% % path for experimental data
% path_exp = 'N:\Research_files\Experimental_data\DOE_data_Zhi\Zhi_paper_data\spectra_data\spectra';
% % dimensions 
% d_exp = (350: 50: 600); % diameter in micrometer 
% lxy_exp = (700: 50: 1050); % XY Spacing
%  
% % only two bandgap properties, 3 samples at each dimension
% % memory allocation
% bandgap_properties_exp = zeros(length(d_exp)*length(lxy_exp)*3, 4);   
% index_exp = 0;
% 
% % main loop
% fprintf('calculating . . .');
% for i = 1:length(d_exp)
%     for j = 1:length(lxy_exp)
%         for k = 1:3
%             index_exp = index_exp + 1;
%             spectra_exp = load(sprintf('%s_d%d_lxy%d_Sample%d.csv', path_exp, d_exp(i), lxy_exp(j), k));
%             frequency_exp = load('N:\Research_files\Experimental_data\DOE_data_Zhi\Zhi_paper_data\spectra_data\frequency.csv');
%             [peak_freq_exp, band_width_exp] = calculate_bandgap_properties_exp(frequency_exp, spectra_exp); 
%             peak_freq_exp = peak_freq_exp/1e6; 
%             band_width_exp = band_width_exp/1e6; 
%             bandgap_properties_exp(index_exp, :) = [lxy_exp(j), d_exp(i), peak_freq_exp, band_width_exp]; 
%         end
%     end
%     fprintf('progress: %0.4f%%\n', i/length(d_exp));
% end
% plot_features(bandgap_properties_exp, 'Target');
% save('bandgap_properties_exp.mat', 'bandgap_properties_exp');
% 
% 
% 
% 
% 
% 
