% -----------------------------------------------
%% Spectra visualizations
%% Md Ferdous Alam, 07/05/2021 
% -----------------------------------------------

function plot_single_spectra_comparison(lxy_val, d_val)
d = (350: 50: 600); % diameter in micrometer 
lxy = (700: 50: 1000); % XY Spacing
desired = [0.85, 0.15]; 

s = 0; 
i = find(d==d_val); j = find(lxy==lxy_val); 

% FEM data
path_fem = 'N:\Research_files\Experimental_data\DOE_data_Zhi\FEM_data\FEM_spectra';
fft_data_fem = load(sprintf('%s_d%d_lxy%d.csv', path_fem, d(i), lxy(j)));
fft_data_fem = reshape(fft_data_fem, [length(fft_data_fem), 1]); 
fft_data_fem = fft_data_fem/max(fft_data_fem); 
f_fem = load('N:\Research_files\Experimental_data\DOE_data_Zhi\FEM_data\FEM_frequency.csv');
f_fem = reshape(f_fem, [length(f_fem), 1]); 
% experimental data
path_exp = 'N:\Research_files\Experimental_data\DOE_data_Zhi\Zhi_paper_data\spectra_data\spectra'; 
spectra_exp_sample1 = load(sprintf('%s_d%d_lxy%d_Sample1.csv', path_exp, d(i), lxy(j)));
spectra_exp_sample2 = load(sprintf('%s_d%d_lxy%d_Sample2.csv', path_exp, d(i), lxy(j)));
spectra_exp_sample3 = load(sprintf('%s_d%d_lxy%d_Sample3.csv', path_exp, d(i), lxy(j)));

% take the average
spectra_exp = (spectra_exp_sample1 + spectra_exp_sample2 + spectra_exp_sample3) / 3;         
f_exp = load('N:\Research_files\Experimental_data\DOE_data_Zhi\Zhi_paper_data\spectra_data\frequency.csv');

% normalize
spectra_exp = spectra_exp(1:240); 
spectra_exp = spectra_exp/max(spectra_exp); 

f_exp = f_exp(1:240)/1e6; 
s = s + 1; 
dim = [d(i) lxy(j)];  
plot_spectra_comparison(fft_data_fem, f_fem, spectra_exp, f_exp, dim, desired); 

end

