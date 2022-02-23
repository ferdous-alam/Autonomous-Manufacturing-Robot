% -----------------------------------------------
%% Spectra visualizations
%% Md Ferdous Alam, 07/05/2021 
% -----------------------------------------------
d = (350: 50: 600); % diameter in micrometer 
lxy = (700: 50: 1050); % XY Spacing

desired = [0.85 0.15];
s = 0; 
spectra_exp = zeros(2000, 3); 

for i = 1:length(d)
    for j = 1:length(lxy)
        % FEM data
%         path_fem = 'N:\Research_files\Experimental_data\DOE_data_Zhi\FEM_data\FEM_spectra';
%         fft_data_fem = load(sprintf('%s_d%d_lxy%d.csv', path_fem, d(i), lxy(j)));
%         fft_data_fem = reshape(fft_data_fem, [length(fft_data_fem), 1]); 
%         fft_data_fem = fft_data_fem/max(fft_data_fem); 
%         f_fem = load('N:\Research_files\Experimental_data\DOE_data_Zhi\FEM_data\FEM_frequency.csv');
%         f_fem = reshape(f_fem, [length(f_fem), 1]); 
%         % experimental data
        path_exp = 'N:\Research_files\Experimental_data\DOE_data_Zhi\Zhi_paper_data\spectra_data\spectra'; 
        spectra_exp_sample1 = load(sprintf('%s_d%d_lxy%d_Sample1.csv', path_exp, d(i), lxy(j)));
        spectra_exp_sample2 = load(sprintf('%s_d%d_lxy%d_Sample2.csv', path_exp, d(i), lxy(j)));
        spectra_exp_sample3 = load(sprintf('%s_d%d_lxy%d_Sample3.csv', path_exp, d(i), lxy(j)));

        % take the average
        spectra_exp(:, 1) = spectra_exp_sample1;
        spectra_exp(:, 2) = spectra_exp_sample2;
        spectra_exp(:, 3) = spectra_exp_sample3;
        spectra = sum(spectra_exp, 2); 
        
        f_exp = load('N:\Research_files\Experimental_data\DOE_data_Zhi\Zhi_paper_data\spectra_data\frequency.csv');

        % normalize
        spectra = spectra(1:240); 
        spectra = spectra/max(spectra); 

        f_exp = f_exp(1:240)/1e6; 
        s = s + 1; 
        fprintf('sample#%d\n', s); 
        dim = [d(i) lxy(j)]; 
        type = 'FEM'; 
        plot_single_spectra(spectra, f_exp, desired, dim); 

    end
end
fprintf('################################# complete! ####################\n');


