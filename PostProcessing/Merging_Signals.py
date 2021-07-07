class processingMergingFFT:
    def TimeToFreq(self, d, lxy, sample):
        import math
        import csv
        import numpy as np
        from numpy import genfromtxt
        from datetime import datetime
        import os

        # log - based sequence
        starting_freq = math.log2(1e6) - 4 * (math.log2(4e6) - math.log2(1e6)) / 4
        incrementing_freq = (math.log2(4e6) - math.log2(1e6)) / 4
        ending_freq = math.log2(4e6) + (math.log2(4e6) - math.log2(1e6)) / 4
        freq_updated = starting_freq
        freq_full_range = []
        while freq_updated <= ending_freq:
            freq_full_range.append(2 ** freq_updated)
            freq_updated += incrementing_freq

        # freq_breaks for freq localizing basis functions
        ending_freq = math.log2(4e6) + 2 * (math.log2(4e6) - math.log2(1e6)) / 4
        freq_updated = starting_freq
        freq_breaks = []
        while freq_updated <= ending_freq:
            freq_breaks.append((3 / 5) * 2 ** freq_updated)
            freq_updated += incrementing_freq
        # raw waveform for merging
        data_freq_range = [1 * 1e6, 1.5 * 1e6, 2 * 1e6, 2.75 * 1e6, 4 * 1e6]
        num_freq = len(data_freq_range)
        Fs = 50e6  # sampling frequency: 50 MHz

        # choose the interested segment waveform
        lowerBoundary = 649     # PnC lower boundary
        upperBoundary = 2150    # PnC upper boundary
        L_master = 4000         # final length after FFT
        t = np.linspace(0, 80, 4000)

        # Frequency axis
        freq_temp = 0
        f = []
        while freq_temp <= math.floor(L_master / 2):
            f.append((Fs / L_master) * freq_temp)       # frequency axis
            freq_temp += 1

        sig_cum=np.zeros(int(L_master/2+1))                  # cumulative signal
        trace_data = []
        freq_data = 1000
        # read and merge data with filtering
        while freq_data<= 4000:
            # location of the input files
            TD_file= 'PnC-signal-TD_d' + str(d) + '_lxy'+ str(lxy)+ '_' + str(sample) + '_freq' +str(freq_data) + 'kHz.csv'
            file_path = '/Users/christinad/Desktop/ML/code/TD_data/'+ str(TD_file)
            TD_data =genfromtxt(file_path)
            ii = 0
            while ii< num_freq:
                # remove basis
                # sample average
                avgA = np.mean(TD_data[lowerBoundary-1:upperBoundary-1])
                trace_data = TD_data[lowerBoundary-1:upperBoundary-1] - avgA
                ii += 1
            # merging with filtering
            pmFFT = processingMergingFFT()
            act_cumA = pmFFT.Filtering(trace_data, Fs, num_freq, data_freq_range, freq_breaks, L_master)
            sig_cum = sig_cum + act_cumA
            freq_data += 250
        # merged average PnC Signal
        avg_sig = sig_cum/12
        # remove the first point
        freq = f[1:]        # the freq for plotting
        Merged_Avg_Sig = avg_sig[1:]    # the fft of the merged signal
        today = datetime.now()
        # folder name and file name of the output location
        foldername = '/Users/christinad/Desktop/ML/code/data/merged_data/' + str(sample) + '_' + str(today)
        os.mkdir(foldername)
        filename = str(foldername) + '/freq.domain.d'+ str(d) + '.lxy' + str(lxy) +'.csv'
        file_t = open(filename, 'w')
        writer = csv.writer(file_t)
        writer.writerow(Merged_Avg_Sig)


        return freq, Merged_Avg_Sig


    def plotAvgFFT(self, d, lxy, sample):
        from numpy import genfromtxt
        from scipy.fftpack import fft
        import matplotlib.pyplot as plt
        pmFFT = processingMergingFFT()
        freq, Merged_Avg_Sig = pmFFT.TimeToFreq(d, lxy, sample)
        plt.figure()
        plt.plot(freq, Merged_Avg_Sig)
        plt.title('Averged Signal with Merged Multiple Datasets')
        leg = str(sample) + '   d: ' + str(d) + '   lxy: ' + str(lxy)
        plt.legend(leg)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Amplitude')
        plt.xlim([0, 5e6])
        plt.show()

    def Filtering(self, trace_data, Fs, num_freq, Data_freq_range, freq_breaks, L_master):
        import math
        import numpy as np
        import cmath
        from scipy import signal
        from scipy.signal.ltisys import TransferFunction
        import control
        # log - based sequence
        starting_freq = math.log2(1e6) - 4 * (math.log2(4e6) - math.log2(1e6)) / 4
        incrementing_freq = (math.log2(4e6) - math.log2(1e6)) / 4
        ending_freq = math.log2(4e6) + (math.log2(4e6) - math.log2(1e6)) / 4
        freq_updated = starting_freq
        freq_full_range = []
        j = complex(0,1)
        while freq_updated <= ending_freq:
            freq_full_range.append(2 ** freq_updated)
            freq_updated += incrementing_freq
        # Frequency axis
        freq_temp = 0
        f = []
        while freq_temp <= math.floor(L_master / 2):
            f.append((Fs / L_master) * freq_temp)  # frequency axis
            freq_temp += 1
        # frequency localizing basis functions (Equation 35 of Welsh 2007)
        q = 0
        i = 0
        F = []

        while i < len(freq_full_range):
            # F(i+1) filter is for for F(i)
            # left part of the equation
            F.append(control.tf(freq_breaks[i+1]*2*np.pi, 1))
            k = 0
            while k < i+1:
                den = [1, freq_breaks[k]*2*np.pi]   #coefficient of (s+Pi)
                qq=0
                while qq < q-1:
                    # convolve() to get the coefficent of den 1/(s+Pi)^q
                    den = np.convolve(den, [1, freq_breaks[k]*2*np.pi])
                    qq += 1
                F[i] = np.dot(F[i],control.tf([1], den))
                k += 1
            bode_f = 2 * np.pi * np.asarray(f)

            [mag, phas, omega] = control.bode(F[i], bode_f)
            i += 1

        # merging data with selected frequency localizing basis functions
        act_cum = np.zeros(int(L_master/2+1))
        i = 0
        act = []
        wind = []
        pro_data = []
        pro_data_complex = [[]]
        while i < num_freq:
            L_orig=len(trace_data)
            act[0:L_orig-1]=trace_data
            L_zero_padding =L_master-L_orig
            # Signal and windowing
            wind[0:L_orig-1]= np.blackman(L_orig)
            act_winded = []
            index_act=0
            while index_act< len(wind):
                act_winded.append(wind[i]*act[i])
                index_act += 1
            # FFT

            L = len(act_winded)
            ACT = np.fft.fft(act_winded)
            ACT2= abs(ACT/L_orig)
            SSB = ACT2[1:int(np.floor(L/2))]
            SSB[1:len(SSB)-2] = [element * 2 for element in SSB[1:len(SSB)-2]]
            pro_data[i:]= SSB
            [mag, phas, omega]= control.bode(F[i+4], bode_f)
            # Filter data with frequency localizing basis functions
            k = 0
            while k < len(pro_data):
                pro_data_complex[k:]=[(complex(pro_data[i], 0)*mag[k]*cmath.exp(j*complex(phas[k], 0)*complex(np.pi/180,0)))]
                k += 1
            # take absolute value
            index_abs=0
            while index_abs < len(pro_data_complex):
                act_cum[index_abs:] = abs(pro_data_complex[index_abs])
                index_abs+=1
            i += 1
        return act_cum


    def plotTimeDomain(self, d, lxy, sample):
        import matplotlib.pyplot as plt
        import numpy as np
        from numpy import genfromtxt
        t = np.linspact(0, 80, 4000)
        freq_data = 1000
        # read and merge data with filtering
        while freq_data <= 4000:
            # location of the input files
            TD_file = 'PnC-signal-TD_d' + str(d) + '_lxy' + str(lxy) + '_' + str(sample) + '_freq' + str(
                freq_data) + 'kHz.csv'
            file_path = '/Users/christinad/Desktop/ML/code/TD_data/' + str(TD_file)
            TD_data = genfromtxt(file_path)
            # data check; plot time domain
            plt.figure(1)
            plt.plot(t, TD_data)
            plt.title('Time Domain d:' + str(d) + ' lxy:' + str(lxy))
            plt.xlabel('time [micro second]')
            plt.ylabel('Amplitude')

            freq_data += 250


    def plotTwo(self, d1, lxy1, d2, lxy2):
        import matplotlib.pyplot as plt

