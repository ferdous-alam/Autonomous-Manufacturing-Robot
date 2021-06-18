class processingFFT:
    def calcfft(self, filename):
        from numpy import genfromtxt
        from scipy.fftpack import fft
        import numpy as np
        A = genfromtxt(filename)

        # I need to fix the ranges 6/21
        SigRangeA_lower = 550
        SigRangeA_upper = 1050
        Sampling_frequency = 50 * 10 ** 6  # Sampling Frequency (MHz): 50; 25; 12.5; 6.25

        n = len(A) + 1  # number of data points !for some reason len function is returning 3999 instead of 4000
        Total_time = n / Sampling_frequency  # total time
        dt = Total_time / n  # difference in time for plotting

        # time for measurements of A and B csv files, goes to n-1 bc at n it repeats !check this
        index = 0
        t = [None] * n
        while index < n:
            t[index] = (dt * index * 10 ** 6)
            index += 1

        # calculating FFT
        df = 1 / dt  # difference between frequency

        # set mean value before and after pulse, eliminates noise and reflections for A and B
        No_Reflections_A = [np.mean(A)] * n  # array of mean A and pulse
        No_Reflections_A[SigRangeA_lower:SigRangeA_upper] = A[SigRangeA_lower:SigRangeA_upper]

        # take fft A
        fft_A = abs(fft(No_Reflections_A))
        index = int(n / 2)
        while len(fft_A) != index:
            fft_A = np.delete(fft_A, index)

        # create frequency depth points for plotting
        frq_A_points = []
        index = 0
        while index < (int(n / 2)):
            frq_A_points.append(index * df / n)
            index += 1
        frq_A_plot = []
        for frq in frq_A_points:
            frq_A_plot.append(frq * 10 ** -6)

        return fft_A, frq_A_plot

    def plot(self, filename):
        import matplotlib.pyplot as plt
        ppfft = processingFFT()
        fft, frq = ppfft.calcfft(filename)
        fft_max = max(fft[2:len(fft)])
        plt.plot(frq, fft)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Amplitude [au]')
        plt.title('Frequency Domain')
        plt.xlim([0.2, 5])
        plt.ylim([0, fft_max])
        plt.legend(['Sample (d=0.00mm, a=0.00mm)'])
        plt.grid()
        # save frequency domain plot
        plt.savefig('_FreqDomain.jpeg', bbox_inches='tight')
        plt.show()

    def plotWithReference(self, filenameRef, filename):
        import matplotlib.pyplot as plt
        ppfft = processingFFT()
        fftA, frqA = ppfft.calcfft(filenameRef)
        fftB, frqB = ppfft.calcfft(filename)
        fftA_max = max(fftA[2:len(fftA)])
        fftB_max = max(fftB[2:len(fftB)])
        fft_max =fftB_max
        if fftA_max > fftB_max:
            fft_max = fftA_max

        plt.plot(frqA, fftA)
        plt.plot(frqB, fftB)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Amplitude [au]')
        plt.title('Frequency Domain')
        plt.xlim([0.2, 5])
        plt.ylim([0, fft_max])
        plt.legend(['Reference Signal', 'Sample (d=0.00mm, a=0.00mm)'])
        plt.grid()
        # save frequency domain plot
        plt.savefig('_FreqDomain.jpeg', bbox_inches='tight')
        plt.show()
