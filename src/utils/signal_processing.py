import numpy as np 
import pywt
import pywt.data
from scipy.signal import butter, lfilter
class SignalProcess:
    def __init__(self):
        self.fs = 800000 * 50
        self.high_cut = 10e4
        
    def butter_highpass(self,  order=3):
        nyq = 0.5 * self.fs
        high = self.high_cut / nyq
        b, a = butter(order, high, btype='high')
        return b, a

    def butter_highpass_filter(self,data, order=3):
        b, a = self.butter_highpass(order=order)
        y = lfilter(b, a, data)
        return y
    
    def denoise_signal(self,signal, wavelet="db4"):
        """Denoise filtered spectrum using wavelet transform. 
        The following is performed:
        1. Obtain detail coefficients (cD) from discrete wavelet transform
        2. Determine middle point of cD (the median is the convention)
        3. Calculate sigma and the threshold level as described 
        in the thesis on pp. 19-20
        4. Apply hard thresholding as described in the thesis on p. 19
        5. Perform inverse discrete wavelet transform to obtain 
        the denoised spectrum
        """ 
        _, cD = pywt.dwt(signal, wavelet, mode="per")
        # Center point; take to be median
        median = np.median(cD)
        # MAD calculation
        mad = np.mean(np.abs(np.abs(cD) - np.median(cD)))
        # Sigma calculation
        sigma = 1.0 / 0.6745 * mad
        # Threshold calculation (log10 works better than log (base e))
        td = sigma * np.sqrt(2.0 * np.log10(len(cD)))
        # Apply Threshold
        cD_t = pywt.threshold(cD, td, mode='hard', substitute=0)
        # Reconstruct signal (denoised)
        reconstructed = pywt.idwt(cA=None, cD=cD_t, wavelet="db4", mode='per')
        return reconstructed
    def corona_denoise(self,data):
    	data_size = len(data)
    	indices = np.arange(0,data_size)
    	non_zero_array = np.abs(data) >= 50
    	non_zero = data[non_zero_array]
    	non_zero_indices = indices[non_zero_array]
    	ratio = np.divide(non_zero[1:],non_zero[:-1])
    	distance = np.diff(non_zero_indices)
    	beyond_max_ratio = ratio <= -1
    	within_max_distance = distance <= 10
    	corona_start = non_zero_indices[:-1][(non_zero[:-1] > 0) & within_max_distance & beyond_max_ratio] 
    	ranges = [np.arange(i,i + 100) for i in corona_start]
    	if ranges:
            zero_indices = np.unique(np.concatenate(ranges))
            zero_indices = zero_indices[zero_indices < data_size]
            copied = np.copy(data)
            copied[zero_indices] = 0
    	else:
            copied = np.copy(data)
    	return copied
