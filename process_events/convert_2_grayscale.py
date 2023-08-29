import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb


save_path = '/home/nano01/a/tao88/RoadEvent-shared/patent-grayscale/fig7'
if not os.path.exists(save_path): os.makedirs(save_path)

# Load wheelAccel array, plot
wheelAccel_arr = np.load('/home/nano01/a/tao88/RoadEvent-Dataset/wheelAccels/wheelAccel_session_75366_event_388.375.npy')
plt.figure()
plt.plot(wheelAccel_arr, linewidth=3, color='black')
# plt.axis('off')
plt.savefig(os.path.join(save_path, 'rlwheelAccel.png'), bbox_inches='tight')
# pdb.set_trace()

# Plot spec
sampling_freq = 500
N_window_FFT = 32
N_overlap = 16
plt.figure()
spectrum, freqs, t_bins, im = plt.specgram(x=wheelAccel_arr, 
                                            NFFT=N_window_FFT, 
                                            noverlap=N_overlap, 
                                            Fs=sampling_freq, 
                                            Fc=0,
                                            mode='default',
                                            scale='default',
                                            scale_by_freq=True)
                                            # cmap='gray') # (17,16) or (33,8)
# plt.axis('off')
plt.savefig(os.path.join(save_path, 'spec_color.png'), bbox_inches='tight')
plt.close()


# Background to grayscale
rgb_img = cv2.imread('/home/nano01/a/tao88/RoadEvent-shared/patent-grayscale/fig7/Screenshot_2023-08-28_10-56-37.png')
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(save_path, 'UI.png'), gray_img)

pdb.set_trace()