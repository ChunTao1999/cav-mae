import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb


save_path = '/home/nano01/a/tao88/RoadEvent-shared/patent-grayscale'
if not os.path.exists(save_path): os.makedirs(save_path)

# Load wheelAccel array, plot
# wheelAccel_arr = np.load('/home/nano01/a/tao88/RoadEvent-shared/CV/events_7.26/wheelAccels/wheelAccel_session_75151_event_307.072.npy')
# plt.figure()
# plt.plot(wheelAccel_arr, linewidth=3, color='black')
# plt.axis('off')
# plt.savefig(os.path.join(save_path, 'wheelAccel.png'), bbox_inches='tight')
# pdb.set_trace()

# Plot spec
# sampling_freq = 500
# N_window_FFT = 32
# plt.figure()
# spectrum, freqs, t_bins, im = plt.specgram(x=wheelAccel_arr, 
#                                             NFFT=N_window_FFT, 
#                                             noverlap=0, 
#                                             Fs=sampling_freq, 
#                                             Fc=0,
#                                             mode='default',
#                                             scale='default',
#                                             scale_by_freq=True,
#                                             cmap='gray') # (17,16) or (33,8)
# plt.axis('off')
# plt.savefig(os.path.join(save_path, 'spec.png'), bbox_inches='tight')
# plt.close()


# Plot road event frame
rgb_img = cv2.imread('/home/nano01/a/tao88/RoadEvent-shared/patent-grayscale/event_307.072_frame_0_at_time_307.401_dist_5.083.png')
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(save_path, 'event_orig.png'), gray_img)
rgb_img = cv2.imread('/home/nano01/a/tao88/RoadEvent-shared/patent-grayscale/event_307.072_frame_0_at_time_307.401_dist_5.083_rv.png')
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(save_path, 'event_labeled_gt.png'), gray_img)
# draw the annotated figure with both ground-truth and predicted bounding boxes



pdb.set_trace()