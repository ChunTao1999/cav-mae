from utils import calibrate_camera, define_perspective_transform, plot_image, plot_conf_matrix, plot_loss_curves
import os
import pdb

exp_name = f"EventResnetreg_{20}_{0.001}_new"
exp_path = os.path.join("/home/nano01/a/tao88/cav-mae/process_events/train_results", exp_name)
# plot_loss_curves(exp_path)
plot_conf_matrix(exp_path, [], [])
pdb.set_trace()