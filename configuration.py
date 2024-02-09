import torch
import data as d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lambda_mse = 0.1
grad_clamp = 15
eval_test = 1
lr_init = 1.0e-3
batch_size = 2000
n_epochs = 100
pre_low_lr  = 0 
final_decay = 0.02 
l2_weight_reg  = 1e-5 # L2 weight regularization of model parameters
adam_betas = (0.9, 0.95) # Parameters beta1, beta2 of the Adam optimizer

##  Data dimensions ##
pad_dim = 0
ndim_x = d.input
ndim_pad_x = pad_dim

ndim_y = d.output
ndim_z = ndim_x - ndim_y
ndim_pad_zy = pad_dim

train_forward_mmd = True
train_backward_mmd = True
train_reconstruction = False
train_max_likelihood = False

lambd_fit_forw = 1
lambd_mmd_forw = 1
lambd_reconstruct = 1
lambd_mmd_back = 1
lambd_max_likelihood = 1

# Both for fitting, and for the reconstruction, perturb y with Gaussian
# noise of this sigma
add_y_noise = 5e-3
# For reconstruction, perturb z
add_z_noise = 2e-3
# In all cases, perturb the zero padding
add_pad_noise = 1e-3
# MLE loss
zeros_noise_scale = 5e-3

# For noisy forward processes, the sigma on y (assumed equal in all dimensions).
# This is only used if mmd_back_weighted of train_max_likelihoiod are True.
y_uncertainty_sigma = 0.12 * 4

mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
mmd_back_weighted = False

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.10
#
N_blocks = 6 
#
exponent_clamping = 2.0
#
hidden_layer_sizes = 256
#
use_permutation = True
#
verbose_construction = False
