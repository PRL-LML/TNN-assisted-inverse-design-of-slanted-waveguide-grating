import torch
import numpy as np
import model
import scaling as s
from torch.autograd import Variable
import configuration as c
import data as d
from joblib import dump, load
import time
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filename = d.pklname
pretrained_model = torch.load(filename, map_location = lambda storage, loc: storage)
generator = model.model
generator.load_state_dict(pretrained_model)

MLP_gene = load('MLP_Red.joblib') 

def pre_transmittance(X):
    X = s.scaler_X.transform([X])
    transmittance = MLP_gene.predict(X)
    final = s.scaler_y.inverse_transform(transmittance)
    return np.round(final, 3)

# Initialize the value of k, where k is the maximum transmittance.
def absolute_error(target, prediction):
    return abs(target - prediction)

# def imbalance(min_T, max_T):
#     if min_T <= 0 or max_T <= 0:
#         return float('inf')  
#     else:
#         dB = abs(10 * math.log10(min_T / max_T))
#         return dB

def uniformity(min_T, max_T):
    uniform = 1 - (max_T - min_T ) / (max_T + min_T)
    return uniform

k = 1
rate = 0.0001

start_time = time.time()
while True:
    y_target = [k]  
    y_transform = d.scaler_y.transform([y_target])

    n_samps = 1

    y_fix = np.zeros((n_samps, len(y_target))) + y_transform
    y_fix = torch.tensor(y_fix, dtype=torch.float)
    y_fix = torch.cat([torch.randn(n_samps, c.ndim_z), c.add_z_noise * torch.zeros(n_samps, 0), y_fix], dim=1)
    y_fix = y_fix.to(device)

    # posterior samples
    rev_x0 = generator(y_fix, rev=True)[0]
        
    ## Save the predicted X ###
    rev_x = rev_x0.cpu().data.numpy()
    rev_x = torch.tensor(d.inverse_transform_x(rev_x))
    rev_x = torch.mean(rev_x, dim=0)
    rev_x = np.array(rev_x.detach().cpu())
        
    prediction_T = pre_transmittance(rev_x)
    max_T = np.max(prediction_T)
    ave_T = np.mean(prediction_T)
    min_T = np.min(prediction_T)
    
#     dB = imbalance(min_T, max_T)
    uniform = uniformity(min_T, max_T)
    
    max_APE = absolute_error(k, max_T)
    ave_APE = absolute_error(k, ave_T)
    uni_APE = absolute_error(k, uniform)
       
    if d.grating in ['00', '10', '20']:
        APE = max_APE
    elif d.grating in ['01', '11', '21']:
        APE = ave_APE
    elif d.grating in ['02', '12', '22']:
        APE = uni_APE  
    
    if APE <= 0.005:
#     if APE <= 0.005 and ave_T >= 0.88:
        break
    else:
        k = k - rate

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.1f} s".format(elapsed_time))

fname = 'gen_samps_inn_x.csv' 
np.savetxt(fname, rev_x, fmt='%.5f', delimiter=',')

print('FF	H	P	B	A:\n', rev_x)
print('Predicted T:', prediction_T)
print('Specified T:', k)
print('Predicted max_T:', max_T)
# print('Predicted uniform:', uniform)
# print('Predicted ave_T:', ave_T)
# print('Uniformity:', uniform)
# print('Gap:', dB)



# while True:
#     y_target = [k]  
#     y_transform = d.scaler_y.transform([y_target])
# 
#     n_samps = 10000
#     
#     y_fix = np.zeros((n_samps, len(y_target))) + y_transform
#     y_fix = torch.tensor(y_fix, dtype = torch.float)
#     y_fix = torch.cat([torch.randn(n_samps, c.ndim_z), c.add_z_noise * torch.zeros(n_samps, 0), y_fix], dim=1)
#     y_fix = y_fix.to(device)
# 
#     # posterior samples
#     rev_x0 = generator(y_fix, rev=True)[0]
#     
#     ## Save the predicted X ###
#     rev_x = rev_x0.cpu().data.numpy()
#     rev_x = torch.tensor(d.inverse_transform_x(rev_x))
#     rev_x = torch.mean(rev_x, dim=0)
#     rev_x = np.array(rev_x.detach().cpu())
#     
#     fname = 'gen_samps_inn_x.csv' 
#     np.savetxt(fname, rev_x, fmt='%.5f', delimiter=',')
# 
#     prediction_T = m.pre_transmittance([rev_x])
#     max_T = np.max(prediction_T)
#     print(max_T)
#     
#     APE = absolute_percentage_error(k, max_T)
#     if APE > 0.1:
#         k = k - 0.0001 
#     else:
#         break

# print('X:', rev_x)
# print('Optimized T:', k)
# print('Predicted T:', prediction_T)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Elapsed time: {:.1f} s".format(elapsed_time))



