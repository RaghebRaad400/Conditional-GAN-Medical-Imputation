import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from scipy.linalg import qr
from sklearn.cluster import MeanShift, estimate_bandwidth
from functools import partial
from config import cla
from modelsDENSEpix2pix import *
from utilspix2pix import *
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linalg as LA
from tensorflow.keras import backend as K
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import skimage

PARAMS = cla()

np.random.seed(PARAMS.seed_no)

#============== Parameters ======================
n_epoch     = PARAMS.n_epoch
n_critic    = PARAMS.n_critic
z_dim       = PARAMS.z_dim
batch_size  = PARAMS.batch_size
n_train     = PARAMS.n_train
log_iter    = np.int(np.floor(n_train/batch_size))
gp_coef     = PARAMS.gp_coef
save_suffix = PARAMS.save_suffix
data_suffix = PARAMS.data_suffix
datadir     = PARAMS.datadir
#checkpoint_id = PARAMS.checkpoint_id
main_dir    = PARAMS.main_dir
n_z_stat    = PARAMS.MC_samples    # Number of 'z' samples to evaluate the statistics
n_out       = PARAMS.sample_plots  # Number of test samples pairs to work with 
reg_param   = PARAMS.reg_param
patient_id  = PARAMS.patient_id
act_param=PARAMS.act_param
Etype=PARAMS.Etype
ADDtanh = PARAMS.ADDtanh

out_channels=16
layers=3

GANdir = f"exps/Nsamples{n_train}_Ncritic{n_critic}_LAMBDA{PARAMS.LAMBDA}_Etype{Etype}_ADDtanh{ADDtanh}_BS{batch_size}_Nepoch{n_epoch}_seed{PARAMS.seed_no}_GP{gp_coef}{save_suffix}"    # add seed 

datadir_main = get_main_dir(datadir)
savedir = f"{GANdir}/Testing{data_suffix}"
if not os.path.exists(savedir):
    os.makedirs(savedir)    


# @Ragheb: Replace this with the way you load the 'validation data' not
# training data
test_data, N_data, x_W, x_H,x_C = read_test_images_mat(img_dir=datadir)  
    
    
print('\n USING CONDITIONAL GAN MODEL:\n')

G_model = generatorpix2pix(Input_W=x_W,Input_H=x_H,Input_C=x_C,Z_Dim=z_dim,act_param=act_param,reg_param=reg_param,out_channels=out_channels,layers=layers,ADDtanh=ADDtanh,Etype=Etype) 

glv = partial(get_lat_var,x_W=x_W,x_H=x_H,z_dim=z_dim)


test_x   = test_data[:][0:n_out,:,:,0:4]   #change all CODES!   go through all of this and fix channels
test_y   = tf.constant(test_data[:])[0:n_out,:,:,4::]


MSEcheckpoints=np.zeros((148,16))        # have a code that gives all MSE values then take file and work with it 
SSIMcheckpoints=np.zeros((148,16))
PSNRcheckpoints=np.zeros((148,16))

#MSEcheckpoints=np.zeros((148,9))        # have a code that gives all MSE values then take file and work with it
#SSIMcheckpoints=np.zeros((148,9))
#PSNRcheckpoints=np.zeros((148,9))


for checkpoint_id in range(750, 801, 50):
  K.clear_session()
  G_model = generatorpix2pix(Input_W=x_W,Input_H=x_H,Input_C=x_C,Z_Dim=z_dim,act_param=act_param,reg_param=reg_param,out_channels=out_channels,layers=layers,ADDtanh=ADDtanh,Etype=Etype)


  G_model.load_weights(f'{GANdir}/checkpoints/G_checkpoint_{checkpoint_id}')
  j=int(checkpoint_id/50-1)
 # j=int(checkpoint_id/10-1)


  print(f"Computing mean and variance")      
  sample_mean = np.zeros((n_out,x_W,x_H,x_C))     
  sample_var  = np.zeros((n_out,x_W,x_H,x_C))

  MSEvalues=[]
  SSIMvalues=[]
  PSNRvalues=[]
  for i in range(n_z_stat):
      print(f"z sample : {i+1}")
      test_z = glv(batch_size=n_out)
      pred_ = G_model([test_y,test_z],training=None).numpy()[:,:,:,0:4]
      currentX=pred_[patient_id,:,:,patient_id%4]
      currentJ=0.5*(currentX+1)
      normJ=LA.norm(currentJ,'fro')

            
      old_mean    = np.copy(sample_mean)
      sample_mean = sample_mean + (pred_-sample_mean)/(i+1)
      sample_var  = sample_var  + (pred_-sample_mean)*(pred_-old_mean)
  #sample_var /= (n_z_stat-1)
  #sample_std = np.sqrt(sample_var)
  sample_mean=np.clip(sample_mean,-1,1)
  
  for i in range(148):    # loop for phase,  i range 37, j range 4, save all values in single np array 
    MSEcheckpoints[i,j]=LA.norm(test_x[i,:,:,i%4]-sample_mean[i,:,:,i%4],'fro')/LA.norm(test_x[i,:,:,i%4],'fro')
    #MSEvalues.append(MSE)
    
    SSIMcheckpoints[i,j]=ssim(0.5*(test_x[i,:,:,i%4]+1), 0.5*(sample_mean[i,:,:,i%4]+1), data_range=1)
    #SSIMvalues.append(SSIM)
    
    PSNRcheckpoints[i,j]=skimage.metrics.peak_signal_noise_ratio(0.5*(test_x[i,:,:,i%4]+1), 0.5*(sample_mean[i,:,:,i%4]+1), data_range=1)
    #PSNRvalues.append(PSNR)
    
  #MSEcheckpoints.append(sum(MSEvalues)/len(MSEvalues))    #not necessary, do it later in a separate code 
  #SSIMcheckponts.append(sum(SSIMvalues)/len(SSIMvalues))
  #PSNRcheckpoints.append(np.mean(np.PSNRvalues))
    


###  save np file and then extract later without needing any libraries, how does mean metrics evolve for each of the phases separately, ? we want to see how are we converging as training is evolving, how does trend change across seeds?  generate plots


#save_loss(MSEcheckpoints,'MSE',savedir,n_epoch)
#save_loss(SSIMcheckponts,'SSIM',savedir,n_epoch)  
#save_loss(PSNRcheckpoints,'PSNR',savedir,n_epoch) 
np.save(savedir+'/MSE_{}_750.npy'.format(PARAMS.seed_no),MSEcheckpoints)
np.save(savedir+'/SSIM_{}_750.npy'.format(PARAMS.seed_no),SSIMcheckpoints)
np.save(savedir+'/PSNR_{}_750.npy'.format(PARAMS.seed_no),PSNRcheckpoints)




print("----------------------- DONE --------------------------")



