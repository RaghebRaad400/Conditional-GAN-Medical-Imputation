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

G_model = generatorpix2pix(Input_W=x_W,Input_H=x_H,Input_C=x_C,Z_Dim=z_dim,act_param=act_param,reg_param=reg_param,out_channels=out_channels, layers=layers,ADDtanh=ADDtanh,Etype=Etype) 

glv = partial(get_lat_var,x_W=x_W,x_H=x_H,z_dim=z_dim)


test_x   = test_data[:][0:n_out,:,:,0:4]   #change all CODES!   go through all of this and fix channels
test_y   = tf.constant(test_data[:])[0:n_out,:,:,4::]


        # have a code that gives all MSE values then take file and work with it 



for checkpoint_id in range(750, 801, 50):
#for checkpoint_id in range(10, 71, 10):
  K.clear_session()
  G_model = generatorpix2pix(Input_W=x_W,Input_H=x_H,Input_C=x_C,Z_Dim=z_dim,act_param=act_param,reg_param=reg_param,out_channels=out_channels,layers=layers,ADDtanh=ADDtanh,Etype=Etype)
  G_model.load_weights(f'{GANdir}/checkpoints/G_checkpoint_{checkpoint_id}')
  j=checkpoint_id/50-1
 # j=checkpoint_id/10-1


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
      #sample_var  = sample_var  + (pred_-sample_mean)*(pred_-old_mean)
  #sample_var /= (n_z_stat-1)
  #sample_std = np.sqrt(sample_var)
  
  for i in range(148):    # loop for phase,  i range 37, j range 4, save all values in single np array 
    ncol = 1
    fig1,axs1 = plt.subplots(3, ncol, dpi=100, figsize=(20,20))  # pull inside
    ax1 = axs1.flatten()
    ax_ind = 0
    #axs = ax1[ax_ind]
    #pcm = axs.imshow(valid_y[t].numpy(),aspect='equal')
    #if t==0:
    #    axs.set_title(f'measurement',fontsize=30)
    #axs.axis('off')
    #ax_ind +=1

    axs = ax1[ax_ind]
    pcm = axs.imshow(test_x[i,:,:,i%4],aspect='equal',vmin=-1, vmax=1, cmap='pink')
        #if t==0:
    axs.set_title(f'True Image',fontsize=30)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(pcm,ax=axs,cax=cax)
    axs.axis('off')
    ax_ind +=1

#new one here
    axs = ax1[ax_ind]
    pcm = axs.imshow(test_y[i,:,:,i%4],aspect='equal',vmin=-1, vmax=1, cmap='pink')
        #if t==0:
    axs.set_title(f'Regression Image',fontsize=30)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(pcm,ax=axs,cax=cax)
    axs.axis('off')
    ax_ind +=1

#ends here
        
    axs = ax1[ax_ind]
    pcm = axs.imshow(sample_mean[i,:,:,i%4],aspect='equal',vmin=-1, vmax=1, cmap='pink')
        #if t==0:
    axs.set_title(f'Prediction, min={round(np.min(sample_mean[i,:,:,i%4]),3)},max={round(np.max(sample_mean[i,:,:,i%4]),3)}',fontsize=30)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)                
    fig1.colorbar(pcm,ax=axs,cax=cax)
    axs.axis('off')
    ax_ind +=1
    np.save(savedir+'/MEAN_{}_{}_{}.npy'.format(checkpoint_id,i,PARAMS.seed_no),sample_mean[i,:,:,i%4])



    fig1.tight_layout()
    fig1.savefig(f"{savedir}/test_stats_{datadir_main}_image_true+mean+stdONLY{i}_{checkpoint_id}_seed{PARAMS.seed_no}.png")    #pull inside  i shifted the spacing two tabs
    plt.close('all')
    


###  save np file and then extract later without needing any libraries, how does mean metrics evolve for each of the phases separately, ? we want to see how are we converging as training is evolving, how does trend change across seeds?  generate plots





print("----------------------- DONE --------------------------")



