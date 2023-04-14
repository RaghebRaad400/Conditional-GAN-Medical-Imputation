import os
from config import cla
PARAMS = cla()
os.environ['PYTHONHASHSEED']=str(PARAMS.seed_no)

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from functools import partial
import time
from os.path import dirname, join as pjoin
from modelsDENSE import *
from utils import *
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random


random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
tf.random.set_seed(PARAMS.seed_no)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)


print('\n ============== LAUNCHING TRAINING SCRIPT =================\n')

#============== Parameters ======================
n_epoch     = PARAMS.n_epoch
n_epochfolder     = PARAMS.n_epochfolder

n_critic    = PARAMS.n_critic
z_dim       = PARAMS.z_dim
batch_size  = PARAMS.batch_size
n_train     = PARAMS.n_train
log_iter    = np.int(np.floor(n_train/batch_size))
gp_coef     = PARAMS.gp_coef
reg_param   = PARAMS.reg_param
save_suffix = PARAMS.save_suffix
datadir     = PARAMS.datadir
n_z_stat    = PARAMS.MC_samples
n_out       = PARAMS.sample_plots
act_param=PARAMS.act_param
Etype=PARAMS.Etype
ADDtanh = PARAMS.ADDtanh

checkpoint_retrain=PARAMS.checkpoint_retrain

print('\n --- Creating network folder \n')
savedir = f"exps/Nsamples{n_train}_Ncritic{n_critic}_Etype{Etype}_ADDtanh{ADDtanh}_Zdim{z_dim}_BS{batch_size}_seed{PARAMS.seed_no}_Nepoch{n_epochfolder}_GP{gp_coef}{save_suffix}"

if not os.path.exists(savedir):
    os.makedirs(savedir)    
else:
    print('\n     *** Folder already exists!\n')    

print('\n --- Saving parameters to file \n')
param_file = savedir + '/parameters.txt'
with open(param_file,"w") as fid:
    for pname in vars(PARAMS):
        fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")


print('\n --- Loading data from files\n')
# IMPORTANT NOTES ABOUT DATA:
# We assume
# 1. The image width and height is of the form 2^n
# 2. The number of chanels in X and Y are >=1
# 3. The datasets are of the shape N x H x W x 2*C, with each sample stacked as [X,Y]

#train_data, valid_data, N_valid, x_W, x_H, x_C = read_images_hdf5(img_dir=datadir,
 #                                                                  N_train=n_train,
  #                                                                 batch_size=batch_size,
   #                                                                drop_remainder=True
    #                                                              )
 
train_data, valid_data, N_valid, x_W, x_H, x_C = read_images_mat(img_dir=datadir,
                                                                  N_train=n_train,
                                                                 batch_size=batch_size,
                                                                drop_remainder=True
                                                              )      
 
    
print('\n --- Creating conditional GAN models\n')

G_model = generator(Input_W=x_W,Input_H=x_H,Input_C=x_C,Z_Dim=z_dim,act_param=act_param,reg_param=reg_param,out_channels=16,layers=3) 
D_model = critic(Input_W=x_W,Input_H=x_H,Input_C=x_C,act_param=act_param,reg_param=reg_param,out_channels=16,layers=2)

G_optim = tf.keras.optimizers.Adam(0.001,beta_1=0.5, beta_2=0.9)
D_optim = tf.keras.optimizers.Adam(0.001,beta_1=0.5, beta_2=0.9)



ones   = tf.ones([batch_size, 1])
zeros  = tf.zeros([batch_size, 1])

glv = partial(get_lat_var,x_W=x_W,x_H=x_H,z_dim=z_dim)  #get latent variables

stat_z   = glv(batch_size=n_z_stat*n_out)
valid_x = valid_data[0:n_out][:,:,:,0:x_C]
valid_y = tf.constant(valid_data[0:n_out])[:,:,:,x_C::]
stat_y  = tf.repeat(valid_y,n_z_stat,axis=0)
stat_x  = tf.repeat(tf.constant(valid_x),n_z_stat,axis=0)

plt.imshow(valid_x[0,:,:,0],aspect='equal',vmin=-1, vmax=1, cmap='pink')
plt.savefig("test.png")
# ============ Training ==================
print('\n --- Staring training \n')
n_iters = 1
G_loss_log    = []
D_loss_log    = []
wd_loss_log   = []
l1_met_log    = []


#checkpoint_dir = f'{savedir}/checkpoints'
#checkpoint_prefix = os.path.join(checkpoint_dir, "chkpt")

ckpt_id   = tf.Variable(0)

ckpt = tf.train.Checkpoint(ckpt_id=ckpt_id,G_optim=G_optim, D_optim=D_optim, G_model=G_model, D_model=D_model)
  
manager = tf.train.CheckpointManager(ckpt, f'{savedir}/tf_ckpts', max_to_keep=None,step_counter=tf.Variable(0))

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
   # g_loss_hist   = np.loadtxt(f'{savedir}/g_loss.txt')
    start_ = int(ckpt_id.numpy())
    #g_loss_hist=np.concatenate((g_loss_hist,np.zeros(n_epoch-start_)))
   # g_loss_hist=g_loss_hist.tolist()

    #d_loss_hist   = np.loadtxt(f'{savedir}/d_loss.txt')
    start_ = int(ckpt_id.numpy())
    #d_loss_hist=np.concatenate((d_loss_hist,np.zeros(n_epoch-start_)))
   # d_loss_hist=d_loss_hist.tolist()    

    #wd_loss_hist   = np.loadtxt(f'{savedir}/wd_loss.txt')
    start_ = int(ckpt_id.numpy())
    #wd_loss_hist=np.concatenate((wd_loss_hist,np.zeros(n_epoch-start_)))
    #wd_loss_hist=wd_loss_hist.tolist()

else:
    print("Initializing from scratch.")
    #g_loss_hist = np.zeros(n_epoch)
    #d_loss_hist = np.zeros(n_epoch)
   # wd_loss_hist = np.zeros(n_epoch)
    g_loss_hist=[]
    d_loss_hist=[]
    wd_loss_hist=[]

    start_ = 0

manager.step_counter = ckpt_id    

#G_loss_log    = g_loss_hist
#D_loss_log    = d_loss_hist
#wd_loss_log   = wd_loss_hist

#z = glv(batch_size=batch_size)


#test_z = glv(batch_size=n_out)

#pred_ = G_model([stat_y,stat_z],training=None) 
#print(pred_)
for i in range(start_,n_epoch):

    

    for true in train_data:

        true_X = true[:,:,:,0:x_C]  
        true_Y = true[:,:,:,x_C::]  

        z = glv(batch_size=batch_size)

        with tf.GradientTape() as tape:
            fake_X      = G_model([true_Y,z],training=True)
            fake        = tf.squeeze(tf.concat([fake_X,true_Y],axis=3))
            fake_XY_val = D_model(fake,training=True)
            true_XY_val = D_model(true,training=True)

            gp = gradient_penalty(fake_X, true_X, true_Y,D_model, p=2)

            fake_loss = tf.reduce_mean(fake_XY_val)
            true_loss = tf.reduce_mean(true_XY_val) 
            wd_loss = true_loss - fake_loss                
            D_loss = -wd_loss + gp_coef*gp    

            # Adding model regularization terms
            D_loss += sum(D_model.losses)           

            

        D_gradient = tape.gradient(D_loss, D_model.trainable_variables)
        D_optim.apply_gradients(zip(D_gradient, D_model.trainable_variables))


        print(f"     *** iter:{n_iters} ---> d_loss:{D_loss.numpy():.4e}, gp_term:{gp.numpy():.4e}, wd:{wd_loss.numpy():.4e}")
        
        # Appending losses
        #D_loss_log=np.append(D_loss_log,D_loss.numpy())
        #wd_loss_log=np.append(wd_loss_log,wd_loss.numpy())
       #### D_loss_log.append(D_loss.numpy())
       #### wd_loss_log.append(wd_loss.numpy())


        del tape

        if (n_iters) % (n_critic) == 0:

            with tf.GradientTape() as tape:
                fake_X      = G_model([true_Y,z],training=True)
                fake        = tf.squeeze(tf.concat([fake_X,true_Y],axis=3))
                fake_XY_val = D_model(fake,training=True)

                G_loss = -tf.reduce_mean(fake_XY_val)

                # Adding model regularization terms
                G_loss += sum(G_model.losses)

            gen_gradient = tape.gradient(G_loss, G_model.trainable_variables)
            G_optim.apply_gradients(zip(gen_gradient, G_model.trainable_variables))


            print(f"     ***           ---> g_loss:{G_loss.numpy():.4e}") 
            
            # Appending losses
            #G_loss_log=np.append(G_loss_log,G_loss.numpy())
           #### G_loss_log.append(G_loss.numpy())
            
            del tape

        n_iters += 1    

#create a loop, with three columns: true Xs, mean of generated xs and std and 4 rows for each phase. 
    if (i==0) or ((i+1) % PARAMS.evalmet_freq == 0):
        
        pred_ = G_model([stat_y,stat_z],training=None)  #nout*nzstat*128*128    check tf.repeat and what it does

       # l1_err = tf.norm(pred_-stat_x,ord='fro',axis=[1,2])# ---> Check size

       # l1_met_log.append(tf.reduce_mean(l1_err).numpy())



   # if (i==0) or
    if  ((i+1) % PARAMS.savefig_freq == 0):
        print("     *** Saving plots and network checkpoint") 

        pred_ = G_model([stat_y,stat_z],training=None).numpy() 

        for t in range(n_out):   ####  TWO I?NDENTS  fix it
            ncol = 3
            fig1,axs1 = plt.subplots(4, ncol, dpi=100, figsize=(ncol*5,n_out*5))  # pull inside
            ax1 = axs1.flatten()
            ax_ind = 0
            for phase in range(4):
            #axs = ax1[ax_ind]
            #pcm = axs.imshow(valid_y[t].numpy(),aspect='equal')
            #if t==0:
            #    axs.set_title(f'measurement',fontsize=30)
            #axs.axis('off')
            #ax_ind +=1

                axs = ax1[ax_ind]
                pcm = axs.imshow(valid_x[t,:,:,phase],aspect='equal',vmin=-1, vmax=1, cmap='pink')
                #if t==0:
                axs.set_title(f'True Image',fontsize=30)
                divider = make_axes_locatable(axs)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig1.colorbar(pcm,ax=axs,cax=cax)
                axs.axis('off')
                ax_ind +=1

                sample_mean = tf.math.reduce_mean(pred_[t*n_z_stat:(t+1)*n_z_stat],axis=0).numpy()
                sample_std = tf.math.reduce_std(pred_[t*n_z_stat:(t+1)*n_z_stat],axis=0).numpy()
            
                axs = ax1[ax_ind]
                pcm = axs.imshow(sample_mean[:,:,phase],aspect='equal',vmin=-1, vmax=1, cmap='pink')
                #if t==0:
                axs.set_title(f'mean',fontsize=30)
                divider = make_axes_locatable(axs)
                cax = divider.append_axes("right", size="5%", pad=0.05)                
                fig1.colorbar(pcm,ax=axs,cax=cax)
                axs.axis('off')
                ax_ind +=1

                axs = ax1[ax_ind]
                pcm = axs.imshow(sample_std[:,:,phase],aspect='equal',vmin=0, vmax=1, cmap='pink')
                #if t==0:
                axs.set_title(f'std',fontsize=30)
                divider = make_axes_locatable(axs)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig1.colorbar(pcm,ax=axs,cax=cax)
                axs.axis('off')
                ax_ind +=1

           # fig1.tight_layout()
            fig1.savefig(f"{savedir}/sample_stats_{i+1}_{t}.png")    #pull inside  i shifted the spacing two tabs
            plt.close('all')

        ckpt_id.assign_add(PARAMS.savefig_freq)
        save_path = manager.save(checkpoint_number=i+1)
        print(f"Saved checkpoint for epoch {i+1}: {save_path}")
        G_model.save_weights(f'{savedir}/checkpoints/G_checkpoint_{i+1}')

#save_loss(G_loss_log,'g_loss',savedir,n_epoch)
#save_loss(D_loss_log,'d_loss',savedir,n_epoch)  
#save_loss(wd_loss_log,'wd_loss',savedir,n_epoch) 
#save_loss(l1_met_log,'l1_met_val',savedir,n_epoch)



print('\n ============== DONE =================\n')



