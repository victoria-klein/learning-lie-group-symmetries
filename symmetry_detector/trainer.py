import PIL

## Standard libraries
import os
import numpy as np
### JAX
import jax
import jax.numpy as jnp
### Flax
from flax import linen as nn
from flax.training import train_state, checkpoints
### Optax
import optax

from flax.core.frozen_dict import freeze, unfreeze
from typing import Sequence
import wandb
import matplotlib.pyplot as plt
from scipy.linalg import expm, sinm, cosm, logm

class MLPEncoder(nn.Module):

    num_hidden : Sequence[int]

    @nn.compact
    def __call__(self, X, Y):

        x = jnp.concatenate([X, Y], axis=-1)

        x = nn.Dense(features=self.num_hidden[0], kernel_init=nn.initializers.glorot_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden[1], kernel_init=nn.initializers.glorot_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden[2], kernel_init=nn.initializers.glorot_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden[3], kernel_init=nn.initializers.glorot_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden[4], kernel_init=nn.initializers.glorot_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden[5], kernel_init=nn.initializers.glorot_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=1, kernel_init=nn.initializers.glorot_normal())(x)

        return x


class CNNEncoder(nn.Module):

    c_hid : int
    latent_dim : int

    @nn.compact
    def __call__(self, X, Y):
        
        X = X.reshape(X.shape[0], 28, 28, 1)
        Y = Y.reshape(Y.shape[0], 28, 28, 1)

        x = jnp.concatenate([X, Y], axis=-1)

        x = nn.Conv(self.c_hid, kernel_size=(4,4), strides=(2,2), padding=(0,0), kernel_init=nn.initializers.glorot_normal())(x)  # 28x28 => 13x13
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(4,4), kernel_init=nn.initializers.glorot_normal())(x) # 13x13 => 13x13
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3,3), strides=(2,2), padding=(0,0), kernel_init=nn.initializers.glorot_normal())(x)  # 13x13 => 6x6
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3,3), kernel_init=nn.initializers.glorot_normal())(x) # 6x6 => 6x6
        x = nn.gelu(x)
        x = nn.Conv(features=4*self.c_hid, kernel_size=(2,2), strides=(2,2), padding=(0,0), kernel_init=nn.initializers.glorot_normal())(x)  # 6x6 => 3x3
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.latent_dim, kernel_init= nn.initializers.glorot_normal())(x) # X.SHAPE = (batch_size, 1) ? (batch_size, )
        return x


def exp_taylor(t,x,P,X,D,n):
    
    G = (P[0,0] + P[0,1] * X[0] + P[0,2] * X[1]) @ D[0] + (P[1,0] + P[1,1] * X[0] + P[1,2] * X[1]) @ D[1]
    
    temp = jnp.eye(784)
    running_fac = 1.
    running_pow = jnp.eye(784)
    for k in range(1,n+1):
      running_fac *= k
      running_pow = running_pow @ (t*G)
      temp += running_pow/running_fac
    return temp @ x

batch_exp_taylor = jax.vmap(exp_taylor, (0, 0, None, None, None, None), 0)

def exp_scipy(t,x,P,X,D):
    G = (P[0,0] + P[0,1] * X[0] + P[0,2] * X[1]) @ D[0] + (P[1,0] + P[1,1] * X[0] + P[1,2] * X[1]) @ D[1] 
    return jax.scipy.linalg.expm(t*G) @ x

batch_exp_scipy = jax.vmap(exp_scipy, (0, 0, None, None, None), 0)

def xydxdy(input_size):

    d = 28 #sqrt(input_size) input_size=784 for MNIST
    coords = jnp.flip(jnp.stack(jnp.meshgrid(jnp.arange(d)-d/2,jnp.arange(d)-d/2)),axis=0)
    x,y = jnp.reshape(coords,(2,-1))
    dx = (x[:,jnp.newaxis]-x) * (y[:,jnp.newaxis]==y) # make sure distance calculated only for the same y
    dy = (y[:,jnp.newaxis]-y) * (x[:,jnp.newaxis]==x) # ... same x

    return jnp.diag(x), jnp.diag(y), dx, dy

def L0(z, input_size): 

    d = 56 #2*sqrt(input_size) input_size=784 for MNIST
    L = jnp.sum(jnp.array([-2*jnp.pi*p/d**2 * jnp.sin(2*jnp.pi*p/d *z) for p in jnp.arange(-d/2+1,d/2)]), axis=0)

    return L

batch_interpolate_sw = jax.vmap(L0, (0, None), 0)

class LGNetworkInterp(nn.Module):

    input_size : int
    order : int
    taylor: bool
    encoder_cnn: bool
    init_std: float

    def setup(self):
       
       if self.encoder_cnn:
         self.encoder = CNNEncoder(32, 1)
       else:
         self.encoder = MLPEncoder([1000, 900, 800, 500, 100, 20])
       
       self.a_dx = self.param('a_dx', nn.initializers.normal(self.init_std), (1,)) # .zeros(1.)
       self.b_dx = self.param('b_dx', nn.initializers.normal(self.init_std), (1,))
       self.c_dx = self.param('c_dx', nn.initializers.normal(self.init_std), (1,))
       self.a_dy = self.param('a_dy', nn.initializers.normal(self.init_std), (1,))
       self.b_dy = self.param('b_dy', nn.initializers.normal(self.init_std), (1,))
       self.c_dy = self.param('c_dy', nn.initializers.normal(self.init_std), (1,))
       
       self.xs, self.ys, self.dx, self.dy = xydxdy(self.input_size)

       self.d_dx = batch_interpolate_sw(self.dx, self.input_size)
       self.d_dy = batch_interpolate_sw(self.dy, self.input_size)

    def __call__(self, input):
       
       x, y, theta_true = input

       theta = self.encoder(x, y)
       #theta = theta[:,0]
       #theta = jnp.reshape(theta_true, (jnp.shape(theta)[0],jnp.shape(theta)[1])) #theta constant

       P = jnp.array([[self.a_dx,self.b_dx,self.c_dx],[self.a_dy,self.b_dy,self.c_dy]]) #jnp.tanh()
       X = jnp.stack([self.xs,self.ys])
       D = jnp.stack([self.d_dx,self.d_dy])
      
       if self.taylor:
          y_hat = batch_exp_taylor(theta, x, P, X, D, self.order)

       else:
          y_hat = batch_exp_scipy(theta, x, P, X, D)

       y_true = y

       return y_hat, y_true, theta, theta_true, x

def create_LGNetworkInterp(input_size, order, taylor, encoder_cnn, init_std):

  return LGNetworkInterp(input_size=input_size, order=order, taylor=taylor, encoder_cnn=encoder_cnn, init_std=init_std)





def mse_loss(model_state, model_t_params, model_a_params, batch, lambda_lasso):

    model_params = {'params': {'a_dx': model_a_params['a_dx'], 'b_dx': model_a_params['b_dx'], 'c_dx': model_a_params['c_dx'],
                               'a_dy': model_a_params['a_dy'], 'b_dy': model_a_params['b_dy'], 'c_dy': model_a_params['c_dy'],
                               'encoder': model_t_params}}
                               
    y_hat, y, theta, theta_true, x = model_state.apply_fn(model_params, batch)

    lasso = jnp.abs(model_params['params']['a_dx'][0]) \
            +  jnp.abs(model_params['params']['b_dx'][0]) \
            +  jnp.abs(model_params['params']['c_dx'][0]) \
            +  jnp.abs(model_params['params']['a_dy'][0]) \
            +  jnp.abs(model_params['params']['b_dy'][0]) \
            +  jnp.abs(model_params['params']['c_dy'][0])

    loss = (y_hat - y)**2

    return loss.mean() + lambda_lasso*lasso, (theta, theta_true, y_hat, y, x)

def mse_loss_single(model_state, model_params, batch, lambda_lasso):
                               
    y_hat, y, theta, theta_true, x = model_state.apply_fn(model_params, batch)

    lasso = jnp.abs(model_params['params']['a_dx'][0]) \
            +  jnp.abs(model_params['params']['b_dx'][0]) \
            +  jnp.abs(model_params['params']['c_dx'][0]) \
            +  jnp.abs(model_params['params']['a_dy'][0]) \
            +  jnp.abs(model_params['params']['b_dy'][0]) \
            +  jnp.abs(model_params['params']['c_dy'][0])

    loss = (y_hat - y)**2

    return loss.mean() + lambda_lasso*lasso, (theta, theta_true, y_hat, y, x) 
class TrainerModule:

    def __init__(self, model_name, # wandb model name.
                                train_loader, # train dataset loader.
                                val_loader, # validation dataset loader (currently not used).
                                CHECKPOINT_PATH, # pathfrom which/to retrieve/save checkpoints.
                                input_size = 784,
                                lr=1e-3, # initial learning rate.
                                transition_steps=5000, # total transition steps for exponential decay.
                                decay_rate=0.001, # decay rate.
                                end_lr=1e-08, #end learning rate.
                                clip=0.01,
                                seed=42,
                                order=12,
                                encoder_cnn=False,
                                init_std=0,
                                lambda_lasso=0,
                                at_steps=[0,0],
                                val=False,
                                wandb_log=False):

        super().__init__()

        self.model_name = model_name
        self.lr = lr
        self.transition_steps=transition_steps
        self.seed = seed
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.CHECKPOINT_PATH=CHECKPOINT_PATH
        self.decay_rate=decay_rate
        self.end_lr=end_lr
        self.clip_at=clip
        self.input_size = input_size
        self.order = order
        self.encoder_cnn = encoder_cnn
        self.init_std = init_std
        self.lambda_lasso = lambda_lasso
        self.t_steps = at_steps[1]
        self.a_steps = at_steps[0]
        self.wandb_log=wandb_log
        self.val = val

        if self.order==0:
            self.taylor=False
        else:
            self.taylor=True
        
        if at_steps==[0,0]:
            self.multi_step=False
        else:
            self.multi_step=True

        ### ### ###

        self.model = create_LGNetworkInterp(input_size=self.input_size, order=self.order, taylor=self.taylor, encoder_cnn=self.encoder_cnn, init_std=self.init_std)

        self.exmp_imgs = next(iter(train_loader))
        
        self.create_functions()
        self.init_model()

    def create_functions(self):
        
        if self.multi_step:

            # Training function
            def train_step(model_state, batch):

                a_s = {} #for logging
                g1_s = {}
                g2_s = {}
                loss_s = {}
                
                for i in range(self.a_steps):

                    a_grad = jax.value_and_grad(mse_loss,
                                            argnums=2,
                                            has_aux=True
                                            )
                    model_a_params = {'a_dx': model_state.params['params']['a_dx'], 'b_dx': model_state.params['params']['b_dx'], 'c_dx':model_state.params['params']['c_dx'],
                                  'a_dy': model_state.params['params']['a_dy'], 'b_dy': model_state.params['params']['b_dy'], 'c_dy': model_state.params['params']['c_dy']}
                    
                    a_s[i] = model_a_params #a's for logging
                    
                    (loss, tees), grads = a_grad(model_state, model_state.params['params']['encoder'], model_a_params, batch, self.lambda_lasso)
                    
                    g1_s[i] = tree_to_vec(grads) #a grads for logging
                    loss_s[i] = loss #losses for logging
                    
                    encoder_dict = jax.tree_map(lambda x: x*0,  model_state.params['params']['encoder']).unfreeze()
                    grads = {'a_dx': grads['a_dx'], 'b_dx': grads['b_dx'], 'c_dx': grads['c_dx'], 'a_dy': grads['a_dy'], 'b_dy': grads['b_dy'], 'c_dy': grads['c_dy'], 'encoder': encoder_dict}
                    model_state = model_state.apply_gradients(grads=freeze({'params': grads}))

                for i in range(self.t_steps):

                    t_grad = jax.value_and_grad(mse_loss, 
                                            argnums=1,
                                            has_aux=True
                                            )
                    model_a_params = {'a_dx': model_state.params['params']['a_dx'], 'b_dx': model_state.params['params']['b_dx'], 'c_dx':model_state.params['params']['c_dx'],
                                  'a_dy': model_state.params['params']['a_dy'], 'b_dy': model_state.params['params']['b_dy'], 'c_dy': model_state.params['params']['c_dy']}
                    
                    a_s[self.a_steps+i] = model_a_params #a's for logging
                    
                    (loss, tees), grads = t_grad(model_state, model_state.params['params']['encoder'], model_a_params, batch, self.lambda_lasso)
                    
                    g2_s[i] = tree_to_vec(grads) #t grads
                    loss_s[self.a_steps+i] = loss #losses for logging
                    
                    a_dict = jax.tree_map(lambda x: x*0,  model_a_params)
                    grads = {'a_dx': a_dict['a_dx'], 'b_dx': a_dict['b_dx'], 'c_dx': a_dict['c_dx'], 'a_dy': a_dict['a_dy'], 'b_dy': a_dict['b_dy'], 'c_dy': a_dict['c_dy'], 'encoder': grads}
                    model_state = model_state.apply_gradients(grads=freeze({'params': grads}))
                
                return model_state, loss_s, tees, (g1_s,g2_s), a_s

        else:
            # Training function
            def train_step(model_state, batch):

                grad = jax.value_and_grad(mse_loss_single,
                                          argnums=1,
                                          has_aux=True
                                          )
                (loss, tees), grads = grad(model_state, model_state.params, batch, self.lambda_lasso)

                model_state = model_state.apply_gradients(grads)

                return model_state, loss, tees, grads


        self.train_step = jax.jit(train_step) 


        # Eval function
        def eval_step(model_state, batch):
            
            (loss, tees) = mse_loss_single(model_state, model_state.params, batch, self.lambda_lasso)

            return loss, tees

        self.eval_step = jax.jit(eval_step)


    def init_model(self):

        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, inp_model = jax.random.split(rng, 2)

        model_params = self.model.init(inp_model, self.exmp_imgs)  # ['params']

        # Optimizer

        optimizer = optax.adam(learning_rate=self.lr)

        # GRADIENT CLIPPING + ADAM
        
        # optimizer = optax.chain(
        #     optax.clip(self.clip_at),
        #     optax.adam(learning_rate=self.lr))

        # EXPONENTIAL DECAY LEARNING RATE

#         init_learning_rate = self.lr # initial learning rate for Adam
#         self.exponential_decay_scheduler = optax.exponential_decay(init_value=init_learning_rate, transition_steps=self.transition_steps,
#                                                             decay_rate=self.decay_rate, transition_begin=50, end_value=self.end_lr,
#                                                             staircase=False)

#         optimizer = optax.chain(
#             optax.clip(self.clip_at),
#             optax.adam(learning_rate=self.exponential_decay_scheduler))

        # Initialize training state
        self.model_state = train_state.TrainState.create(apply_fn=self.model.apply, params=model_params, tx=optimizer)

    def train_model(self, num_epochs=50):

        for epoch_idx in range(1, num_epochs+1):

            if self.val:
                (loss, val_loss), coefficients = self.train_epoch()
                print("epoch: ", epoch_idx, ", loss : ", loss, coefficients, "validation loss : ", val_loss)
            else:
                loss, coefficients = self.train_epoch()
                print("epoch: ", epoch_idx, ", loss : ", loss, coefficients)
            self.save_model(step=epoch_idx)

    def train_epoch(self):

        plot_count = 0

        for batch in self.train_loader:

            if self.multi_step:
                self.model_state, loss_s, tees, g_s, a_s = self.train_step(self.model_state, batch)
                (g1_s,g2_s) = g_s
            else:
                self.model_state, loss, tees, grads = self.train_step(self.model_state, batch)

            # Log plots with WandB
            
            if self.wandb_log:

                if plot_count%100==0:
                    
                    t_hat , t, y_hat, y, x = tees
                    tf_hat = t_hat.flatten()
                    tf_true = t.flatten()

                    #Histogram of theta_hat
                    plt.hist(tf_hat, bins=10, alpha=0.5, label="t_hat", density=True)  # density=False would make counts
                    plt.legend(loc='upper right')
                    plt.ylabel('Probability')
                    plt.xlabel('Theta')

                    plt.savefig("./"+self.model_name)
                    img_t_hat = PIL.Image.open(self.model_name+".png")
                    os.remove(self.model_name+".png")
                    
                    plt.close()
                    
                    #Histogram of theta_true

                    plt.hist(tf_true, bins=10, alpha=0.5, label="t_true", density=True)
                    plt.legend(loc='upper right')
                    plt.ylabel('Probability')
                    plt.xlabel('Theta')

                    plt.savefig("./"+self.model_name)
                    img_t_true = PIL.Image.open(self.model_name+".png")
                    os.remove(self.model_name+".png")

                    plt.close()

                    #Imshow of y_true
                    i = 24
                    plt.imshow(jnp.reshape(y[i],(28,28)), cmap='gray', vmin=0, vmax=1)
                    plt.colorbar()
                    plt.axis('off')

                    plt.savefig("./"+self.model_name)
                    img_y_true = PIL.Image.open(self.model_name+".png")
                    os.remove(self.model_name+".png")

                    plt.close()

                    #Imshow of y_hat
                    plt.imshow(jnp.reshape(y_hat[i],(28,28)), cmap='gray', vmin=0, vmax=1)
                    plt.colorbar()
                    plt.axis('off')

                    plt.savefig("./"+self.model_name)
                    img_y_hat = PIL.Image.open(self.model_name+".png")
                    os.remove(self.model_name+".png")

                    plt.close()

                    #Imshow of x
                    plt.imshow(jnp.reshape(x[i],(28,28)), cmap='gray', vmin=0, vmax=1)
                    plt.colorbar()
                    plt.axis('off')

                    plt.savefig("./"+self.model_name)
                    img_x = PIL.Image.open(self.model_name+".png")
                    os.remove(self.model_name+".png")
                    
                    plt.close()

                    wandb.log({"Training":{ "Theta": {"Histograms": {"t_hat": wandb.Image(img_t_hat), "t_true": wandb.Image(img_t_true)},
                                          "Stats": {"t_hat_mean": jnp.mean(tf_hat), "t_hat_std": jnp.std(tf_hat), "t_true_mean": jnp.mean(tf_true), "t_true_std": jnp.std(tf_true)}
                                          },
                                          "Images": {"y_true": wandb.Image(img_y_true),"y_hat": wandb.Image(img_y_hat),"x": wandb.Image(img_x)}
                              }}, commit=False)

                if self.multi_step:
                    for i in range(self.a_steps):
                        wandb.log({"Training":{"Coefficients": {"a_dx": a_s[i]['a_dx'],"b_dx": a_s[i]['b_dx'],"c_dx": a_s[i]['c_dx'],"a_dy": a_s[i]['a_dy'],"b_dy": a_s[i]['b_dy'],"c_dy": a_s[i]['c_dy']},
                                              "Loss and gradients": {"Loss": loss_s[i],"a gradients mean squared": (g1_s[i]**2).mean()}
                                  }})
                    for i in range(self.t_steps):
                        wandb.log({"Training":{"Coefficients": {"a_dx": a_s[self.a_steps+i]['a_dx'],"b_dx": a_s[self.a_steps+i]['b_dx'], "c_dx": a_s[self.a_steps+i]['c_dx'],"a_dy": a_s[self.a_steps+i]['a_dy'],"b_dy": a_s[self.a_steps+i]['b_dy'],"c_dy": a_s[self.a_steps+i]['c_dy']},
                                              "Loss and gradients": {"Loss": loss_s[self.a_steps+i],"t gradients mean squared": (g2_s[i]**2).mean()}
                                  }})
                else:
                    wandb.log({"Training":{"Coefficients": {"a_dx": self.model_state.params['params']['a_dx'],
                                                          "b_dx": self.model_state.params['params']['b_dx'],
                                                          "c_dx": self.model_state.params['params']['c_dx'],
                                                          "a_dy": self.model_state.params['params']['a_dy'],
                                                          "b_dy": self.model_state.params['params']['b_dy'],
                                                          "c_dy": self.model_state.params['params']['c_dy']},
                                          "Loss and gradients": {"loss": loss,"gradients mean squared": (grads**2).mean()}
                              }})
     
            plot_count+=1

        if self.val:

            for batch in self.val_loader:

                val_loss, val_tees = self.eval_step(self.model_state, batch)

            # Log plots at end of validation with WandB
            
            if self.wandb_log:

                t_hat , t, y_hat, y, x = val_tees
                tf_hat = t_hat.flatten()
                tf_true = t.flatten()

                #Histogram of theta_hat
                plt.hist(tf_hat, bins=10, alpha=0.5, label="t_hat", density=True)  # density=False would make counts
                plt.legend(loc='upper right')
                plt.ylabel('Probability')
                plt.xlabel('Theta')

                plt.savefig("./"+self.model_name)
                img_t_hat = PIL.Image.open(self.model_name+".png")
                os.remove(self.model_name+".png")
                
                plt.close()
                
                #Histogram of theta_true

                plt.hist(tf_true, bins=10, alpha=0.5, label="t_true", density=True)
                plt.legend(loc='upper right')
                plt.ylabel('Probability')
                plt.xlabel('Theta')

                plt.savefig("./"+self.model_name)
                img_t_true = PIL.Image.open(self.model_name+".png")
                os.remove(self.model_name+".png")

                plt.close()

                #Imshow of y_true
                i = 24
                plt.imshow(jnp.reshape(y[i],(28,28)), cmap='gray', vmin=0, vmax=1)
                plt.colorbar()
                plt.axis('off')

                plt.savefig("./"+self.model_name)
                img_y_true = PIL.Image.open(self.model_name+".png")
                os.remove(self.model_name+".png")

                plt.close()

                #Imshow of y_hat
                plt.imshow(jnp.reshape(y_hat[i],(28,28)), cmap='gray', vmin=0, vmax=1)
                plt.colorbar()
                plt.axis('off')

                plt.savefig("./"+self.model_name)
                img_y_hat = PIL.Image.open(self.model_name+".png")
                os.remove(self.model_name+".png")

                plt.close()

                #Imshow of x
                plt.imshow(jnp.reshape(x[i],(28,28)), cmap='gray', vmin=0, vmax=1)
                plt.colorbar()
                plt.axis('off')

                plt.savefig("./"+self.model_name)
                img_x = PIL.Image.open(self.model_name+".png")
                os.remove(self.model_name+".png")
                
                plt.close()

                wandb.log({"Validation":{"Loss": val_loss,
                                        "Theta": {"Histograms": {"t_hat": wandb.Image(img_t_hat), "t_true": wandb.Image(img_t_true)},
                                                  "Stats": {"t_hat_mean": jnp.mean(tf_hat), "t_hat_std": jnp.std(tf_hat), "t_true_mean": jnp.mean(tf_true), "t_true_std": jnp.std(tf_true)}},
                                        "Images": {"y_true": wandb.Image(img_y_true),"y_hat": wandb.Image(img_y_hat),"x": wandb.Image(img_x)}
                          }})

        if self.multi_step:
            i = self.a_steps+self.t_steps-1
            last_loss = loss_s[i]
            coefficients = {"a_dx": a_s[i]['a_dx'],"b_dx": a_s[i]['b_dx'],"c_dx": a_s[i]['c_dx'],"a_dy": a_s[i]['a_dy'],"b_dy": a_s[i]['b_dy'],"c_dy": a_s[i]['c_dy']}
        else:
            last_loss = loss
            coefficients = {"a_dx": self.model_state.params['params']['a_dx'],
                            "b_dx": self.model_state.params['params']['b_dx'],
                            "c_dx": self.model_state.params['params']['c_dx'],
                            "a_dy": self.model_state.params['params']['a_dy'],
                            "b_dy": self.model_state.params['params']['b_dy'],
                            "c_dy": self.model_state.params['params']['c_dy']}

        if self.val:
            last_loss = (last_loss, val_loss)

        return last_loss, coefficients

    def save_model(self, step):

        # Save current model at certain training iteration
   
        checkpoints.save_checkpoint(ckpt_dir=self.CHECKPOINT_PATH + '/checkpoints/',  # Folder to save checkpoint in
                            target=self.model_state,  # What to save. To only save parameters, use model_state.params
                            step=step,  # Training step or other metric to save best model on
                            prefix='model',  # Checkpoint file name prefix
                            keep=200,
                            overwrite=False   # Overwrite existing checkpoint files
                           )

    def load_model(self, step):

        model_state = checkpoints.restore_checkpoint(
                                        ckpt_dir=self.CHECKPOINT_PATH +'/checkpoints/',   # Folder with the checkpoints
                                        target=self.model_state,   # (optional) matching object to rebuild state in
                                        prefix='model',
                                        step=step
                                        )
        
        trained_model = self.model.bind(model_state.params)
     
        return trained_model
    
    def get_model(self):

        return self.model


def tree_to_vec(params):

    '''
    input: jax.pytree of model parameters.
    output: jax.numpy array of parameters.
    '''

    parameters_list = jax.tree_util.tree_leaves(params)
    parameters = jnp.concatenate([
        param.flatten() for param in parameters_list
    ])

    return parameters
