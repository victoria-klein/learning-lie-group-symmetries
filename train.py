
import hydra
from omegaconf import OmegaConf

import torchvision
import torchvision.transforms as transforms
import cv2

## Standard libraries
import os
import shutil
import numpy as np

import symmetry_detector
from symmetry_detector.trainer import TrainerModule
from symmetry_detector.utils import numpy_collate

import datasets
from datasets.datasets import SymNIST
## JAX

import jax
import jax.numpy as jnp

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

## PyTorch
import torch
import torch.utils.data as data

import wandb


import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg):

        # We possibly want to add fields to the config file. Thus, we set struct to False.
        OmegaConf.set_struct(cfg, False)

        newlr = str(cfg.lr).replace(".", "_")
        newinitstd = str(cfg.init_std).replace(".", "_")
        func_name = 'init_std_'+newinitstd+'_lambda_lasso_'+str(cfg.lambda_lasso)+'_lr_'+newlr

        if cfg.encoder_cnn:
              func_name += '_encoder_cnn'
        else:
              func_name += '_encoder_mlp'

        if cfg.at_steps==[0,0]:
                func_name += '_single_step'
        else:
                func_name += '_[a,t]_'+str(cfg.at_steps)

        if cfg.order==0:
                func_name += '_scipy'
        else:
                func_name += '_taylor_order_'+str(cfg.order)
        
        newstd = str(cfg.data_dict.std).replace(".", "_")
        
        if cfg.data_dict.type=="normal":
               trans_name = 'normal_mean_'+str(cfg.data_dict.type)+'_std_'+str(newstd)

        elif cfg.data_dict.type=="uniform":
                trans_name = 'uniform_size_'+str(cfg.data_dict.type)+'_type_'+str(newstd)

        if cfg.data_dict.r>0:
                model_name = 'Rotation'+'____________'+trans_name+'____________'+func_name
        elif cfg.data_dict.s>0:
               model_name = 'Shift'+'____________'+trans_name+'____________'+func_name
        

        project_name = 'TopG'

        CHECKPOINT_PATH = cfg.checkpoint.root + model_name

        if cfg.checkpoint.overwrite_cp and os.path.exists(CHECKPOINT_PATH):
                shutil.rmtree(CHECKPOINT_PATH)

        if cfg.wandb_log:
                run = wandb.init(project=project_name, entity="m_l_d_s", name=model_name, config = {
                        "Training": {"transition_steps": cfg.training.transition_steps,
                                     "decay_rate": cfg.training.decay_rate,
                                     "end_lr": cfg.training.end_lr,
                                     "clip": cfg.training.clip,
                                     "num_epochs": cfg.training.num_epochs,
                                     "val": cfg.training.val,
                                     "batch_size": cfg.training.batch_size,},
                        "Global": {"input_size": cfg.input_size},
                                   "seed": cfg.seed,
                        "Params": {"lr": cfg.lr,
                                   "order": cfg.order,
                                   "encoder_cnn": cfg.encoder_cnn,
                                   "init_std": cfg.init_std,
                                   "lambda_lasso": cfg.lambda_lasso,
                                   "at_steps": cfg.at_steps},
                        "Checkpoint": {"CHECKPOINT_PATH": CHECKPOINT_PATH},
                        "Dataset": {"blurring_kernel": cfg.data_dict.blurring_kernel,
                                    "rotation": cfg.data_dict.r,
                                    "translation": cfg.data_dict.t,
                                    "scale": cfg.data_dict.s,
                                    "type": cfg.data_dict.type,
                                    "std": cfg.data_dict.std}  
                })
        
        ### ### ###  ###  ###  ###  ### ### ###
        ### ### ### ### DATASET ### ### ### ###
        ### ### ###  ###  ###  ###  ### ### ###

        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        data_dir = './data'

        train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                                train=True, 
                                                transform=trans,
                                                download=True)

        test_dataset = torchvision.datasets.MNIST(root=data_dir,
                                                train=False, 
                                                transform=trans,
                                                download=True)

        x_train, y_train = train_dataset.data, train_dataset.targets # 60000x28x28 and 60000
        x_test, y_test = test_dataset.data, test_dataset.targets # 10000x28x28 and 10000

        x_train, y_train = x_train[:,:,:], y_train[:]
        x_test = x_test[:,:,:]

        f = 28
        points = 60000

        dataset_train = SymNIST(size=points, blurred=cfg.data_dict.blurring_kernel, source=(x_train,x_train), frame=f, rotation=cfg.data_dict.r, translation=cfg.data_dict.t, scale=cfg.data_dict.s, rotation_type=cfg.data_dict.type, rotation_std=cfg.data_dict.std)
        # dataset_test = SymNIST(size=int(points/6), source=(x_test,x_test), frame=f, rotation=r, translation=t, scale=s, rotation_type=data_dict.rotation_type, rotation_std=data_dict.rotation_std)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [int(9*points/10), int(points/10)])

        train_loader = data.DataLoader(dataset_train, batch_size=cfg.training.batch_size, collate_fn=numpy_collate, shuffle=True)
        val_loader = data.DataLoader(dataset_val, batch_size=int(points/10), collate_fn=numpy_collate, shuffle=False)
        # test_loader = data.DataLoader(dataset_test, batch_size=cfg.training.batch_size, collate_fn=hp.numpy_collate, shuffle=False)

        ### ###  ###  ###  ###  ###  ### ### ###
        ### ### ### ### TRAINING ### ### ### ###
        ### ###  ###  ###  ###  ###  ### ### ###

        trainer = TrainerModule(model_name,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                CHECKPOINT_PATH=CHECKPOINT_PATH,
                                input_size=cfg.input_size,
                                lr=cfg.lr,
                                transition_steps=cfg.training.transition_steps,
                                decay_rate=cfg.training.decay_rate,
                                end_lr=cfg.training.end_lr,
                                clip=cfg.training.clip,
                                seed=cfg.seed,
                                order=cfg.order,
                                encoder_cnn=cfg.encoder_cnn,
                                init_std=cfg.init_std,
                                lambda_lasso=cfg.lambda_lasso,
                                at_steps=cfg.at_steps,
                                val=cfg.training.val,
                                wandb_log=cfg.wandb_log)

        trainer.train_model(num_epochs=cfg.training.num_epochs)


if __name__ == "__main__":
    main()
