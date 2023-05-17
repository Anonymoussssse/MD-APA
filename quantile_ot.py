from re import M
import torch
from torchvision import models,  datasets, transforms
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import numpy as np
import json

import os
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision.utils import save_image

import math

from loss import Smoothness_loss, OT_Loss_quantile
from dataset import *
from models import Ensemble
from patch_utils import PatchTransformer


def train_quantile_ot(config):

    if not os.path.exists(config["noise_result_dir"]):
        os.makedirs(config["noise_result_dir"])

    with open(os.path.join(config["noise_result_dir"], 'config.json'), 'w') as fp:
        json.dump(config, fp,  indent=4)
    print(config["noise_result_dir"],)

    os.environ['CUDA_VISIBLE_DEVICES'] = config["device_number"]
    noise_result_dir = config["noise_result_dir"]

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("seed : ", torch.cuda.initial_seed())

    donnees = "imagenet" # "imagenet" "pascal"
    ensemble = True

    all_models = config["all_models"]

    n_models = len(all_models)
    batch_size = config["batch_size"]
    num_iter = config["num_iter"]
    target_classe = config["target_classe"] 
    print("target class", target_classe)

    print(all_models)


    m = Ensemble()
    m.append_models(all_models)


    m.cuda()
    m.eval()

    patch_transfor = PatchTransformer(config)

    transform = {}


    # if n_models >1 and ensemble:
    #     transform = process_image(all_models)
    #     ensemble_train = True
    # else :
    #     transform[all_models[0]] = transforms.Lambda(lambda x : x)
    #     ensemble_train = False
    for mm in range(len(all_models)):
        transform[all_models[mm]] = transforms.Lambda(lambda x : x)
    ensemble_train = False



    print("Attack on Imagenet")

    dataset_imagenet = get_dataset("imagenet", "test")
    dataset_imagenet = get_loader(all_models, dataset_imagenet, ensemble_train)

    dataset_imagenet = dataset_imagenet[all_models[0]]

    train_set, test_set = torch.utils.data.random_split(dataset_imagenet, [40000, 10000], generator=torch.Generator().manual_seed(seed))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    normalise = normalisation(all_models)


    patchSize = config["patchSize"] 
    # start_x = 5
    # start_y = 5
    start_x = np.array([5]) # 5 86 73 61 48 36 patch 50X50 75X75 100X100
    start_y = np.array([5])
    patch_coord = [start_y, start_x, patchSize]


    max_epochs = config["max_epochs"] 

    noise = torch.zeros(1, 3, patchSize, patchSize)


    noise = noise.cuda().requires_grad_(True)

    weight_decay = config["weight_decay"]  #0.0005
    momentum = config["momentum"] 
    lr = config["lr"] 
    print("learning rate = ", lr)

    # optimizer = torch.optim.Adam([noise], lr=lr)

    optimizer = torch.optim.SGD([noise], lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)


    ot_loss = OT_Loss_quantile(all_models, target_classe, config["onedim"])


    n_feature_attacked = 1

    Sc = torch.zeros(1000)

    train_loss_list = torch.zeros((n_models,max_epochs))
    l2_loss_list = torch.zeros((n_models,max_epochs))
    l1_loss_list = torch.zeros((n_models,max_epochs))


    features = []
    def hook_feature(module, input, output):
        features.append(output)

    H = []
    for model in all_models:

        if model in ["resnet50", "resnet50_v2", "resnet50_self", "resnet34", "resnet18", "resnet50_relu_adv"]:
            # H.append(m.models[model].layer1.register_forward_hook(hook_feature))
            # H.append(m.models[model].layer2.register_forward_hook(hook_feature))
            # H.append(m.models[model].layer3.register_forward_hook(hook_feature))
            H.append(m.models[model].layer4.register_forward_hook(hook_feature))

        elif model in ["efficientnet_b4","densenet161","vgg19","densenet121", "densenet169", "densenet201",
                        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "convnext_small", "convnext_tiny"
                        ]:
            H.append(m.models[model].features.register_forward_hook(hook_feature))

        elif model in ["swin_t", "swin_s", "swin_b"]:
            H.append(m.models[model].avgpool.register_forward_hook(hook_feature))
        
        elif model in ["deit_s", "deit_s_adv", "deit_t", "deit_b"]:
            # H.append(m.models[model].blocks.register_forward_hook(hook_feature))
            H.append(m.models[model].pre_logits.register_forward_hook(hook_feature))

        elif model == "inceptionv3":
            H.append(m.models[model].Mixed_7c.register_forward_hook(hook_feature))
    print(H)
        

    for epoch in range(0, max_epochs):
        print('Epoch {}'.format(epoch))

        train_loss_sum = torch.zeros(n_models)
        l2_loss_sum = torch.zeros(n_models)
        l1_loss_sum = torch.zeros(n_models)



        for batch_idx, x in enumerate(train_loader):
            print("Trying to fool:" + str(batch_idx))
            
            data0, y = x

            data0 = data0.cuda()
            data0 = Variable(data0, requires_grad=False)

            y = y.cpu()

            for nc in np.unique(y):
                pc = np.where(y==nc)[0]
                Sc[nc] = Sc[nc] + pc.shape[0]

            features = []

            _ = [m(normalise[model](transform[model](data0)),model) for model in  all_models]
            f_target = [i.detach() for i in features]



            r = 0


            for i in range(num_iter):
                features = []
                loss_ensemble = torch.zeros(n_models, device="cuda")
                grad_ensemble = torch.zeros((n_models, 1, 3, patchSize, patchSize), device="cuda")
                
                for n_model in range(n_models):
                    model = all_models[n_model]
                    data_cloned = data0.clone()
                    data_cloned = transform[model](data_cloned)

                    if config["use_eot"] == True:
                        data_cloned = patch_transfor(data_cloned, noise)
                    else :      
                        data_cloned[:, :, start_y[r]:start_y[r] + patchSize, start_x[r]:start_x[r] + patchSize] = noise


                    data_cloned = torch.clamp(data_cloned, 0, 1)
                    
                    data_cloned = normalise[model](data_cloned)
                    _ = m(data_cloned, model) # forward patched image

                    loss, l2, l1 = ot_loss(features[n_model], model)

                    loss_ensemble[n_model] = loss

                    train_loss_sum[n_model] += loss.detach().cpu()
                    l2_loss_sum[n_model] += l2.detach().cpu()
                    l1_loss_sum[n_model] += l1.detach().cpu()



                optimizer.zero_grad()

                smooth_loss = Smoothness_loss(noise)  

                def norm(v):
                    n = torch.norm(v, p=float('2'))
                    return (v/n) if n > 0 else v # to avoid NaN error 
                
            
                    
                for count, l in enumerate(loss_ensemble):
                    optimizer.zero_grad()
                    l.backward(retain_graph=True)
                    grad_loss = noise.grad.data.clone()
                    grad_ensemble[count] = norm(grad_loss)
                    # print(grad_ensemble.device)




                optimizer.zero_grad()
                smooth_loss.backward()
                grad_loss = noise.grad.data.clone()
                norm_grad_loss = norm(grad_loss)
            
                final_grad_adv = grad_ensemble.sum(0)/n_models


                noise.grad = final_grad_adv + 0.1 * norm_grad_loss 


                optimizer.step()

                noise.data = torch.clamp(noise.data, 0, 1)


                print("[" + time.asctime(time.localtime(time.time())) + "]" + 'Batch num:%d Ite: %d / %d Loss ensemble: %f Smooth loss : %f noise norm: %f ' \
                                    % (batch_idx, i, num_iter, loss_ensemble.sum().item(), smooth_loss.item(), noise.data.norm()))

            if batch_idx >= 1000//(num_iter*batch_size):
                break

        train_loss_list[:,epoch] = train_loss_sum/1000
        l2_loss_list[:,epoch] = l2_loss_sum/1000
        l1_loss_list[:,epoch] = l1_loss_sum/1000




        if (epoch+1)%25 == 0:
            noise.data = torch.clamp(noise.data, 0, 1)
            np.save(noise_result_dir +'/'+ 'epoch_' + str(epoch+1) + '_universal_patch.npy', noise.data.cpu().squeeze())
            save_image(noise.data.squeeze(), noise_result_dir +'/'+ 'epoch_' + str(epoch+1) + '_universal_patch.png')


    torch.save(Sc, noise_result_dir +'/'+ 'Sc.pt')


    for n_model in range(n_models):

        plt.clf()
        plt.plot(np.arange(1,max_epochs+1),train_loss_list[n_model,:])
        plt.xlabel("epochs")
        plt.savefig(noise_result_dir +'/'+ "train_loss" + all_models[n_model] + str(lr) + ".png")

        plt.clf()
        plt.plot(np.arange(1,max_epochs+1),l2_loss_list[n_model,:])
        plt.xlabel("epochs")
        plt.savefig(noise_result_dir +'/'+ "l2_loss" + all_models[n_model] + str(lr) + ".png")

        plt.clf()
        plt.plot(np.arange(1,max_epochs+1),l1_loss_list[n_model,:])
        plt.xlabel("epochs")
        plt.savefig(noise_result_dir +'/'+ "l1_loss" + all_models[n_model] + str(lr) + ".png")


if __name__ == "__main__":

    whereIam = os.uname()[1]
    print(whereIam)
    
    HOME = whereIam

        
    config = {
        "use_gpu": True,
        "device_number": "1",
        "onedim": True,
        "seed": 100,
        "noise_result_dir" : HOME,
        "all_models": ["swin_t"],
        "batch_size" : 50,
        "num_iter" : 1,
        "target_classe" : 193,
        "max_epochs": 100,
        "weight_decay" : 0,
        "momentum" : 0.9,
        "lr" : 1e-1        
    }

    # 954,banana; 207,golden; 859,toaster; 193,australian_terrier; 980,volcano
    # all_models = ["resnet50", "resnet34", "resnet18", "densenet161", "densenet121", "densenet169", "densenet201",
    #                "convnext_tiny", "convnext_small", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "vgg19", "inceptionv3"]

    config.update({
    'use_eot': True,
    "max_x_rotation": 5,
    "max_y_rotation": 5,
    "max_z_rotation": 10,
    "patchSize" : 110, #50 110
    "min_patch_width": 70, # 30 70
    "max_patch_width": 110, # 70 110
    "min_brightness": -0.1,
    "max_brightness": 0.1,
    "noise_factor": 0.1,
    "min_contrast": 0.8,
    "max_contrast" : 1.2,
    "fixed_patch": None
    })

    if config["use_eot"]==True:
        config["noise_result_dir"] = os.path.join(config["noise_result_dir"], "eot")
    else:
        config["noise_result_dir"] = os.path.join(config["noise_result_dir"], "eot")      


    if len(config["all_models"])==1:
        config["noise_result_dir"] = os.path.join(config["noise_result_dir"], f"single/{config['all_models'][0]}")
    else:
        config["noise_result_dir"] = os.path.join(config["noise_result_dir"], f"multi_{len(config['all_models'])}")      

    config["noise_result_dir"] = os.path.join(config["noise_result_dir"], "all")    


    config["noise_result_dir"] = os.path.join(config["noise_result_dir"], "big_patch")    

    config["noise_result_dir_cache"] = config["noise_result_dir"]

    for c_i in [193]: #193,954,207,859,980
        config["target_classe"] = c_i

        for lr_i in [0.1,0.5,1]:

            if c_i == 193:
                config["noise_result_dir"] = os.path.join(config["noise_result_dir_cache"], f"australian_terrier/{str(lr_i)}")

            elif c_i==954:
                config["noise_result_dir"] = os.path.join(config["noise_result_dir_cache"], f"banana/{str(lr_i)}")

            elif c_i==207:
                config["noise_result_dir"] = os.path.join(config["noise_result_dir_cache"], f"golden/{str(lr_i)}")

            elif c_i==859:
                config["noise_result_dir"] = os.path.join(config["noise_result_dir_cache"], f"toaster/{str(lr_i)}")

            elif c_i==980:
                config["noise_result_dir"] = os.path.join(config["noise_result_dir_cache"], f"volcano/{str(lr_i)}")

            config["lr"] = lr_i
            print("step", c_i, lr_i)
            print(config["noise_result_dir_cache"])
            
            train_quantile_ot(config)

    # config["noise_result_dir"] = os.path.join(config["noise_result_dir"], "L1")    


    # train_quantile_ot(config)


 



