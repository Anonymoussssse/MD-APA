from torchvision import models
import torch.nn as nn
import torch
import timm
import os


whereIam = os.uname()[1]

HOME_DIR = whereIam
DATA_DIR = whereIam



class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleDict({})
            
    def append_models(self, list_models):
        for model in list_models:
            if model == "resnet50":
                self.models[model] = models.resnet50(weights='IMAGENET1K_V1')

            if model == "resnet50_v2":
                self.models[model] = models.resnet50(weights='IMAGENET1K_V2')          

            if model == "resnet50_self":  
                self.models[model] = models.resnet50(weights = None)

                state_dict = torch.load(HOME_DIR + "code/swav_800ep_pretrain.pth.tar", map_location=torch.device('cpu'))
                state_dict_fc = torch.load(HOME_DIR + "code/swav_800ep_eval_linear.pth.tar", map_location=torch.device('cpu'))
                state_dict['fc.weight'] = state_dict_fc["state_dict"]['module.linear.weight']
                state_dict['fc.bias'] = state_dict_fc["state_dict"]['module.linear.bias']
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                self.models[model].load_state_dict(state_dict, strict=False)

            if model == "resnet18":
                self.models[model] = models.resnet18(weights='IMAGENET1K_V1')

            if model == "resnet34":
                self.models[model] = models.resnet34(weights='IMAGENET1K_V1')

            if model == "efficientnet_b0":
                self.models[model] = models.efficientnet_b0(weights='IMAGENET1K_V1')

            if model == "efficientnet_b1":
                self.models[model] = models.efficientnet_b1(weights='IMAGENET1K_V1')

            if model == "efficientnet_b2":
                self.models[model] = models.efficientnet_b2(weights='IMAGENET1K_V1')

            if model == "efficientnet_b3":
                self.models[model] = models.efficientnet_b3(weights='IMAGENET1K_V1')

            if model == "efficientnet_b4":
                self.models[model] = models.efficientnet_b4(weights='IMAGENET1K_V1')

            if model == "densenet121":
                self.models[model] = models.densenet121(weights='IMAGENET1K_V1')

            if model == "densenet161":
                self.models[model] = models.densenet161(weights='IMAGENET1K_V1')

            if model == "densenet169":
                self.models[model] = models.densenet169(weights='IMAGENET1K_V1')

            if model == "densenet201":
                self.models[model] = models.densenet161(weights='IMAGENET1K_V1')

            if model == "inceptionv3":
                self.models[model] = models.inception_v3(weights='IMAGENET1K_V1')

            if model == "vgg19":
                self.models[model] = models.vgg19_bn(weights='IMAGENET1K_V1')

            if model == "convnext_small":
                self.models[model] = models.convnext_small(weights='IMAGENET1K_V1')

            if model == "convnext_tiny":
                self.models[model] = models.convnext_tiny(weights='IMAGENET1K_V1')

            if model == "swin_t":
                self.models[model] = models.swin_t(weights='IMAGENET1K_V1')

            if model == "swin_s":
                self.models[model] = models.swin_s(weights='IMAGENET1K_V1')

            if model == "swin_b":
                self.models[model] = models.swin_b(weights='IMAGENET1K_V1')






            if model == "timm_resnet50":
                self.models[model] = timm.create_model('resnet50', pretrained=True)

            if model == "timm_resnet18":
                self.models[model] = timm.create_model('resnet18', pretrained=True)

            if model == "timm_resnet34":
                self.models[model] = timm.create_model('resnet34', pretrained=True)

            if model == "timm_densenet121":
                self.models[model] = timm.create_model('densenet121', pretrained=True)

            if model == "timm_densenet161":
                self.models[model] = timm.create_model('densenet161', pretrained=True)

            if model == "timm_densenet169":
                self.models[model] = timm.create_model('densenet169', pretrained=True)

            if model == "timm_densenet201":
                self.models[model] = timm.create_model('densenet201', pretrained=True)

            if model == "timm_inceptionv3":
                self.models[model] = timm.create_model('inception_v3', pretrained=True)

            if model == "timm_vgg19":
                self.models[model] = timm.create_model('vgg19', pretrained=True)

            if model == "deit_s":
                self.models[model] = timm.create_model('deit_small_patch16_224', pretrained=True)

            if model == "deit_t":
                self.models[model] = timm.create_model('deit_tiny_patch16_224', pretrained=True)

            if model == "deit_b":
                self.models[model] = timm.create_model('deit_base_patch16_224', pretrained=True)




            if model == "resnet50_relu_adv":



                from adv_script.bn import FourBN
                import adv_script.advresnet_gbn as advres


                norm_layer = FourBN 
                self.models[model] = advres.__dict__["resnet50"](norm_layer=norm_layer)
                self.models[model].set_mix(False)
                self.models[model].set_sing(True)
                self.models[model].set_mixup_fn(False)
                state_dict = torch.load(HOME_DIR + "data/IMAGENET/advres50_relu.pth", map_location="cpu")
                self.models[model].load_state_dict(state_dict["model"], strict=False)


            if model == "deit_s_adv":
                self.models[model] = timm.create_model('deit_small_patch16_224', pretrained=True)
                state_dict = torch.load(HOME_DIR + "data/IMAGENET/advdeit_small.pth", map_location=torch.device('cpu'))
                self.models["deit_s_adv"].load_state_dict(state_dict["model"])




    def forward(self, x, model):
        x = self.models[model](x)
        return x
