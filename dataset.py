from torchvision import transforms, datasets, models
import torch
import os
from torch.utils.data import Dataset
import numpy as np

whereIam = os.uname()[1]
print(whereIam)

print("Put the Imagenet directory location bellow")
print("line 11 dataset.py")
os.environ["IMAGENET_DIR"] = "/imagenet"

IMAGENET_LOC_ENV = "IMAGENET_DIR"
# list of all datasets
DATASETS = ["imagenet", "cifar10"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        # transform = transforms.Compose([
        #     transforms.RandomSizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()
        # ])
    elif split == "test":
        #supposed here that the model is resnet
        subdir = os.path.join(dir, "val")
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor()
        # ])

    return subdir #datasets.ImageFolder(subdir, transform)

def get_loader(list_models, subdir, ensemble_train):

    data_loader = {}
    if ensemble_train: 
        if len(list_models)>1:
            n_resize = -np.inf
            n_crop = -np.inf
            for model in list_models:
                if model in ["resnet50", "resnet50_v2", "resnet50_self", "resnet50_relu_adv", "resnet18", "resnet34",
"vgg19", "densenet161", "densenet121", "densenet169",
"densenet201", "convnext_small", "convnext_tiny", "efficientnet_b0", "deit_s", "deit_s_adv", "deit_t", "deit_b"]:
                    n_resize = max(n_resize, 256)
                    n_crop = max(n_crop, 224)

                elif model == "efficientnet_b1":
                    n_resize = max([n_resize, 256])
                    n_crop = max([n_crop, 240])

                elif model == "efficientnet_b2":
                    n_resize = max([n_resize, 288])
                    n_crop = max([n_crop, 288])

                elif model == "efficientnet_b3":
                    n_resize = max([n_resize, 320])
                    n_crop = max([n_crop, 300])

                elif model == "efficientnet_b4":
                    n_resize = max([n_resize, 384])
                    n_crop = max([n_crop, 380])

                elif model == "inceptionv3":
                    n_resize = max([n_resize, 342])
                    n_crop = max([n_crop, 299])

                elif model == "swin_t":
                    n_resize = max([n_resize, 232])
                    n_crop = max([n_crop, 224])

                elif model == "swin_b":
                    n_resize = max([n_resize, 246])
                    n_crop = max([n_crop, 224])
                    
                elif model == "swin_s":
                    n_resize = max([n_resize, 238])
                    n_crop = max([n_crop, 224])

            transform = transforms.Compose([
            transforms.Resize(n_resize),
            transforms.CenterCrop(n_crop),
            transforms.ToTensor()])
            data_loader = datasets.ImageFolder(subdir, transform)

    else:
        for model in list_models:
            if model in ["resnet50", "resnet50_v2", "resnet50_self", "resnet50_relu_adv", "resnet18", "resnet34",
"vgg19", "densenet161", "densenet121", "densenet169",
"densenet201", "convnext_small", "convnext_tiny"]:
                transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()])
                
                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model in ["efficientnet_b0", "deit_s", "deit_s_adv", "deit_t", "deit_b"]:
                transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model == "efficientnet_b1":
                transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(240),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model == "efficientnet_b2":
                transform = transforms.Compose([
                transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(288),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model == "efficientnet_b3":
                transform = transforms.Compose([
                transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(300),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model == "efficientnet_b4":
                transform = transforms.Compose([
                transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(380),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model == "inceptionv3":
                transform = transforms.Compose([
                transforms.Resize(342),
                transforms.CenterCrop(299),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model == "swin_t":
                transform = transforms.Compose([
                transforms.Resize(232, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model == "swin_b":
                transform = transforms.Compose([
                transforms.Resize(246, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

            elif model == "swin_s":
                transform = transforms.Compose([
                transforms.Resize(238, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

                data_loader[model] = datasets.ImageFolder(subdir, transform)

    return data_loader


def process_image(list_models):
    transform = {}
    for model in list_models:
        if model in ["resnet50", "resnet50_v2", "resnet50_self", "resnet50_relu_adv", "resnet18", "resnet34",
"vgg19", "densenet161", "densenet121", "densenet169",
"densenet201", "convnext_small", "convnext_tiny"]:
            transform[model] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)])

        elif model == "efficientnet_b4":
            transform[model] = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(380)])

        elif model == "inceptionv3":
            transform[model] = transforms.Compose([
            transforms.Resize(342),
            transforms.CenterCrop(299)])

        elif model in ["efficientnet_b0", "deit_s", "deit_s_adv", "deit_t", "deit_b"]:
            transform[model] = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224)])

        elif model == "efficientnet_b1":
            transform[model] = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(240)])

        elif model == "efficientnet_b2":
            transform[model] = transforms.Compose([
            transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(288)])

        elif model == "swin_t":
            transform[model] = transforms.Compose([
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224)])

        elif model == "swin_s":
            transform[model] = transforms.Compose([
            transforms.Resize(246, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224)])

        elif model == "swin_b":
            transform[model] = transforms.Compose([
            transforms.Resize(238, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224)])

    return transform


def normalisation(list_models):
    transform = {}
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    for model in list_models:
        if model == "resnet50_relu_adv":
            transform[model] = transforms.Normalize(mean = torch.Tensor([0.5, 0.5, 0.5]),
            std= torch.Tensor([0.5, 0.5, 0.5]))
        else :
            transform[model] = transforms.Normalize(mean = mean, std= std)
    return transform
