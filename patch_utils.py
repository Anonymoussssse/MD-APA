import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def get_position(patch_position, h_pad, w_pad):
    if patch_position == [0,0]:
        random_x_delta = -w_pad
        random_y_delta = -h_pad

    elif  patch_position == [0,1]:
        random_x_delta = -w_pad
        random_y_delta = 0

    elif  patch_position == [0,2]:
        random_x_delta = -w_pad
        random_y_delta = h_pad

    elif  patch_position == [1,0]:
        random_x_delta = 0
        random_y_delta = -h_pad

    elif  patch_position == [1,1]: 
        random_x_delta = 0
        random_y_delta = 0

    elif  patch_position == [1,2]: 
        random_x_delta = 0
        random_y_delta = h_pad

    elif  patch_position == [2,0]:
        random_x_delta = w_pad
        random_y_delta = -h_pad

    elif  patch_position == [2,1]: 
        random_x_delta = w_pad
        random_y_delta = 0

    elif  patch_position == [2,2]: 
        random_x_delta = w_pad
        random_y_delta = h_pad

    return random_x_delta, random_y_delta 

def get_batched(tensor, batch_size):
    return tensor.repeat((batch_size,) + (1,) * len(tensor.shape))

def apply_patch(image_batch, patch, tform_batch):

    batch_size = image_batch.size(0)

    # patch_batch = get_batched(patch, batch_size)
    mask_batch = torch.ones(image_batch.shape).cuda()

    affine_grid = F.affine_grid(tform_batch, mask_batch.shape).cuda()


    mask_batch_tform = F.grid_sample(mask_batch, affine_grid)
    patch_batch_tform = F.grid_sample(patch, affine_grid) 


    # crop out the letterbox padding to convert square to potentially non-square image shape
    mask_batch_tform = crop(mask_batch_tform, image_batch.shape)
    patch_batch_tform = crop(patch_batch_tform, image_batch.shape)

    input_batch = (1-mask_batch_tform) * image_batch + mask_batch_tform * patch_batch_tform
    return input_batch, patch_batch_tform


def crop(image_batch, shape):
    h, w = shape[-2:]
    shape_max = max(h,w)

    crop_h = int((shape_max - h) / 2)
    crop_w = int((shape_max - w) / 2)

    return image_batch[:,:, crop_h: crop_h + h, crop_w: crop_w + w]

def get_random_tform(image_shape, rotation_range=np.zeros(3), 
    patch_size_range=(30, 70), use_eot=False, fixed_patch=None, patch_position=None):
    
    random_rotation = np.zeros(3)
    if use_eot :
        random_rotation[0] = np.random.uniform(*rotation_range[0]) * np.pi/180.
        random_rotation[1] = np.random.uniform(*rotation_range[1]) * np.pi/180.
        random_rotation[2] = np.random.uniform(*rotation_range[2]) * np.pi/180.


    image_h, image_w = image_shape[-2:] #416x416
    image_max = max(image_h, image_w)
    # sample scaling factor
    random_size = np.random.uniform(*patch_size_range)
    random_scale = random_size / image_max
    # translations should be between -1 and 1; -1,-1 is top left, 1,1 is bottom right
    # note that the total "width", "height" is therefore 2, 2
    h_pad = image_h / image_max - random_scale
    w_pad = image_w / image_max - random_scale

    # sample translation
    if fixed_patch is None:
        random_x_delta = np.random.uniform(-w_pad, w_pad)
        random_y_delta = np.random.uniform(-h_pad, h_pad)
    else:
        # fix to top left
        random_x_delta, random_y_delta = get_position(patch_position,h_pad,w_pad)

    # yaw matrix
    yaw_mat = torch.eye(3)
    yaw_mat[1:,1:] = torch.Tensor([
        [np.cos(random_rotation[0]), -np.sin(random_rotation[0])],
        [np.sin(random_rotation[0]), np.cos(random_rotation[0])]
    ])

    # rotate_range = [-self.max_y_rotation, self.max_y_rotation]
    # random_rotation = np.random.uniform(*rotation_range) * np.pi/180.

    # pitch matrix
    pitch_mat = torch.eye(3)
    pitch_mat[::2,::2] = torch.Tensor([
        [np.cos(random_rotation[1]), -np.sin(random_rotation[1])],
        [np.sin(random_rotation[1]), np.cos(random_rotation[1])]
    ])

    # rotate_range = [-self.max_z_rotation, self.max_z_rotation]
    # random_rotation = np.random.uniform(*rotation_range) * np.pi/180.

    # rotation matrix
    rotate_mat = torch.eye(3)
    rotate_mat[:2,:2] = torch.Tensor([
        [np.cos(random_rotation[2]), np.sin(random_rotation[2])],
        [-np.sin(random_rotation[2]), np.cos(random_rotation[2])]
    ])

    # scale matrix
    scale_mat = torch.eye(3) / random_scale
    scale_mat[2,2] = 1.

    # print( random_x_delta * np.sin(random_rotation) / scale_mat + random_y_delta * np.cos(random_rotation) / scale_mat )

    # translation matrix
    translate_mat = torch.eye(3)
    translate_mat[:2,2] = torch.Tensor([-random_x_delta, -random_y_delta])

    # build transformation matrix
    tform_mat = rotate_mat.mm(pitch_mat.mm(yaw_mat.mm(scale_mat.mm(translate_mat))))
    # keep only the top two rows
    return tform_mat[:2,:]
    
def get_random_tform_batch(image_batch, rotations, max_patch_width, use_eot, fixed_patch, patch_position):
    tform_batch = []
    for img in image_batch:
        tform_batch.append(get_random_tform(img.shape, rotations, max_patch_width, use_eot, fixed_patch, patch_position))
    return torch.stack(tform_batch)


class PatchTransformer(nn.Module):

    def __init__(self, cfg):
        super(PatchTransformer, self).__init__()

        self.use_eot = cfg["use_eot"]

        self.min_contrast = cfg["min_contrast"] # 0.8
        self.max_contrast = cfg["min_contrast"] # 1.2
        self.min_brightness = cfg["min_brightness"] # -0.1
        self.max_brightness = cfg["max_brightness"] # 0.1
        self.noise_factor = cfg["noise_factor"] #0.10
        self.min_patch_width = cfg["min_patch_width"]
        self.max_patch_width = cfg["max_patch_width"]


        # in degrees
        self.max_x_rotation = cfg["max_x_rotation"] 
        self.max_y_rotation = cfg["max_y_rotation"] 
        self.max_z_rotation = cfg["max_z_rotation"] 

        self.rotations = np.array([[-self.max_x_rotation,self.max_x_rotation],
                                  [-self.max_y_rotation,self.max_y_rotation],
                                  [-self.max_z_rotation,self.max_z_rotation]])

        if cfg["fixed_patch"] is not None:
            self.fixed_patch = cfg["fixed_patch"]
            self.patch_position = cfg["patch_position"]
        else : 
            self.fixed_patch = None
            self.patch_position = None   


    def forward(self, image_batch, patch):

            batch_size = image_batch.size(0)
            patch_size = patch.size(2)


            # Contrast, brightness and noise transforms
            # Create random contrast tensor
            contrast = torch.zeros(batch_size, device="cuda").uniform_(self.min_contrast, self.max_contrast)
            contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


            # Create random brightness tensor
            # brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
            brightness = torch.zeros(batch_size, device="cuda").uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)



            # Create random noise tensor
            noise = torch.randn_like(patch, device="cuda") * self.noise_factor

            # Apply contrast/brightness/noise, clamp
            patch = patch * contrast + brightness + noise
            patch = torch.clamp(patch, 0.000001, 0.99999)


            tform_batch = torch.cat([get_random_tform_batch(image_batch, self.rotations, 
                        (self.min_patch_width, self.max_patch_width), self.use_eot, self.fixed_patch, self.patch_position)], dim=0)
            
            input_batch, patch_batch_tform = apply_patch(image_batch, patch, tform_batch)
            input_batch = input_batch.cuda()

            adv_batch_t = torch.clamp(input_batch, 0.000001, 0.999999)


            return adv_batch_t

