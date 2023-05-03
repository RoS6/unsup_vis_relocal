import pathlib
import tempfile

import torch
import torchvision
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import roma
import tqdm

from Unet2 import UNet
from se3 import se3_log_map, se3_exp_map
# from functorch import jacfwd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transform_from_params(params):
    return se3_exp_map(params).mT
def params_from_transform(transform):
    return se3_log_map(transform.mT)
def pose_from_transform(transform):
    return transform[..., :3, :3], transform[..., :3,[3]]

def pose_from_transform(transform):
    return transform[..., :3, :3], transform[..., :3,[3]]

def load_data_as_memmap(filename, directory):
    directory = pathlib.Path(directory)
    dataset = np.load(filename)
        
    output = {
        'K': dataset['K'],
        'transforms': dataset['transforms'],
    }
    
    # Send the RGB & depth data to a file on disk to avoid OOM
    for key in ('rgbs', 'depths'):
        data = dataset[key]
        filename = directory / f'{key}.npy'
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=data.shape)
        fp[:] = data[:]
        output[key] = np.memmap(filename, dtype='float32', mode='r', shape=data.shape)
    return output
class CarlaTriplesDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset):
        self.K = torch.from_numpy(dataset['K']).float()
        print(self.K)
        self.K[0] = self.K[0] / 800
        self.K[1] = self.K[1] / 600
        self.rgbs = dataset['rgbs']
        self.depths = dataset['depths']
        self.transforms = dataset['transforms']

    def __len__(self):
        # The size of the dataset should be the number of triples
        # We will assume that each query image uses its adjacent images,
        # so we have N - 2 query images
        return len(self.transforms) - 2

    def __getitem__(self, idx):
        # Our query images in CHW format
        # Note for Yuxin: The images needed to be divided by 255!
        image_0 = torch.from_numpy(self.rgbs[idx+0].copy()).permute(2, 0, 1) / 255
        image_q = torch.from_numpy(self.rgbs[idx+1].copy()).permute(2, 0, 1) / 255
        image_1 = torch.from_numpy(self.rgbs[idx+2].copy()).permute(2, 0, 1) / 255
        # Our depth images i
        depth_0 = torch.from_numpy(self.depths[idx+0].copy() * 1000)
        depth_q = torch.from_numpy(self.depths[idx+1].copy() * 1000)
        depth_1 = torch.from_numpy(self.depths[idx+2].copy() * 1000)

        # Deal with the handness difference between intrinsic and carla extrinsic
        # The mapping is x, y, z -> -z, x, y
        axes = torch.tensor([
            [0, 0,-1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]).double()
        
        # Our poses
        extrinsic_0 = torch.from_numpy(self.transforms[idx+0]) @ axes
        extrinsic_q = torch.from_numpy(self.transforms[idx+1]) @ axes
        extrinsic_1 = torch.from_numpy(self.transforms[idx+2]) @ axes
        
        transform_0q = torch.linalg.inv(extrinsic_0) @ extrinsic_q
        transform_1q = torch.linalg.inv(extrinsic_1) @ extrinsic_q
        transform_10 = torch.linalg.inv(extrinsic_1) @ extrinsic_0
        
        #transformation: resize from 600*800 to 120 160
        # resize = torchvision.transforms.Resize((120,160))
        def resize(x):
            # return torchvision.transforms.functional.resize(x, (300, 400), antialias=True)
            return torchvision.transforms.functional.resize(x,(225,300),antialias=True)
            # return x[..., 0:200, 50:350]
            # return  x[..., 0:200, 50:350]
        
        return (
            self.K.unsqueeze(0),
            resize(image_q),
            resize(image_0),
            resize(image_1),
            resize(depth_q.unsqueeze(0)).squeeze(),
            resize(depth_0.unsqueeze(0)).squeeze(),
            resize(depth_1.unsqueeze(0)).squeeze(),
            transform_0q.float(),
            transform_1q.float(),
            transform_10.float(),
        )

def apply_transform(points, transform):
    """
    Project points to camera reference frame
    """
    R, t = transform[..., :3, :3], transform[..., :3, [3]]
    return R @ points + t
def perspective(points, epsilon=1e-8):
    """
    Perspective division
    """
    return points[..., :-1, :] / (points[..., [-1], :] + epsilon)
def project(points, K, height, width):
    """
    Project 3D points into a 2D image with intrinsic matrix K
    """
    coords = K @ points
    coords = perspective(coords)
    coords = coords.view(-1, height, width, 2)
    coords = (coords - 0.5) * 2
    return coords
def backproject(depth, K):
    """
    Backproject a depth map into 3D space with normalized intrinsic matrix K.
    The depth map should be normalized between (min_depth, max_depth).
    """
    height, width = depth.shape[-2:]
    size = height * width
    y = torch.linspace(0, 1, height, device=depth.device, dtype=depth.dtype)
    x = torch.linspace(0, 1, width,  device=depth.device, dtype=depth.dtype)
    u, v = torch.meshgrid(x, y, indexing='xy')
    ones = torch.ones(size, device=depth.device)
    points = torch.column_stack([u.ravel(), v.ravel(), ones]).view(-1, size, 3, 1)
    points = torch.linalg.inv(K) @ points
    return points * depth.view(-1, size, 1, 1)
def pix_coords(height, width, device):
    """
    Create a grid of camera pixels in normalized device coordinates
    """
    size = height * width
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    u, v = torch.meshgrid(x, y, indexing='xy')
    ones = torch.ones(size, device=device)
    return torch.column_stack([u.ravel(), v.ravel(), ones]).view(-1, size, 3, 1)
def warp(homography, image, padding_mode="border"):
    """
    Warp an image using homography
    """
    height, width = image.shape[-2:]
    coords = pix_coords(height, width, device=device)
    warped = project(homography @ coords, height, width)
    warped = F.grid_sample(image, warped, padding_mode=padding_mode, align_corners=False)
    return warped
def grid_sample(image, flow):
    """
    Hacky implementation of torch.nn.functional.grid_sample,
    created so that the derivative can be computed with jacfwd.
    """
    N, C, IH, IW = image.shape
    _, H, W, _ = flow.shape

    ix = flow[..., 0]
    iy = flow[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    with torch.no_grad():
        ix_nw = torch.clamp(ix_nw, 0, IW-1)
        iy_nw = torch.clamp(iy_nw, 0, IH-1)
        ix_ne = torch.clamp(ix_ne, 0, IW-1)
        iy_ne = torch.clamp(iy_ne, 0, IH-1)
        ix_sw = torch.clamp(ix_sw, 0, IW-1)
        iy_sw = torch.clamp(iy_sw, 0, IH-1)
        ix_se = torch.clamp(ix_se, 0, IW-1)
        iy_se = torch.clamp(iy_se, 0, IH-1)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    return (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +    
        ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +       
        sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
        se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )
def resample(depth, K, transform, target):
    """
    Backproject a depth map with K and transform it into a new reference frame T.
    Sample colors from the target image.
    """
    points_3d = backproject(depth, K)
    points_3d = apply_transform(points_3d, transform)
    points_2d = project(points_3d, K, depth.shape[-2], depth.shape[-1])
    return grid_sample(target, points_2d)

class GaussNewtonFeatureOptimizer(torch.nn.Module):
    
    def __init__(self, K, features, saliency, depth):
        super().__init__()
        self.K = K
        self.features = features
        self.saliency = saliency
        self.depth = depth
        self.residuals_grad = torch.func.jacfwd(self.residuals)

    def residuals(self, params, features, saliency):
        transform = transform_from_params(params).unsqueeze(1)
        resampled = resample(self.depth, self.K, transform, features)
        # resampled_saliency = resample(self.depth, self.K, transform, saliency)
        # residuals = self.features - resampled

        residuals = torch.nn.HuberLoss(reduction='none', delta = 0.2)(self.features - resampled,torch.zeros(self.features.shape).to(device)).reshape(features.shape)
        # residuals = self.saliency * resampled_saliency *(self.features - resampled)
        # residuals = self.saliency * resampled_saliency*residuals
        residuals = residuals[...,0:190,20:280]
        return residuals.ravel()
    
    def update_step(self, params, features, saliency):
        r = self.residuals(params, features, saliency)
        J = self.residuals_grad(params, features, saliency).squeeze()
        H = J.mT @ J + 1e-6 * torch.eye(6, device=J.device)
        return torch.linalg.lstsq(H, J.mT @ -r).solution
    
    def solve(self, params, features, saliency, iterations):
        output = params.unsqueeze(0)
        for i in range(iterations):
            update = self.update_step(output.detach(), features, saliency).unsqueeze(0)
            output = params_from_transform(
                transform_from_params(output) @
                transform_from_params(update)
            )
        return output

    def forward(self, batch, features, saliency, iterations=30):
        # TODO: Try to use VMap for batching
        output = batch.clone()
        for i, params in enumerate(batch):
            output[i] = self.solve(params, features, saliency, iterations)
        return output

def transform_consistency_loss(T_q0, T_q1, T_01):
    T_1q = torch.linalg.inv(T_q1)
    transform_consistency = T_01.to(T_q0.device) @ T_1q @ T_q0
    return torch.linalg.norm(se3_log_map(transform_consistency.mT))
def transform_accuracy_loss(T_hat, T_inv_est):
    transform_accuracy = T_hat @ T_inv_est.to(T_hat.device)
    return torch.linalg.norm(se3_log_map(transform_accuracy.mT))

# memmap = load_data_as_memmap('train_data_405.npz', 'data')
memmap_d_rain = load_data_as_memmap("train_rain_dynamics_404.npz",'data/dr')
memmap_d_sun = load_data_as_memmap("train_sunny_dynamics_404.npz",'data/ds')
memmap_nd_rain = load_data_as_memmap("train_rain_noDynamics_404.npz",'data/ndr')
memmap_nd_sun =  load_data_as_memmap("train_sunny_noDynamics_404.npz",'data/nds')
# dataset = CarlaTriplesDataset(memmap)
dataset_dr = CarlaTriplesDataset(memmap_d_rain)
dataset_ds = CarlaTriplesDataset(memmap_d_sun)
dataset_ndr = CarlaTriplesDataset(memmap_nd_rain)
dataset_nds = CarlaTriplesDataset(memmap_nd_sun)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
dataloader_dr = torch.utils.data.DataLoader(dataset_dr, batch_size=1, shuffle=True)
dataloader_ds = torch.utils.data.DataLoader(dataset_ds, batch_size=1, shuffle=True)
dataloader_ndr = torch.utils.data.DataLoader(dataset_ndr, batch_size=1, shuffle=True)
dataloader_nds = torch.utils.data.DataLoader(dataset_nds, batch_size=1, shuffle=True)

net = UNet(3, 17).to(device)
# net.load_state_dict(torch.load("check_points7/checkpoint4.pth"))
net.train()
loss_batch = []
loss_history = []
optimizer = torch.optim.Adam([
    {'params': net.parameters(), 'lr': 1e-4},
    # {'params': damping, 'lr': 5e-4},
])


accum_iter = 16
# first 5 epoch run lambda = 10



#from check points
# dict = torch.load("check_points/checkpoint1.pth")
# net.load_state_dict(dict['model_state_dict'])
# tensorboard
import datetime
from tensorflow import summary
import tensorflow as tf
current_time = str(datetime.datetime.now().timestamp())
train_log_dir = 'logs/tensorboard/train/' + "huber_noSaliency"
train_summary_writer = summary.create_file_writer(train_log_dir)

torch.manual_seed(0)

def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + train_log_dir)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()


epochs = range(0, 5)
steps = 0
for e in epochs:
    lambd = 1.0 if e > 1 else 10
    group = optimizer.param_groups[0]
    # group['lr'] = 1e-4 if e < 1 else 1e-5
    group['lr']  = 1e-4 if e==0 or e ==5 else 1e-5
    for dataloader in [dataloader_ndr, dataloader_nds,dataloader_dr, dataloader_ds]:
        progress = tqdm.tqdm(dataloader)

        for batch_idx, batch in enumerate(progress):

            # set query image and 2 reference image
            K, image_q, image_0, image_1, depth_q, depth_0, depth_1, T_0q, T_1q, T_10 = (
                x.to(device)
                for x in batch
            )
            T_01 = torch.linalg.inv(T_10)

            result_0q = torch.zeros(len(K), 6, device=device)
            result_1q = torch.zeros(len(K), 6, device=device)
            result_01 = torch.zeros(len(K), 6, device=device)

            pyramid_q = net.forward(image_q)
            pyramid_0 = net.forward(image_0)
            pyramid_1 = net.forward(image_1)
            
            levels = [0, 1, 2, 3]
            iterations = [16, 12, 8, 4]

            for level in levels:
                features_q, saliency_q = pyramid_q[level]
                features_0, saliency_0 = pyramid_0[level]
                features_1, saliency_1 = pyramid_1[level]
                
                # Resample depth maps for pyramid
                size = features_q.shape[-2:]
                
                with torch.no_grad():
                    depth_0_ = torchvision.transforms.functional.resize(depth_0, size, antialias=True).unsqueeze(0)
                    depth_1_ = torchvision.transforms.functional.resize(depth_1, size, antialias=True).unsqueeze(0)

                # Align image_0 to query image
                image_0_optimizer = GaussNewtonFeatureOptimizer(K, features_0, saliency_0, depth_0_)
                result_0q = image_0_optimizer.forward(result_0q, features_q, saliency_q, iterations[level])
                
                # # Align image_1 to query image
                image_1_optimizer = GaussNewtonFeatureOptimizer(K, features_1, saliency_1, depth_1_)
                result_1q = image_1_optimizer.forward(result_1q, features_q, saliency_q, iterations[level])

                # Align image_0 to image_1
                result_01 = image_0_optimizer.forward(result_01, features_1, saliency_1, iterations[level])
            
            # Transform consistency loss
            T_q0_est = torch.linalg.inv(transform_from_params(result_0q))
            T_q1_est = torch.linalg.inv(transform_from_params(result_1q))
            consistency_loss_01 = transform_consistency_loss(T_q0_est, T_q1_est, T_01)
            consistency_loss_10 = transform_consistency_loss(T_q1_est, T_q0_est, T_10)
            consistency_loss = consistency_loss_01 + consistency_loss_10

            # Compute transform accuracy loss
            T_01_est = transform_from_params(result_01)
            accuracy_loss = transform_accuracy_loss(T_10, T_01_est)
            
            # Total loss
            loss  = consistency_loss + lambd * accuracy_loss
            # loss = accuracy_loss
            loss = accuracy_loss / accum_iter

            loss.backward()
            loss_batch.append(loss.item())
            progress.set_postfix({
                'loss': loss.item(),
                'accuracy': accuracy_loss.item(),
                'consistency': consistency_loss.item(),
                'lambda': lambd,
                'lr': group['lr'],

            })
            # with train_summary_writer.as_default():
            #     tf.summary.scalar('loss', loss.item(), step=steps)
            #     tf.summary.scalar('accuracy',accuracy_loss.item(),step=steps)
            #     tf.summary.scalar('consistency', consistency_loss.item(),step=steps)
            # steps += 1

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                loss_history.append(sum(loss_batch))
                loss_batch = []
                optimizer.step()
                optimizer.zero_grad()
            # torch.cuda.empty_cache()
    torch.save(net.state_dict(),'no_saliency/checkpoint{}.pth'.format(str(e)))

