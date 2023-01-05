import random

import footsteps
import icon_registration as icon
import icon_registration.networks as networks
import torch
import os
import itk
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import random

from icon_registration.losses import ICONLoss, to_floats

def write_stats(writer, stats: ICONLoss, ite):
    for k, v in to_floats(stats)._asdict().items():
        writer.add_scalar(k, v, ite)



BATCH_SIZE = 2
input_shape = [BATCH_SIZE, 4, 155, 240, 240]

GPUS = 4


def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5)):
    dimension = len(input_shape) - 2
    input_channels = input_shape[1]
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension, input_channels=input_channels))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=dimension),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension, input_channels=input_channels))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension, input_channels=input_channels)))
    
    net = icon.GradientICON(inner_net, loss_fn, lmbda=lmbda)
    net.assign_identity_map(input_shape)
    return net

class BraTSDataset():
    def __init__(self, batch_size, with_augment=False, data_path="/playpen-raid2/lin.tian/data/BraTS-Reg/BraTSReg_Training_Data_v3/"):
        with open(f"{data_path}/data_id.txt", 'r') as f:
            self.pair_path = list(map(lambda x: x[:-1].split(','), f.readlines()))
        self.data_path = data_path
        self.modality_list = ['flair', 't1', 't1ce', 't2']
        self.batch_size = batch_size
        self.with_augment = with_augment
    
    def __len__(self):
        return len(self.pair_path)

    def process(self, iA):
        iA = iA.astype(np.float32)
        iA =  iA/np.amax(iA, axis=(2,3,4), keepdims=True)
        return iA
    
    def __call__(self):
        case_ids = random.choices(self.pair_path, k=self.batch_size)

        sources = []
        targets = []
        for c in case_ids:
            source = []
            target = []
            for m in self.modality_list:
                source.append(itk.imread(f"{self.data_path}/{c[0]}_{m}.nii.gz"))
                target.append(itk.imread(f"{self.data_path}/{c[1]}_{m}.nii.gz"))
            sources.append(np.array(source))
            targets.append(np.array(target))

        if self.with_augment:
            with torch.no_grad():
                return augment(torch.from_numpy(self.process(np.array(sources))).cuda(), torch.from_numpy(self.process(np.array(targets))).cuda())
        else:
            return torch.from_numpy(self.process(np.array(sources))).cuda(), torch.from_numpy(self.process(np.array(targets))).cuda()

def augment(image_A, image_B):
    identity_list = []
    for i in range(image_A.shape[0]):
        identity = torch.Tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        idxs = set((0, 1, 2))
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1 
        identity = identity * (torch.randint_like(identity, 0, 2) * 2  - 1)
        identity_list.append(identity)

    identity = torch.cat(identity_list)
    
    noise = torch.randn((image_A.shape[0], 3, 4))

    forward = identity + .05 * noise  

    warped_A = F.grid_sample(
        image_A, 
        F.affine_grid(forward.cuda(), image_A.shape, align_corners=True), 
        padding_mode='border', 
        align_corners=True)

    noise = torch.randn((image_A.shape[0], 3, 4))
    forward = identity + .05 * noise  

    warped_B = F.grid_sample(
        image_B, 
        F.affine_grid(forward.cuda(), image_B.shape, align_corners=True), 
        padding_mode='border',
        align_corners=True)

    return warped_A, warped_B

def train_two_stage(input_shape, batch_function, GPUS, ITERATIONS_PER_STEP):

    net = make_network(input_shape, include_last_step=False)

    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    icon.train_batchfunction(net_par, optimizer, batch_function, unwrapped_net=net, steps=ITERATIONS_PER_STEP)
    
    torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + "Step_1_final.trch",
            )

    net_2 = make_network(input_shape, include_last_step=True)

    net_2.regis_net.netPhi.load_state_dict(net.regis_net.state_dict())

    del net
    del net_par
    del optimizer

    if GPUS == 1:
        net_2_par = net_2.cuda()
    else:
        net_2_par = torch.nn.DataParallel(net_2).cuda()
    optimizer = torch.optim.Adam(net_2_par.parameters(), lr=0.00005)

    net_2_par.train()
    
    # We're being weird by training two networks in one script. This hack keeps
    # the second training from overwriting the outputs of the first.
    footsteps.output_dir_impl = footsteps.output_dir + "2nd_step/"
    os.makedirs(footsteps.output_dir)

    icon.train_batchfunction(net_2_par, optimizer, batch_function, unwrapped_net=net_2, steps=ITERATIONS_PER_STEP )
    
    torch.save(
                net_2.regis_net.state_dict(),
                footsteps.output_dir + "Step_2_final.trch",
            )

if __name__ == "__main__":
    footsteps.initialize()
    brains_loader = BraTSDataset(BATCH_SIZE*GPUS, with_augment=True)

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    
    train_two_stage(input_shape, brains_loader, GPUS, 10000)
