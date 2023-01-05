import icon_registration
import pandas as pd
import itk
import torch
import numpy as np
from icon_registration.itk_wrapper import create_itk_transform
import icon_registration.pretrained_models

from train import make_network

input_shape = [1, 4, 155, 240, 240]
data_folder = "/playpen-raid2/lin.tian/data/BraTS-Reg/BraTSReg_Training_Data_v3/"
model_weight = "/playpen-raid2/lin.tian/projects/icon_lung/ICON/results/BraTS/gradicon/debug/2nd_step/Step_2_final.trch"
device = torch.device("cuda:1")

def process(iA):
    iA = iA.astype(np.float32)
    iA =  iA/np.amax(iA, axis=(1,2,3), keepdims=True)
    return iA

with open(f"{data_folder}/data_id.txt", 'r') as f:
    pair_path = list(map(lambda x: x[:-1].split(','), f.readlines()))

model = make_network(input_shape, include_last_step=True)
model.regis_net.load_state_dict(torch.load(model_weight))
model = model.to(device)
model.eval()

mtres = []
with torch.no_grad():
    for p in pair_path:
        source_landmarks = pd.read_csv(f"{data_folder}/{p[0]}_landmarks.csv").values[:, 1:]
        target_landmarks = pd.read_csv( f"{data_folder}/{p[1]}_landmarks.csv").values[:, 1:]

        source = []
        target = []
        for m in ['flair', 't1', 't1ce', 't2']:
            # source.append(itk.imread(f"{data_folder}/{p[0]}_{m}.nii.gz"))
            # target.append(itk.imread(f"{data_folder}/{p[1]}_{m}.nii.gz"))
            source.append(icon_registration.pretrained_models.brain_network_preprocess(
                itk.imread(f"{data_folder}/{p[0]}_{m}.nii.gz")
            ))

            target.append(icon_registration.pretrained_models.brain_network_preprocess(
                itk.imread(f"{data_folder}/{p[1]}_{m}.nii.gz")
            ))
        
        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair_with_multimodalities(
                model, source, target
            )
        # image_A = torch.from_numpy(process(np.array(source)))[None].to(device)
        # image_B = torch.from_numpy(process(np.array(target)))[None].to(device)

        # model = make_network(input_shape, include_last_step=True)
        # model.regis_net.load_state_dict(torch.load(model_weight))
        # model = model.to(device)

        # model.eval()
        # with torch.no_grad():
        #     loss = model(image_A, image_B)
        #     phi_AB = model.phi_AB(model.identity_map)
        #     phi_AB = create_itk_transform(phi_AB, model.identity_map, source[1], target[1])

        warped_target_landmarks = np.array([list(phi_AB.TransformPoint(t)) for t in target_landmarks * 1.0])

        mtres.append(np.sqrt(np.sum((source_landmarks - warped_target_landmarks)**2, axis=1)).mean())

    print(np.mean(np.array(mtres)))