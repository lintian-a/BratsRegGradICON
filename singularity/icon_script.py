import argparse
import glob
import os

import icon_registration as icon
import icon_registration.pretrained_models
import itk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from icon_registration import config, networks
from icon_registration.itk_wrapper import create_itk_transform
from icon_registration.losses import to_floats


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
    
    net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda)
    net.assign_identity_map(input_shape)
    return net

def finetune_execute(model, image_A, image_B, steps):
    state_dict = model.state_dict()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.regis_net.netPsi.net.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.regis_net.netPsi.net.parameters(), lr=0.00005)
    for _ in range(steps):
        optimizer.zero_grad()
        loss_tuple = model(image_A, image_B)
        print(loss_tuple)
        loss_tuple[0].backward()
        optimizer.step()
        del loss_tuple
    with torch.no_grad():
        loss = model(image_A, image_B)
    model.load_state_dict(state_dict)
    return loss

def register_pair_with_multimodalities(
    model, image_A: list, image_B: list, finetune_steps=None, return_artifacts=False
) -> "(itk.CompositeTransform, itk.CompositeTransform)":

    assert len(image_A) == len(image_B), "image_A and image_B should have the same number of modalities."

    # send model to cpu or gpu depending on config- auto detects capability
    model.to(config.device)

    A_npy, B_npy = [], []
    for image_a, image_b in zip(image_A, image_B):
        assert isinstance(image_a, itk.Image)
        assert isinstance(image_b, itk.Image)

        A_npy.append(np.array(image_a))
        B_npy.append(np.array(image_b))

        assert(np.max(A_npy[-1]) != np.min(A_npy[-1]))
        assert(np.max(B_npy[-1]) != np.min(B_npy[-1]))

    # turn images into torch Tensors: add batch dimensions (each of length 1)
    A_trch = torch.Tensor(np.array(A_npy)).to(config.device)[None]
    B_trch = torch.Tensor(np.array(B_npy)).to(config.device)[None]

    shape = model.identity_map.shape[2:]
    if list(A_trch.shape[2:]) != list(shape) or (list(B_trch.shape[2:]) != list(shape)):
        # Here we resize the input images to the shape expected by the neural network. This affects the
        # pixel stride as well as the magnitude of the displacement vectors of the resulting
        # displacement field, which create_itk_transform will have to compensate for.
        A_trch = F.interpolate(
            A_trch, size=shape, mode="trilinear", align_corners=False
        )
        B_trch = F.interpolate(
            B_trch, size=shape, mode="trilinear", align_corners=False
        )

    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        with torch.no_grad():
            loss = model(A_trch, B_trch)
    else:
        loss = finetune_execute(model, A_trch, B_trch, finetune_steps)

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    if hasattr(model, "prepare_for_viz"):
        with torch.no_grad():
            model.prepare_for_viz(A_trch, B_trch)
    phi_AB = model.phi_AB(model.identity_map)
    phi_BA = model.phi_BA(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        create_itk_transform(phi_AB, model.identity_map, image_A[0], image_B[0]),
        create_itk_transform(phi_BA, model.identity_map, image_B[0], image_A[0]),
    )
    if not return_artifacts:
        return itk_transforms
    else:
        return itk_transforms + (to_floats(loss),)

class GradientICONSparse(icon.GradientICON):
    def compute_gradient_icon_loss(self, phi_AB, phi_BA):
        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(self.identity_map.device)
            * 1
            / self.identity_map.shape[-1]
        )[:, :, ::2, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = phi_AB(phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(self.identity_map.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(self.identity_map.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(
                self.identity_map.device
            )
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(self.identity_map.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = phi_AB(phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        return inverse_consistency_loss

def get_model():
    input_shape = [1, 4, 155, 240, 240]

    # Read in your trained model
    model = make_network(input_shape, include_last_step=True)
    # model.regis_net.load_state_dict(torch.load("./singularity/Step_2_final.trch", map_location='cpu'))
    # model.regis_net.load_state_dict(torch.load("/playpen-raid2/lin.tian/projects/icon_lung/ICON/results/BraTS/gradicon_with_augment/debug/Step_1_final.trch", map_location='cpu'))
    # model.regis_net.load_state_dict(torch.load("/playpen-raid2/lin.tian/projects/BratsRegGradICON/results/BraTSReg/gradicon_with_augmentation/debug/2nd_step/Step_2_final.trch", map_location='cpu'))
    # model.regis_net.load_state_dict(torch.load("/playpen-raid2/lin.tian/projects/BratsRegGradICON/results/BraTSReg/gradicon_crosspatient_train/debug/2nd_step/Step_2_final.trch", map_location='cpu'))
    # model.regis_net.load_state_dict(torch.load("/playpen-raid2/lin.tian/projects/BratsRegGradICON/results/BraTSReg/gradicon_finetune/on_crosspatient/debug/2nd_step/Step_2_final.trch", map_location='cpu'))
    # model.regis_net.load_state_dict(torch.load("/playpen-raid2/lin.tian/projects/BratsRegGradICON/results/BraTSReg/gradicon_finetune/on_crosspatient_continue/2nd_step/Step_2_final.trch", map_location='cpu'))
    # model.regis_net.load_state_dict(torch.load("/playpen-raid2/lin.tian/projects/BratsRegGradICON/results/BraTSReg/gradicon/debug/2nd_step/Step_2_final.trch", map_location='cpu'))
    # model.regis_net.load_state_dict(torch.load("/playpen-raid2/lin.tian/projects/BratsRegGradICON/results/BraTSReg/gradicon_with_aug_cross_patient/debug/2nd_step/Step_2_final.trch", map_location='cpu'))
    model.regis_net.load_state_dict(torch.load("/playpen-raid2/lin.tian/projects/BratsRegGradICON/results/BraTSReg/gradicon_finetune/on_with_aug_cross_patient/2nd_step/Step_2_final.trch", map_location='cpu'))
    # model.regis_net.load_state_dict(torch.load("/usr/local/bin/Step_2_final.trch"))
    return model

def cast_itk_image_to_float(image):
    if type(image) == itk.Image[itk.SS, 3] :
        cast_filter = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()
        cast_filter.SetInput(image)
        cast_filter.Update()
        image = cast_filter.GetOutput()
    return image

def cast_itk_transformation_to_dispfield(tr, reference):
    filter = itk.TransformToDisplacementFieldFilter[itk.itkImagePython.itkImageVF33, itk.D].New()
    decorator = itk.DataObjectDecorator[itk.Transform[itk.D, 3, 3]].New()
    decorator.Set(tr)
    filter.SetInput(decorator)
    filter.SetReferenceImage(reference)
    filter.SetUseReferenceImage(True)
    filter.Update()
    return filter.GetOutput()

def read_itk_dispfield(path):
    fieldReader = itk.ImageFileReader[itk.VectorImage[itk.D,3]].New()
    fieldReader.SetFileName(path)
    fieldReader.Update()
    disp = fieldReader.GetOutput()
    disp_tr = itk.DisplacementFieldTransform[(itk.D, 3)].New()
    disp_tr.SetDisplacementField(disp)
    return disp_tr

def apply_field_on_image(tr, moving, interpolation_type):
    return itk.resample_image_filter(
        moving,
        use_reference_image = True,
        reference_image=moving,
        transform=tr,
        interpolator=itk.LinearInterpolateImageFunction.New(moving) if interpolation_type=="trilinear" else itk.NearestNeighborInterpolateImageFunction.New(moving),
        default_pixel_value=0.
    )

def generate_output(args):
    """
    Generates landmarks, detJ, deformation fields (optional), and followup_registered_to_baseline images (optional) for challenge submission
    """
    print("generate_output called")

    input_path = os.path.abspath(args["input"])
    output_path = os.path.abspath(args["output"])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(
        f"* Found following data in the input path {input_path}=",
        os.listdir(input_path),
    )  # Found following data in the input path /input= ['BraTSReg_001', 'BraTSReg_002']
    print(
        "* Output will be written to=", output_path
    )  # Output will be written to= /output

    # You can first check what devices are available to the singularity
    # setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Additional Info when using cuda
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = get_model()

    # Now we iterate through each subject folder under input_path
    for subj_path in glob.glob(os.path.join(input_path, "BraTSReg*")):
        subj = os.path.basename(subj_path)
        print(
            f"Now performing registration on {subj}"
        )  # Now performing registration on BraTSReg_001

        file_list = os.listdir(subj_path)
        source_list = list(filter(lambda x: ("_00_" in x) and ("t1.nii" in x), file_list))
        target_list = list(filter(lambda x: ("_01_" in x) and ("t1.nii" in x), file_list))
        source_id = "_".join(source_list[0].split('_')[:-1])
        target_id = "_".join(target_list[0].split('_')[:-1])

        # Read in your data
        source = []
        target = []
        for m in ['flair', 't1', 't1ce', 't2']:
            source.append(icon_registration.pretrained_models.brain_network_preprocess(
                itk.imread(f"{subj_path}/{source_id}_{m}.nii.gz")
            ))

            target.append(icon_registration.pretrained_models.brain_network_preprocess(
                itk.imread(f"{subj_path}/{target_id}_{m}.nii.gz")
            ))

        phi_pre_post, phi_post_pre = register_pair_with_multimodalities(
            model, source, target, finetune_steps=50
        )

        # Make your prediction segmentation file for case BraTSReg_001

        ## 1. calculate the output landmark points
        post_landmarks = pd.read_csv(os.path.join(subj_path, f"{target_id}_landmarks.csv")).values

        pre_landmarks = np.array(
            [
                [i+1] + list(phi_pre_post.TransformPoint(t[1:]))
                for i, t in enumerate(post_landmarks * 1.0)
            ]
        )


        np.savetxt(os.path.join(args["output"], f"{subj}.csv"), pre_landmarks, header="Landmark,X,Y,Z", delimiter=",", fmt=['%i','%f','%f','%f'], comments='')

        
        ## 2. calculate the determinant of jacobian of the deformation field
        displacement_image_itk = cast_itk_transformation_to_dispfield(phi_pre_post, itk.imread(glob.glob(os.path.join(subj_path, f"{subj}_00_*_t1.nii.gz"))[0]))
        det_itk = itk.displacement_field_jacobian_determinant_filter(displacement_image_itk)
        itk.imwrite(det_itk, os.path.join(args["output"], f"{subj}_detj.nii.gz"))

        if args["def"]:
            # write both the forward and backward deformation fields to the output/ folder
            print("--def flag is set to True")
            itk.imwrite(displacement_image_itk, os.path.join(output_path, f"{subj}_df_b2f.nii.gz"))
            itk.imwrite(
                cast_itk_transformation_to_dispfield(phi_post_pre, itk.imread(glob.glob(os.path.join(subj_path, f"{subj}_01_*_t1.nii.gz"))[0])), 
                os.path.join(output_path, f"{subj}_df_f2b.nii.gz"))

        if args["reg"]:
            # write the followup_registered_to_baseline sequences (all 4 sequences provided) to the output/ folder
            print("--reg flag is set to True")
            for m in ['flair', 't1', 't1ce', 't2']:
                itk.imwrite(
                    apply_field_on_image(phi_post_pre, cast_itk_image_to_float(itk.imread(f"{subj_path}/{target_id}_{m}.nii.gz")), "trilinear"),
                    os.path.join(args["output"], f"{subj}_{m}_f2b.nii.gz")
                )



def apply_deformation(args):
    """
    Applies a deformation field on an input image and saves/returns the output
    """
    print("apply_deformation called")

    # Read the field
    f = read_itk_dispfield(args["field"])

    # Read the input image
    i = cast_itk_image_to_float(itk.imread(args['image']))

    # apply field on image and get output
    o = apply_field_on_image(f, i, args["interpolation"])

    # If a save_path is provided then write the output there, otherwise return the output
    if args["path_to_output_nifti"] is not None:
        itk.imwrite(o, args['path_to_output_nifti'])
    else:
        return o


if __name__ == "__main__":
    

    # Parse the input arguments

    parser = argparse.ArgumentParser(
        description="Argument parser for BraTS_Reg challenge"
    )

    subparsers = parser.add_subparsers()

    command1_parser = subparsers.add_parser("generate_output")
    command1_parser.set_defaults(func=generate_output)
    command1_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="/input",
        help="Provide full path to directory that contains input data",
    )
    command1_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/output",
        help="Provide full path to directory where output will be written",
    )
    command1_parser.add_argument(
        "-d",
        "--def",
        action="store_true",
        help="Output forward and backward deformation fields",
    )
    command1_parser.add_argument(
        "-r",
        "--reg",
        action="store_true",
        help="Output followup scans registered to baseline",
    )

    command2_parser = subparsers.add_parser("apply_deformation")
    command2_parser.set_defaults(func=apply_deformation)
    command2_parser.add_argument(
        "-f",
        "--field",
        type=str,
        required=True,
        help="Provide full path to deformation field",
    )
    command2_parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Provide full path to image on which field will be applied",
    )
    command2_parser.add_argument(
        "-t",
        "--interpolation",
        type=str,
        required=True,
        help="Should be nearest_neighbour (for segmentation mask type images) or trilinear etc. (for normal scans). To be handled inside apply_deformation() function",
    )
    command2_parser.add_argument(
        "-p",
        "--path_to_output_nifti",
        type=str,
        default=None,
        help="Format: /path/to/output_image_after_applying_deformation_field.nii.gz",
    )

    args = vars(parser.parse_args())

    print("* Received the following arguments =", args)

    args["func"](args)
