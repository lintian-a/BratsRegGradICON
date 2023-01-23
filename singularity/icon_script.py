import argparse
import glob
import os

import itk
import numpy as np
import pandas as pd
import torch
import icon_registration
import icon_registration.pretrained_models
from icon_registration import itk_wrapper
from icon_registration import networks
import icon_registration as icon
import SimpleITK as sitk

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
    model.regis_net.load_state_dict(torch.load("/usr/local/bin/Step_2_final.trch"))
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

        phi_pre_post, phi_post_pre = itk_wrapper.register_pair_with_multimodalities(
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
