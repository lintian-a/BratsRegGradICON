import os

# data_folder = "/playpen-raid2/lin.tian/data/BraTS-Reg/BraTSReg_Training_Data_v3"
data_folder = "/playpen-raid2/lin.tian/data/BraTS-Reg/BraTSReg_Validation_Data"

case_list = os.listdir(data_folder)
case_list = list(filter(lambda x: "BraTS" in x.split("/")[-1], case_list))
case_list = sorted(case_list)

cases = []
for c in case_list:
    file_list = os.listdir(f"{data_folder}/{c}")
    source_list = list(filter(lambda x: ("_00_" in x) and ("t1.nii" in x), file_list))
    target_list = list(filter(lambda x: ("_01_" in x) and ("t1.nii" in x), file_list))

    source_id = "_".join(source_list[0].split('_')[:-1])
    target_id = "_".join(target_list[0].split('_')[:-1])
    cases.append(f"{c}/{source_id},{c}/{target_id}")

with open(f"{data_folder}/data_id.txt", "w") as f:
    f.writelines("\n".join(cases)+"\n")

# with open(f"{data_folder}/data_id.txt", "r") as f:
#     t = list(map(lambda x: x[:-1].split(','), f.readlines()))
#     print("here")
