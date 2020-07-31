import os
from shutil import copyfile
import random

# sample_num = 20
sampled_imgs = 0

os.makedirs('test_img_mask_skin', exist_ok=True)

random.seed(1234)
mask_skin_attribtue_lines = open('CelebAMask-HQ-attribute-anno-skin.txt').readlines()
img_list = mask_skin_attribtue_lines[2:]
random.shuffle(img_list)
img_list = img_list[:2000]
with open('celeba_hq_mask_skin_test_img_list.txt', 'w') as f:
    for idx, img_line in enumerate(img_list):
        img_name = img_line.strip().split()[0]
        src_file = os.path.join('/D_data/Face_Editing/face_editing/data/celeba_mask_hq/CelebA-HQ-img', img_name)
        tgt_file = os.path.join('test_img_mask_skin', img_name)
        f.write(img_line)
        copyfile(src_file, tgt_file)

        # sampled_imgs += 1
        # if sampled_imgs >= sample_num:
        #     break
