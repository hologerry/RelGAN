import os
import numpy as np
from shutil import copyfile

imgIndex = np.load("imgIndex.npy", allow_pickle=True)


img_fs = []
for it in imgIndex:
    if it is not None:
        img_fs.append(int(it.split('.')[0]))

print(len(img_fs))
print(min(img_fs))
print(max(img_fs))

attribute_lines = open('list_attr_celeba.txt', 'r').readlines()[1:]  # use image name as idx, from 1, so keep the attribute name for offset

print(len(attribute_lines))

sample_num = 200
sampled_imgs = 0
with open('celeba_test_img_list.txt', 'w') as f:
    for idx in range(1, 202599+1):
        if idx in img_fs:
            continue
        else:
            print(idx)
            src_file = os.path.join('/Users/Gerry/Desktop/face_editing/datasets/celeba/img_align_celeba', f"{idx:06d}.jpg")
            tgt_file = os.path.join('test_img', f'celeba_test_{idx:06d}.jpg')
            f.write(attribute_lines[idx])
            copyfile(src_file, tgt_file)
            sampled_imgs += 1
            if sampled_imgs >= sample_num:
                break


# prepare celeba test dataset
target_celeba_test_dir = '/Users/Gerry/Desktop/Face_Editing/datasets/celeba_test/'
with open(os.path.join(target_celeba_test_dir, 'list_attr_celeba.txt'), 'w') as f:
    for idx in range(1, 202599+1):
        if idx in img_fs:
            continue
        else:
            print(idx)
            src_file = os.path.join('/Users/Gerry/Desktop/face_editing/datasets/celeba/img_align_celeba', f"{idx:06d}.jpg")
            tgt_file = os.path.join(target_celeba_test_dir, 'images', f'{idx:06d}.jpg')
            f.write(attribute_lines[idx])
            copyfile(src_file, tgt_file)
