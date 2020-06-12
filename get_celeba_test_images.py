import os
import numpy as np
import random
from shutil import copyfile

imgIndex = np.load("imgIndex.npy", allow_pickle=True)


img_fs = []
for it in imgIndex:
    if it is not None:
        img_fs.append(int(it.split('.')[0]))

print(len(img_fs))
print(min(img_fs))
print(max(img_fs))

sample_num = 20
sampled_imgs = 0
with open('celeba_test_img_list.txt', 'w') as f:
    while sampled_imgs < sample_num:
        random_idx = random.randint(1, 202599)
        if random_idx in img_fs:
            continue
        else:
            print(random_idx)
            src_file = os.path.join('/Users/Gerry/Desktop/face_editing/datasets/celeba/img_align_celeba', f"{random_idx:06d}.jpg")
            tgt_file = os.path.join('test_img', f'celeba_test_{random_idx:06d}.jpg')
            f.write(f'celeba_test_{random_idx:06d}.jpg\n')
            copyfile(src_file, tgt_file)
            sampled_imgs += 1
