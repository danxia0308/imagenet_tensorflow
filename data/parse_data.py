import numpy as np
import os
import imageio
from tqdm import tqdm
import pdb

base_dir='/data/sophie_bak/'
img_dir=os.path.join(base_dir,'imagenet_train')
bbox_path=os.path.join(base_dir,'bbox_infos.npy')
dst_img_dir='/data/sophie_bak/imagenet_train_crop'
bbox_dict=np.load(bbox_path).item()
for dir_name in tqdm(os.listdir(img_dir)):
    dir_path = os.path.join(img_dir,dir_name)
    dst_dir=os.path.join(dst_img_dir,dir_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path,file_name)
        clear_file_name=file_name.split('.')[0]
        bbox_info=bbox_dict.get(clear_file_name)
        if(bbox_info == None):
            print('{} does not has bbox info.'.format(file_path))
            continue
#         pdb.set_trace()
        xmin=int(bbox_info['xmin'])
        ymin=int(bbox_info['ymin'])
        xmax=int(bbox_info['xmax'])
        ymax=int(bbox_info['ymax'])
        img=imageio.imread(file_path)
        img_dst=img[xmin:ymin,xmax:ymax]
        dst_img_path=os.path.join(dst_img_dir,dir_name,file_name)
        print(dst_img_path)
        imageio.imsave(dst_img_path,img_dst)
        