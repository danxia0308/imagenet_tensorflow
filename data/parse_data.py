import numpy as np
import os
import imageio
from tqdm import tqdm
import pdb

def crop_train_img():
    base_dir='/data/sophie_bak/'
    img_dir=os.path.join(base_dir,'imagenet_train')
    bbox_path=os.path.join(base_dir,'bbox_infos.npy')
    dst_img_dir='/data/sophie_bak/imagenet_train_crop'
    bbox_dict=np.load(bbox_path).item()
    dirs=os.listdir(img_dir)
    dirs.sort()
    for dir_name in tqdm(dirs):
        dir_path = os.path.join(img_dir,dir_name)
        dst_dir=os.path.join(dst_img_dir,dir_name)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        subdirs=os.listdir(dir_path)
        subdirs.sort()
        for file_name in subdirs:
            dst_img_path=os.path.join(dst_img_dir,dir_name,file_name)
            if os.path.exists(dst_img_path):
                continue
            file_path = os.path.join(dir_path,file_name)
            clear_file_name=file_name.split('.')[0]
            bbox_info=bbox_dict.get(clear_file_name)
            if(bbox_info == None):
                continue
            xmin=int(bbox_info['xmin'])
            ymin=int(bbox_info['ymin'])
            xmax=int(bbox_info['xmax'])
            ymax=int(bbox_info['ymax'])
            width=int(bbox_info['width'])
            height=int(bbox_info['height'])
            img=imageio.imread(file_path)
            img_dst=img[ymin:ymax,xmin:xmax]
            if img.ndim==2:
                img_height,img_width=img.shape
            else:
                img_height,img_width,img_channel=img.shape
            if (img_height != height or img_width != width):
                print('{} origin_height={} origin_width={} ann_height={} ann_width={}'.format(file_path, height, width,img_height,img_width))
                continue
            try:
                imageio.imsave(dst_img_path,img_dst)
            except Exception as e:
                print('file_path={}'.format(file_path))
                print(e)

crop_train_img()
    