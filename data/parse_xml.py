import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import numpy as np

def get_file_info(path):
    root=ET.parse(path).getroot()
    filename=root.find('filename').text
    folder=root.find('folder').text
    width=root.find('size').find('width').text
    height=root.find('size').find('height').text
    depth=root.find('size').find('depth').text
    bndbox=root.find('object').find('bndbox')
    xmin=bndbox.find('xmin').text
    ymin=bndbox.find('ymin').text
    xmax=bndbox.find('xmax').text
    ymax=bndbox.find('ymax').text
    file_info_dict={}
    file_info_dict['folder']=folder
    file_info_dict['width']=width
    file_info_dict['height']=height
    file_info_dict['depth']=depth
    file_info_dict['xmin']=xmin
    file_info_dict['ymin']=ymin
    file_info_dict['xmax']=xmax
    file_info_dict['ymax']=ymax
    return filename, file_info_dict
    
#     print('filename={},folder={}'.format(filename,folder))
#     print('width={},height={},depth={}'.format(width,height,depth))
#     print('xmin={},ymin={},xmax={},ymax={}'.format(xmin.text,ymin.text,xmax.text,ymax.text))
#     return [filename,folder,width,height,depth,xmin,ymin,xmax,ymax]
    
# path='/Users/chendanxia/sophie/segmentation_img_set/imagenet/ILSVRC2012_bbox_train_v2/n15075141/n15075141_17815.xml'
# get_file_info(path)

def parse_all_file_info():
    dir_path='/Users/chendanxia/sophie/segmentation_img_set/imagenet/ILSVRC2012_bbox_train_v2/'
    dst_file='/Users/chendanxia/sophie/segmentation_img_set/imagenet/bbox_infos.npy'
    dir_path='/Users/chendanxia/Downloads/imagenetannotations/Annotation/'
    dst_file='/Users/chendanxia/sophie/segmentation_img_set/imagenet/bbox_infos_all.npy'
    names_file='/Users/chendanxia/sophie/segmentation_img_set/imagenet/names.npy'
    names=np.load(names_file)
    files_info_dict={}
    for sub_dir in tqdm(os.listdir(dir_path)):
        sub_dir_path=os.path.join(dir_path,sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        for file_name in os.listdir(sub_dir_path):
            class_name=file_name.split('_')[0]
            if class_name not in names:
                continue
            file_path=os.path.join(sub_dir_path,file_name)
            name,file_info_dict=get_file_info(file_path)
            files_info_dict[name]=file_info_dict
#             info=get_file_info(file_path)
#             infos.append(info)
    print(len(files_info_dict.keys()))
    np.save(dst_file, files_info_dict)

def analysis():
    dst_file='/Users/chendanxia/sophie/segmentation_img_set/imagenet/bbox_infos_all.npy'
    d=np.load(dst_file).item()
    print(len(d.keys()))
    dst_file2='/Users/chendanxia/sophie/segmentation_img_set/imagenet/bbox_infos.npy'
    d2=np.load(dst_file2).item()
    print(len(d2.keys()))
    keys=d.keys()
    keys2=d2.keys()
    for key in tqdm(keys2):
        if key not in keys:
            d[key]=d2[key]
    np.save('/Users/chendanxia/sophie/segmentation_img_set/imagenet/bbox_infos_final.npy',d)
    
    
# parse_all_file_info()
analysis()