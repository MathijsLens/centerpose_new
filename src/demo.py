# python demo.py --demo ../images/ford --arch dlav1_34 --load_model /home/mathijs/ford/code/CenterPose/CenterPose/models/CenterPose/cereal_box_v1_140.pth
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md
# python demo.py --demo ../images/ford --arch res --load_model /home/mathijs/ford/code/CenterPose/CenterPose/models/CenterPose/cereal_box_resnet_140.pth
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.utils.pnp.cuboid_pnp_shell import pnp_shell
import os
import cv2
import json
from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
import glob
import numpy as np

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'pnp', 'track']

# for ford: 
def get_annotations(opt):
    """open the annotations from the 2 jsonfiles (train and validation)
        save the annotations in variable data
    param: root_json_gt: path to the root of the json files
    """
    # take the green_background jsons as GT:  
    if opt.green_background:
        with open("../data/green_boxes/train.json", "r") as f:
            train=json.load(f)


        with open("../data/green_boxes/val.json", "r") as f:
            val=json.load(f)
    else:    
        with open("../data/random_background/train/train_backgrounds.json", "r") as f:
                train=json.load(f)


        with open("../data/random_background/val/val_backgrounds.json", "r") as f:
            val=json.load(f)
    data= train+ val
    
    return data




def demo(opt, meta):
    """
    Used to test the network. 
    Use opt.debug 4 for testing and opt.debug 7 for checking the accuracy.
    

    Args:
        opt (_type_): Configuration parameters for the network
        meta (_type_): meta data for the network and post_processing (like the intricic camerea matrix)
    
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    
    # print the mode of the point representation (see opt.rep_mode)
    print("rep_mode: ",opt.rep_mode) 
    if opt.ford:
        print("Use Green Background? ",opt.green_background)
        annotations=get_annotations(opt)
    else: 
        annotations=None
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    
    
    # change the intricique cam matrix at the bottem of this file! (line 224)
    print("camera_matrix:", meta["camera_matrix"])
    
    
    if opt.use_pnp == True and 'camera_matrix' not in meta.keys():
        raise RuntimeError('Error found. Please give the camera matrix when using pnp algorithm!')

    
    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False

        # Check if camera opened successfully
        if (cam.isOpened() == False):
            print("Error opening video stream or file")

        idx = 0
        while (cam.isOpened()):
            _, img = cam.read()
            # img=cv2.resize(img, [512, 512])
            # try:
            #     cv2.imshow('input', img)
            # except:
            #     exit(1)

            filename = os.path.splitext(os.path.basename(opt.demo))[0] + '_' + str(idx).zfill(
                4) + '.png'
            ret = detector.run(img, meta_inp=meta,
                               filename=filename)
            idx = idx + 1

            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(f'Frame {str(idx).zfill(4)}|' + time_str)
            if cv2.waitKey(1) == 27:
                break
    else:

        # # Option 1: only for XX_test with a lot of sub-folders
        # image_names = []
        # for ext in image_ext:
        #     file_name=glob.glob(os.path.join(opt.demo,f'**/*.{ext}'))
        #     if file_name is not []:
        #         image_names+=file_name

        # Option 2: if we have images just under a folder, uncomment this instead
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        detector.pause = False
        
        for idx, image_name in enumerate(image_names):
            # Todo: External GT input is not enabled in demo yet
            
            
            if opt.debug==7:
                # if debug mode is 7 we use the demo fuction to create jsonfiles to check the annotations
                img_name=image_name.split("/")[-1]
                annotation=None
                print(len(annotations))
                for dict in annotations:
                    if img_name == dict["image_name"] and dict["points"]!= []:
                        annotation=dict
                        break
                if annotation==None:
                    print("skipping image, no annotation")
                else:
                    ret = detector.run(image_name, meta_inp=meta, annotation=annotation)
                    time_str = ''
                    for stat in time_stats:
                        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                    print(f'Frame {idx}|' + time_str)
            else: 
                # test the images -> no annotation needed
                ret = detector.run(image_name, meta_inp=meta)
                time_str = ''
                for stat in time_stats:
                    time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                print(f'Frame {idx}|' + time_str)


if __name__ == '__main__':

    # Default params with commandline input
    opt = opts().parser.parse_args()

    # Local machine configuration example for CenterPose
    # opt.c = 'cup' # Only meaningful when enables show_axes option
    # opt.green_background=False
    
    if opt.green_background:
        opt.load_model = "../models/green_background_best.pth"
        
    else: 
        opt.load_model= "../models/background_random_best.pth"
        # opt.load_model = "../exp/object_pose/background_scale/cereal_best.pth"
        
    # for webcam use -> uncomment
    # opt.demo = "webcam"
    opt.arch = 'res_101'

    
    # opt.debug = 4 # save intermediate results 
    # opt.debug = 7 # test images with annotations, We can use the eval.py script to check the jsons for 3Diou accuracy
    # opt.debug=4 # show intermidiate results 
    
    
    # Default setting
    opt.nms = True
    opt.obj_scale = True

    # Tracking stuff
    if opt.tracking_task == True:
        print('Running tracking')
        opt.pre_img = True
        opt.pre_hm = True
        opt.tracking = True
        opt.pre_hm_hp = True
        opt.tracking_hp = True
        opt.track_thresh = 0.1

        opt.obj_scale_uncertainty = True
        opt.hps_uncertainty = True
        opt.kalman = True
        opt.scale_pool = True

        opt.vis_thresh = max(opt.track_thresh, opt.vis_thresh)
        opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
        opt.new_thresh = max(opt.track_thresh, opt.new_thresh)

        # # For tracking moving objects, better to set up a small threshold
        # opt.max_age = 2

        print('Using tracking threshold for out threshold!', opt.track_thresh)

    # PnP related
    meta = {}
    if opt.cam_intrinsic is None:
        meta['camera_matrix'] = np.array(            
            # ford camera matrix see meta data of img to determine focallenth
            # [[3648, 0, 2736], [0, 3648, 1824], [0, 0, 1]], dtype=np.float32)  
            [[805, 0, 320], [0, 805, 240], [0, 0, 1]], dtype=np.float32)  
        opt.cam_intrinsic = meta['camera_matrix']
    else:
        meta['camera_matrix'] = np.array(opt.cam_intrinsic).reshape(3, 3)

    opt.use_pnp = True

    # Update default configurations
    opt = opts().parse(opt)
    # Update dataset info/training params
    opt = opts().init(opt)
    demo(opt, meta)
