import cv2
import json
from src.lib.utils.pnp.cuboid_pnp_shell import pnp_shell
import matplotlib.pyplot as plt
import numpy as np
from src.lib.opts import opts
from itertools import permutations
import os
import math


def find_correct_size(size, points, meta, opt):
    correct_size=0
    big_value=10000000
    big_size_list=(list(permutations(size)))
    for size in big_size_list:
        bbox= {'kps': points, "obj_scale": size}
        projected_points, point_3d_cam, scale, points_ori, bbox=pnp_shell(opt, meta, bbox, points, size, OPENCV_RETURN=False)
        d=dist(bbox["projected_cuboid"], points)
        if d < big_value:
            big_value=d
            correct_size=size
    return correct_size
    

def dist(pnp, anno):
    distance=0
    for a, p in zip(pnp, anno):
        distance+=math.dist(a, p)
    return distance


def main():
    with open("data/outf_all/ford/order_train_size.json", 'r') as f:
       data=json.load(f)
    print(len(data))
    opt = opts()
    opt.nms = True
    opt.obj_scale = True
    opt.c="cereal_box"
    camera_ford=np.array([[3648, 0, 2736], [0, 3648, 1824], [0, 0, 1]], dtype=np.float32)
    meta={"width": 5472,"height": 3648, "camera_matrix":camera_ford }
    # start the for loop:
    
    wrong_points=[]
    new_json=[]
    for num, dict in enumerate(data):
        correct_size=dict["size"]
        if dict["points"]==[]:
            print("no points")
        else:
            if dict["image_name"] in os.listdir("/apollo/mle/Datasets/boxes"):
                # 1. load img, points remove first:
                img=plt.imread("/apollo/mle/Datasets/boxes/"+dict["image_name"])
                points=dict["points"][1:]
                
                # invert the points:
                for i, p in enumerate(points):
                    points[i]=[p[0], meta["height"]-p[1]]
                
                # plt.imshow(img)
                
                # real annotation:
                # for p in points:
                #     plt.plot(p[0],p[1], 'ro')
                # plt.show()

                #use pnp:
                try:
                    correct_size=find_correct_size(dict["size"], points, meta, opt)
                
                    # bbox= {'kps': points, "obj_scale": correct_size}
               
                    # projected_points, point_3d_cam, scale, points_ori, bbox=pnp_shell(opt, meta, bbox, points, bbox["obj_scale"], OPENCV_RETURN=False)
                    # for p in bbox["projected_cuboid"]:
                    #     plt.plot(p[0],p[1], 'ro')
                    # plt.savefig("labels_pnp/"+dict["image_name"])
                except:
                    wrong_points.append(dict["image_name"])
                # plt.clf()
            else:
                print("nobox")
        new_json.append({"image_name":dict["image_name"], 
                        "points": dict["points"],
                        "size": correct_size})
        print(num)
    print(wrong_points)
    with open ("data/outf_all/ford/order_train_size_correct.json", 'w') as f:
        json.dump(new_json, f)
    
    return

if __name__=="__main__":
    main()