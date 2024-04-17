import json
import matplotlib.pyplot as plt
import os
import numpy as np
from lib.utils.pnp.cuboid_pnp_shell import pnp_shell
from lib.opts import opts
import sys
from tools.objectron_eval.objectron.dataset.box import Box as Boxcls
from tools.objectron_eval.objectron.dataset.iou import IoU
import csv



def get_query_list():
    """_summary_ reads the 2 csv files and combines the information to generate a list of images that can be used as query
    (Not Parallel to the camera)
    """
    query_list=[]
    # csv filenemae: 
    filename ="../csv/csv.csv"
    with open(filename, "r") as f:
        dict_reader= csv.DictReader(f)
        list_of_dict=list(dict_reader)
    
    for dict in list_of_dict:
        if dict["Query"]=="y":
            query_list.append(dict["Picture ID"].split("-")[-1])
    return query_list
    
    

def get_gt_points(dict, meta, opt):
    if opt.green_background:
        invert_points=[]
        for i, p in enumerate(dict["points"]):
            invert_points.append([p[0], meta["height"]-p[1]])
    else:
        invert_points=dict["points"]
        
        
    size=np.array(dict["size"])
   
   # normalize y
    try:  
        bbox= {'kps': invert_points[1:], "obj_scale": size/size[1]}
        projected_points, point_3d_cam, scale, points_ori, bbox=pnp_shell(opt, meta, bbox,invert_points, size/size[1], OPENCV_RETURN=False)
    except:
        print("GT wrong point order")
        return [0]
    return np.array(bbox["kps_3d_cam"])


    
def get_annotations(opt):
    if not opt.green_background:    
        with open("../data/random_background/train/train_backgrounds.json") as f:
                train= json.load(f)
            
         
        with open("../data/random_background/val/val_backgrounds.json") as f:
                val= json.load(f)
                        
        # use green background other jsons and other data 
    else: 
        with open("../data/green_boxes/train.json") as f:
                train= json.load(f)
                

        with open("../data/green_boxes/val.json") as f:
                val= json.load(f)    

   
    return train, val

    
    
def evaluate_img(root_img, root_json_detect, img_id, camera_ford, opt, verbose=False):
    img_list=os.listdir(root_img)
    img_name=img_id+".JPG"
    if img_name not in img_list:
        print("img not found in folder", root_img)
        return 1,0
    train, val=get_annotations(opt)
    annotations=train+val
    annotation=None
    for dict in annotations:
        if img_id in dict["image_name"]:
            annotation=dict
            break
    if annotation is None:
        print("no annotation for given file")
        return 1,0
    img=plt.imread(root_img+img_name)
    if verbose:
        plt.imshow(img)
        plt.show()
    # check if the img has a debug file:
    detections=os.listdir(root_json_detect)
    if img_id+ ".json" not in detections:
        print("no detection json found")
        return 1,0

    with open(root_json_detect+img_id+".json", "r") as f:
        detection=json.load(f)
    
    if len(detection["objects"])==0:
        print("no detection")
        return 1, 0
    detection_points=np.array(detection["objects"][0]["kps_3d_cam"])
    
    
    meta={"width": img.shape[1],"height": img.shape[0], "camera_matrix":camera_ford }
    # get 3D GT:
    gt_points=get_gt_points(annotation, meta, opt)
    if len(gt_points)==1:
        print("wrong annotation point order")
        return 1,0
 
    # make box objects and determine IoU:
    gt_box=Boxcls(gt_points)
    detect_box=Boxcls(detection_points)
    iou=IoU(detect_box,gt_box)
    # print("iou old= ", iou.iou())
    # print(detect_box.vertices[0], gt_box.vertices[0])
    
    # shift box so that it falls on middelpoint detection:
    trans=gt_box.vertices[0]-detect_box.vertices[0]
    # print(detect_box.vertices[0]+trans)
    translated=[]
    for i in range(len(detect_box.vertices)):
        translated.append(detect_box.vertices[i]+trans)
    
    translated=np.array(translated)
    detect_box=Boxcls(translated)
    print("volume detected", detect_box.volume)
    print("volume GT",gt_box.volume)
    
    print("scale detected", detect_box.scale)
    print("scale GT",gt_box.scale)
    
    iou=IoU(detect_box,gt_box)
    result=iou.iou()
    print(result)
    # img=plt.imread(root_json_detect+img_id+"_out_kps_processed_pred.png")
    return result, img


def main(opt, camera_ford,root_img, root_json_detect):
    img_id="IMG_0587"
    args = sys.argv[1:]

    if len(args)==0 or len(args)> 1:
        print("call function with one argument, e.g. '0587'")
        print("using default image= IMG_0587")
    else:
        img_id="IMG_"+args[0]
    
    iou, img=evaluate_img(root_img, root_json_detect, img_id, camera_ford, opt)
    print(img)
    if iou==1:
        print("error see prints")
    
    else:
        iou*=100
        print(f"the intersection of union was {round(iou)}, see the plot for the result")
        plt.imshow(img)
        plt.show(block=False)
        input("hit[enter] to end.")
    plt.close('all')
        


def find_iou(root_json_detect, file):
    with open(root_json_detect+file, "r") as f:
        data=json.load(f)

    if len(data["objects"])==0:
        print("no object found")
        return 0
        
    
    return data["objects"][0]["IOU"]




def get_statistics_query(root_json_detect,opt,query_list, verbose=False):
    train, val=get_annotations(opt)
    print("train length: ",len(train)) 
    print("validation length ",len(val)) 
    json_files=os.listdir(root_json_detect)
    print(len(json_files))
    total_train=0
    total_val=0
    missed_train=0
    missed_val=0
    correct_train=0
    correct_val=0
    correct_train_50=0
    correct_val_50=0
    gt_fail_train=0
    gt_fail_val=0
    fail=[]
    
    for file in json_files:
        name=file.split(".")[0]
        found=False
        Query=False
        for q in query_list:
            if q in name:
                Query=True
                break

        if Query:
            for dict in train:
                if name in dict["image_name"]:
                    found="train"
                    break
                    
            if found==False:
                for dict in val:
                    if name in dict["image_name"]:
                        found="val"
                        break
            # find iou and add statistics      
            iou=find_iou(root_json_detect, file)       
                    
            if found=="val":
                total_val+=1
                if iou==None:
                    gt_fail_val+=1
                elif iou==0:
                    missed_val+=1
                else:
                    correct_val+=iou
                    if iou>=0.5:
                        correct_val_50+=1
                    else:
                        fail.append(file)
                        
                        
            elif found=="train":
                total_train+=1
                if iou==None:
                    gt_fail_train+=1
                elif iou==0:
                    missed_train+=1
                else:
                    correct_train+=iou
                    if iou>=0.5:
                        correct_train_50+=1
                        
                    else:
                        fail.append(file)
            else:
                print("json not in a file")
        
    print()
    print("TRAIN: ")
    print(f"The total number of samples is {total_train}")
    print(f"The total number of miss detections =  {missed_train}")
    print(f"The total number of miss GT =  {gt_fail_train}")
    print(f"The train average iou is {correct_train/(total_train-missed_train-gt_fail_train)}")
    print(f"The train 50% iou {correct_train_50/(total_train-missed_train-gt_fail_train)}")
    print()
    print("VALIDATION: ")
    print(f"The total number of samples is {total_val}")
    print(f"The total number of miss detections =  {missed_val}")
    print(f"The total number of miss GT =  {gt_fail_val}")
    print(f"The validation average iou is {correct_val/(total_val-missed_val-gt_fail_val)}")
    print(f"The val 50% iou {correct_val_50/(total_val-missed_val-gt_fail_val)}")
    
    print("failed: ", fail)

     
def get_statistics(root_json_detect,opt, verbose=False):
    train, val=get_annotations(opt)
    print("train length: ",len(train)) 
    print("validation length ",len(val)) 
    json_files=os.listdir(root_json_detect)
    print(len(json_files))
    total_train=0
    total_val=0
    missed_train=0
    missed_val=0
    correct_train=0
    correct_val=0
    correct_train_50=0
    correct_val_50=0
    gt_fail_train=0
    gt_fail_val=0
    fail=[]
    
    for file in json_files:
        name=file.split(".")[0]
        found=False
        
        for dict in train:
            if name in dict["image_name"]:
              
                found="train"
                break
                   
        if found==False:
            for dict in val:
                if name in dict["image_name"]:
                   
                    found="val"
                    break
        # find iou and add statistics      
        iou=find_iou(root_json_detect, file)       
                
        if found=="val":
            total_val+=1
            if iou==None:
                gt_fail_val+=1
            elif iou==0:
                missed_val+=1
            else:
                correct_val+=iou
                if iou>=0.5:
                    correct_val_50+=1
                else:
                    fail.append(file)
                    
                    
        elif found=="train":
            total_train+=1
            if iou==None:
                gt_fail_train+=1
            elif iou==0:
                missed_train+=1
            else:
                correct_train+=iou
                if iou>=0.5:
                    correct_train_50+=1
                    
                else:
                    fail.append(file)
        else:
            print("json not in a file")
        
    print()
    print("TRAIN: ")
    print(f"The total number of samples is {total_train}")
    print(f"The total number of miss detections =  {missed_train}")
    print(f"The total number of miss GT =  {gt_fail_train}")
    print(f"The train average iou is {correct_train/(total_train-missed_train-gt_fail_train)}")
    print(f"The train 50% iou {correct_train_50/(total_train-missed_train-gt_fail_train)}")
    print()
    print("VALIDATION: ")
    print(f"The total number of samples is {total_val}")
    print(f"The total number of miss detections =  {missed_val}")
    print(f"The total number of miss GT =  {gt_fail_val}")
    print(f"The validation average iou is {correct_val/(total_val-missed_val-gt_fail_val)}")
    print(f"The val 50% iou {correct_val_50/(total_val-missed_val-gt_fail_val)}")
    
    print("failed: ", fail)
            
           
if __name__=="__main__":
    query_list=get_query_list()
    # OPT:
    opt = opts().parser.parse_args()
    print("Green Background? ", opt.green_background)
    opt.nms = True
    opt.obj_scale = True
    opt.c="cereal_box"
    
    # opt.green_background=True
    
    if opt.green_background:
        root_img="../data/green_boxes/"
        root_json_detect="../demo/green_background/"

        
    else: 
        root_img="../data/random_background"
        root_json_detect="../demo/random_background/"
    
    # meta: 
    camera_ford=np.array([[3648, 0, 2736], [0, 3648, 1824], [0, 0, 1]], dtype=np.float32)
    # main(opt, camera_ford, root_img, root_json_detect)
    get_statistics_query(root_json_detect, opt, query_list)
    # get_statistics(root_json_detect, opt)


    
        
    
