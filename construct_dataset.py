import numpy as np
from random import randrange
import os
import copy
import json
import cv2
import matplotlib.pyplot as plt
from fnmatch import fnmatch



def list_backgrounds(root):
    """Lists all the images in all the subdirectorys of the root folder. 

    Args:
        root (string): path to the root folder

    Returns:
        lsit_backgorund: the list of all the paths of the images inside the root folder. 
    """
    list_background=[]
    pattern = "*.jpg"

    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                list_background.append(os.path.join(path, name))
    return list_background

def init(task):
    """
    Reads in the paths of all the background images. 
    If the task is Validation we take a Ford_van as background

    Args:
        task (string): train or val based on the thask at hand

    Returns:
    Backgrounds: (list): list of all the paths of the images containing backgrounds.
    img_size: (list): a list containing the with and height of the original green backgound images 
    """
    
    
    img_size=[5472, 3648]
    # code for random background:   
    back="/apollo/mle/Datasets/Backgrounds/"
    if task=="train":
        backgrounds=list_backgrounds(back)
    else:
        backgrounds=[]
        b=os.listdir("/apollo/mle/Datasets/backgrounds_ford/")
        for back in b:
            backgrounds.append("/apollo/mle/Datasets/backgrounds_ford/"+ back)
    return img_size, backgrounds

def shift_points(h,w, points, cropsize):
    """Shifts the Groundthuth points based on the cutout height and withd so that they 
    correspond whenn we use the full picture. 
            
    Args:
        h (int): The height from were the crop was taken
        w (int): The width from were the crop was taken
        points (list): the old annotation-points 
        cropsize (int): size of the croped tile

    Returns:
        new_points: np_array: The shited points so that the annotation remains correct for this corp
    """
    new_points=[]
    for p in points:
        new_points.append([p[0]+w, p[1]+h])
    return np.array(new_points)

def rescale_points(img_size, cropsize, points):
    """Rescales the points to the imagedimentions of the cropped image. 
     

    Args:
        img_size: the original annotation image size
        cropsize (int): the dimention of the corpsize 
        points (array): The old annotation points the needs to be rescaled
        to the croptout img size

    Returns:
        points: (np.array): rescaled annotation points so they fit on the cropt out img. 
    """
    x_max = img_size[0]
    y_max= img_size[1]
    new_points=[]
    for p in points:
        p_x=int((p[0]/x_max)*cropsize)
        p_y=int((p[1]/y_max)*(cropsize))
        new_points.append([p_x, p_y])
    return np.array(new_points)

def get_background_crop_video(backgrounds, cropsize):
    """Get a random background and get a random crop out of the Background
    Here we will place the box. 

    Args:
        backgrounds (list): List containing all the possible background-imagepaths 
        cropsize (int): the size of random crop out of the image

    Returns:
        background: The randomly selected background
        h: the start height of the crop
        w: the start width of the crop
    """
    # size of the end image
    background_size=[512, 512]
    # select a background:
    x=True
    while x==True:
        # select a random image from all the possible backgrounds
        select=randrange(0, len(backgrounds))
    
        try:
            background=cv2.imread(backgrounds[select])
            back=cv2.resize(background,background_size)
            x=False
        except:
            x=True
    
    h,w, c = back.shape
    # randomly crop a part out off the background
    # in that part the box will be placed
    h1 = randrange(0, h - cropsize)
    w1 = randrange(0, w - cropsize)

    return copy.deepcopy(back), h1, w1 



def make_mask(image, points):
    """Creates a perfect boxmask based on the annotatinos (points) fo the box
    Only keep what is instide of the annotation. 

    Args:
        image (cv2.image): The green background image
        points (list): The annotation points
    
    Returns:
        binary image: a binary mask. 1 were there is box, 0 no box. 
    """
    cimg = np.zeros_like(image)
    # find the outside points
    hull = cv2.convexHull(np.array(points))
    # create binary image
    cv2.drawContours(cimg, [hull], -1, (255,255,255), -1)
    cimg=cimg[:,:,0]
   
    # Creating kernel
    kernel = np.ones((6,6), np.uint8)
    # erode the binary image so the green border around the box is removed
    erode = cv2.erode(cimg, kernel)
    return erode

def change_background2(img, backgrounds, points, img_size):
    """ Main function were we replace the green backgound with a random background, 
    place the box in a random position and change the annotations accordingly

    Args:
        img (cv2_img): the green background img
        backgrounds (list): list containing the paths to all the random backgrounds
        points (list): the annotation points
        img_size : the original img_size for the green background images    

    Returns:
       background: the newly construckted img
       points: The corresponding annotations for this new img (shifted and rescaled) 
    """
    
    # randomly choose a size new size for the box
    cropsize=randrange(100, 200)
    image=cv2.resize(img, [cropsize, cropsize])
    # rescale the points so they fit scale of the corped img
    points=rescale_points(img_size, cropsize, points)
    masked=make_mask(image, points)
    # use mask to get only the box pixels
    result= cv2.bitwise_and(image, image, mask=masked)    
   
    # get the background
    background, h1, w1=get_background_crop_video(backgrounds, cropsize)
    # create the crop
    cropje=background[h1:h1+cropsize,w1:w1+cropsize,:]
    # invert the mask (1 is background, 0 is box)
    masked = cv2.bitwise_and(cropje,cropje, mask=255-masked)
    # creat the cutout with box and background
    cutout= masked+result
    
    # put the cutout back in the background
    background[h1:h1+cropsize,w1:w1+cropsize,:]=cutout
    background=cv2.cvtColor(background, cv2.COLOR_BGR2RGB) 
    # correct the annotations
    points=shift_points(h1, w1, points, cropsize)
    return background, points




def main():
    
    """Generate Training and Validation images with a different background. Randomly place the box in the image and add random zoom. 
        The Green backgrouond is replqced by a random background selected from 2 datasets: 
        - http://web.mit.edu/torralba/www/indoor.html
        - https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset?resource=download
        """
    # task="train"
    task="train"
    new_jsons=[]
    img_size, backgrounds=init(task)
    boxes=os.listdir("/apollo/mle/Datasets/boxes/")

    with open("data/outf_all/ford/order_"+task+"_size_correct.json", "r") as f:
        data=json.load(f)
        
    for intteration in range(0, 51):
        tel=0
        for i, dict in enumerate(data):
    
            name=dict["image_name"]
      
            if name in boxes:
                if dict["points"]==[]:
                    print("no_anno")
                else:
                    
                    points=[]
                    for p in dict["points"]:
                        points.append([p[0], img_size[1]-p[1]])             
                        
                    img=cv2.imread("/apollo/mle/Datasets/boxes/"+ name)
                    
                    
                    output_img, points=change_background2(img, backgrounds,points, img_size )
                    output_img=cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                    
                    
                    # for p in points:
                    #     output_img = cv2.circle(output_img, (p[0],p[1]), radius=2, color=(0, 0, 255), thickness=-1) 
                    cv2.imwrite("/apollo/mle/Datasets/"+task+"/"+ str(intteration)+"_"+name, output_img)
                    # for p in points[1:]:
                    #     plt.plot(p[0],p[1], 'ro')
                    # plt.savefig("/apollo/mle/Datasets/images_background/"+ str(intteration)+"_"+name)
                    # plt.clf()
                    points=points.tolist()
                    
                    # plt.imsave("/apollo/mle/Datasets/images_background/"+ str(intteration)+"_"+name, output_img)
                    new_dict={"image_name":  str(intteration)+"_"+name, "points": points, "size":dict["size"]}
                    new_jsons.append(new_dict)
                    tel+=1        
          
        print(intteration)
        
        # create big file:             
    with open("/apollo/mle/Datasets/"+task+"/"+task+"_backgrounds.json", 'w') as f:
        json.dump(new_jsons, f)

        
            
            
if __name__=="__main__":
    main()