import json
import numpy as np
import cv2
import os

def find_bound(points):
    """finds the boundingbox around a list of point
    input: the 8 cornerpoints of the box
    output:the boundingbox
    """
    x_coordinates, y_coordinates = zip(*points)

    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def read_json(path):
    """returns the content of a Jsonfile
    input: path to the json file
    output: all the content of the jsonfile (list of dict)
    """
    with open(path, "r") as f:
        data=json.load(f)
    print(len(data))
    return data



def main():
    # read in the jsonfiles
    train=read_json("data/random_background/train/train_backgrounds.json")
    val=read_json("data/random_background/val/val_backgrounds.json")
    
    
    # generate trainginglist: 
    for img in os.listdir("data/random_background/train/"):
        print(img)
        for dict in train:
            if dict["image_name"]==img:
                break
            
            
        boundingbox=find_bound(np.array(dict["points"][1:])) # exclude the first point (center of the box)
        img=cv2.imread("data/random_background/train/"+ img)

        color = (255, 0, 0)
        

        thickness = 2
        

        img = cv2.rectangle(img, boundingbox[0], boundingbox[1], color, thickness) 
        cv2.imshow("boudningbox", img)
        keyCode = cv2.waitKey(0)
        
        # stop programma met q
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            return
        
                
        
    

if __name__=="__main__":
    main()