import cv2
import json
data_dir="D:/Ford/data/data/"


font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 5

# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 5
def main():
    bad_list=[]    
    with open("data/outf_all/ford/order_val.json", 'r') as f:
        data=json.load(f)
    
    print(data[0]) 
    h,w,c=cv2.imread(data_dir+data[0]["image_name"]).shape

    for num,  dict in enumerate(data):
        if dict["points"]==[]:
            continue
        print("index= ", num)
        
        cv2.namedWindow(dict["image_name"], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(dict["image_name"], 800, 500)
        img=cv2.imread(data_dir+dict["image_name"])
        
        for i, p in enumerate(dict["points"][1:]):
            img=cv2.putText(img, str(i+1), (p[0],h-p[1]), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow(dict["image_name"], img)
        k = cv2.waitKey()
        if k==27:    # Esc key to stop
            break
        elif k==103:  # correct  g
            print('right')
        elif k==98:  # not_correct b
            print('wrong')
            bad_list.append(num)
        else:
            print(k) # else print its value
        cv2.destroyAllWindows()
    print(bad_list)
    with open("bad_list.txt", 'w') as f:
        for item in bad_list:
            f.writelines(str(item)+ "\n")
    return
        
    
    
    
    
if __name__=='__main__':
    main() 

