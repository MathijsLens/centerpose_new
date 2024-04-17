import os
import cv2



dir="../images/ford/"
for i in os.listdir(dir):
    print(i)
    image=cv2.imread(dir+i)
    image=cv2.resize(image, (512,512))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(dir+i,image)