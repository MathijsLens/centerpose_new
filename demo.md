# Demo.py How to use:

The main use of the demo.py file is to test a trained network. 
We can do this for the 2 version of the network. (green background and random background)
To run the file we use 
    python demo.py
The Default weights are the ones for the random background network. 

There are multiple ways to test the network. The input can be specifyed with the --demo atribute:

1. Test with a single file:  --demo /path/to/file
2. Use the computer webcam: --demo webcam
3. Test on all images in a directory: --demo /path/to/directory
4. Test with a video: --demo /path/to/video


There are also multiple ways to view the output. the --debug attribute specifies each outputmode. 

1. Show the result in a cv2 window: --demo 0  (Default)
2. Show the results and intermediate results (bounding box, keypoints object center, result, post-processing result): --demo 2
3. Save the results and intermediate results in the /demo directory  --demo 4
4. Generate jsons files for evaluation of the network (calculating the 3D IOU) --demo 7


