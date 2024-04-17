# Main_CenterPose.py How to use:
This is the python file used to train the network.
As a default we start from the weights trained for cereal boxes. 
If you don't want this and you want a fresh start make: 
opt.load_model=""



There are a few parameters that can be changed. 
- Batchsize 
- learningrate
- startpoint (weightfile for the given architecture from were to start the training) 
- epochs
- not_random_crop
- rotation
  