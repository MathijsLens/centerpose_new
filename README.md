# Documentation for the CenterPose network
This is the documentation for the CenterPose network finetuned for Parcels. 
You can find the original repository here: https://github.com/NVlabs/CenterPose 
The original Paper can be found here: https://arxiv.org/abs/2109.06161 


## What was Done: 
The network was trained on 2 different image-sets: 
  1. On the Green background images
  2. boxes on a random background, with random position and random scale.

All the trainings were started from a network trained on the objectron dataset trained for cerealboxes. 
More info about the Objectron dataset can be found here: https://github.com/google-research-datasets/Objectron 



## instalation: 
How to install modifyed CenterPose: 
1. Git clone the repo
2. Download python 3.8 via https://www.python.org/downloads/windows/ 
3. check if the pythonversion is used: python -V  
    the output should be "python 3.8.16"
4. make a new vertial env using:  python3.8 -m venv name_of_the_virtual_env
5. activate the virtualenv: name_of_the_virtual_env\Scripts\activate
6. install the dependencys: pip install torch
7. whenn finished install the rest of the dependencys: pip install -r mini_requirements.txt
8. test if the environement can find your GPU/cuda with: python test_torch/test_torch.py
9. the script should print your GPU name and give no erros


## Data and weightfiles:
For the data and the weight files look in the sharepoint.
In CenterPose/Models -> you can find the trained network files. Put these in the models directory.


you can find the full Train and val datasets for the random background on the sharepoint under the Data/CenterPose_data directory. 


## List of files importent files:
- src/main_CenterPose.py : The main file for training the network
- src/lib/opts.py: The main file containing all the parameters 
- src/lib/datasets/dataset_combined.py: Contains the code for loading the trainingdata. 
- src/demo.py: Code used to test the network and create the jsonfiles for evaluation. 
- construct_dataset.py : used to change the background and randomly put the box in an image
- eval.py : Reads jsons generated with debug.py and gives IOU scores


## Notebooks:
- Change_background: explains how we replaced the green background by a randombackground. 
- point_order: Explains the importance of the pointorder and the size. Also explains how the PNP algorithm works. 

## Test the network: 
- activate the virtual env
- cd to the src directory
- run debug.py -> more info [here](demo.md)


## Train the network: 
- activate the virtual env
- cd to the src directory
- run main_CenterPose.py -> more info [here](main.md)


## the Exploration folder can be ignored (test functions and old notebooks)

