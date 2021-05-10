All tasks have a seperate folder with a corresponding name.
Inside should be a coco file, which is a modification of the original coco trainig file, and a Excersice\_training file which trains the model. NOTE, if you run this it will overwrite the model!

validate\_only files calls the validation.py in utils and saves a random picture with a caption. Should be a folder called photos with some of theese already in there. 

to run on the ml6/ml7 cluster, first load 
module load PyTorch-bundle/1.7.0 
module load PyTorch-bundle/1.7.0 Java 

CUDA\_VISIBLE\_DEVICES=x TMP=./tmp python3 \<filename\>

where x is the gpu

validate only files generate a plot of an image with a corresponding caption and saves them 
