# Threat-Object-Detection-on-Airport-baggage-dataset
By training Region Proposal Layer and classification layer individually, this repository aims to detect Threat Objects(Knife/Scissor) on Airport Baggage Dataset

## How to use this: 

1. First train the classifier. For this move to the [readme of classifier](Classifier/readme.md) and insert data in the required format.<br />
2. Then use your GAN Images and augment them using the instructions given in readme of GAN Annotation folder. <br />
3. Then finally, move to the RPN folder and run testing and training scripts as per the instructions given in it's [readme](RPN/readme.md).<br /> 

Your final detections are ready in RPN/Final_Results folder!
