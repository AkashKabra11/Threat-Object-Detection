******************************************
## TRAINING AND TESTING RPN
******************************************

Move to RPN folder now. <br />
Folder should have: 
1. train_rpn.py
2. test_end2end.py
3. test_images folder with images to be tested
4. keras_frcnn folder with multiple py scripts
5. train_images folder made post GANs code
6. annotate.txt file made post GANs code
7. resnet50_weights_tf_dim_ordering_tf_kernels.h5 downloaded as per he instructions given below. 

PRE-CONDITION: <br />
1. You need to provide the testing images in the "test_images" folder now. 
Note that these are NOT the GAN images that we had obtained earlier, they should be original and non-preprocessed real-time images. <br />
2. You need to download "resnet50_weights_tf_dim_ordering_tf_kernels.h5" pretrained weights from this [link](https://github.com/fchollet/deep-learning-models/releases/tag/v0.1)  
and put it in the same RPN directory.<br />


3. Check again that you have "annotate.txt" and "train_images" in this folder. <br />

********************************************
## For TRAINING RPN MODEL  
*******************************************


1. Run the script ** :<br />
python train_rpn_precise.py -o simple -p annotate.txt <br />

POST-CONDITION : <br />
- You should have multiple models saved in the newly made directory : ./models/rpn. These are intermediate models saved after every 4 epochs. 
- You should have "model_frcnn_rpn.hdf5" and "config.pickle" file made in the same directory

*IN CASE OF THIS TYPE OF ERROR*:
"ValueError: Shape must be rank 1 but is rank 0 for 'bn_conv1/Reshape_4' (op: 'Reshape') with input shapes: [1,1,1,64], []."
We found that this error is some backend issue in keras 2.2.4
You need to move to 'lib/python3.6/site-packages/keras/backend'and change tensorflow_backend.py with [this script](https://drive.google.com/file/d/1fXMql4Ln892b4NtO0jp39MPTysICNp59/view?usp=sharing)
<br />
******************************************
## FOR TESTING RPN LAYER:
******************************************
PRE-CONDITION: <br />
Check if there is no folder named: Final_Results. <br />
If it does, delete it. <br /><br />

Run the Script: <br />
python test_end2end.py -p test_images <br /><br />

POST-CONDITION: <br />
- "time.txt" contains execution time of each file. <br />
- "Final_Results" folder contains the final results post classification and Localization. <br /><br />

END OF README. 


















