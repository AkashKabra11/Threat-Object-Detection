Folder should have: 
1. frcnn-test-main.py
2. quick-start.py
3. df.pkl
4. data_aug folder with 2 scripts in between
5. images folder with original GAN images. 

This folder is responsible for annnotations of GAN images that we have obtained. 
Note that, we do not claim on the originality of code snippets in this folder, but it is given as an additional helper files to generate augmented images. 

Steps to use it: 
1. Put the original GAN images in the "images" folder in the "GAN images annotation" directory. 

2. PRE-CONDITION: A df.pkl file should be present in the directory
Run this script on terminal: 
python quick-start.py
POST-CONDITION: A df_final.pkl file should be made, "images" folder should be populated with more images. 


3. Run this script on terminal: 
python frcnn-test-main.py
POST-CONDITION: A folder named "test_images" and other files: "all_data.txt", "csv_bboxes" should be made in "GAN images annotation" directory.
A folder named "train_images" and other files: "annotate.txt" should be made in "RPN" directory. 

Now your training data with annotations has been created.  
 
