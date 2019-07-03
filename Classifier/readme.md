******************************************
###### TRAINING AND TESTING CLASSIFIER
******************************************
Folder should have: 
1. Data folder
2. Train_top_layer.py
3. Train_bottom_layer.py
4. Classifier_initialization.py
5. Testing.py
6. vgg weights which would be downloaded as per instructions given below. 

**PRE-CONDITION:** 
- Ensure that data has this structure

data<br />
|--train<br />
|   |------knife<br />
|   |------scissor<br />
|--validation<br />
|   |------knife<br />
|   |------scissor<br />
|--test<br />
 
- Put the **224 X 224** images of **knife and scissor** in the given fields. 

-Download the pretrained vgg model 
[here](https://github.com/MinerKasch/applied_deep_learning/blob/master/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5) for train_top

1. Open the file "Classifier_initialization.py"<br />
Update the fields nb_train_samples_knife, nb_train_samples_scis, nb_validation_samples_knife , nb_validation_samples_scis
depending on number of train/validation samples that you have divided. <br /><br />

2. Now, to train the top layer of classifier: <br />
Run the script: <br />
python Train_top_layer.py<br />
**POST-CONDITION**: "tt_result.h5" file should be made in the same directory<br /><br />

3. To train the bottom layer of classifier: <br />
Run the script:<br />
python Train_bottom_layers.py<br />
**POST-CONDITION**: "class_model.h5" & "class_weights.h5" should be made in the same directory. <br /><br />

4. To test the classifier (Optional, not necessary for flow of the model): <br />
Run the script: <br />
python testing.py <br />

END OF README


