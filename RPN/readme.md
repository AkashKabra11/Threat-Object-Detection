============================================================================================================================
						TRAINING AND TESTING RPN
============================================================================================================================

Move to RPN folder now. 
Folder should have: 
1. train_rpn.py
2. test_end2end.py
3. test_images folder with images to be tested
4. keras_frcnn folder with multiple py scripts
5. train_images folder made post GANs code
6. annotate.txt file made post GANs code
7. resnet50_weights_tf_dim_ordering_tf_kernels.h5 downloaded as per he instructions given below. 

PRE-CONDITION: 
1. You need to provide the testing images in the "test_images" folder now. 
Note that these are NOT the GAN images that we had obtained earlier, they should be original and non-preprocessed real-time images. 
2. You need to download "resnet50_weights_tf_dim_ordering_tf_kernels.h5" pretrained weights from https://github.com/fchollet/deep-learning-models/releases/tag/v0.1 and put in in the same RPN directory
3. Check again that you have "annotate.txt" and "train_images" in this folder. 

********************************************
FOR TRAINING RPN MODEL  
*******************************************


1. Run the script ** :
python train_rpn_precise.py -o simple -p annotate.txt 

POST-CONDITION : 
- You should have multiple models saved in the newly made directory : ./models/rpn. These are intermediate models saved after every 4 epochs. 
- You should have "model_frcnn_rpn.hdf5" and "config.pickle" file made in the same directory

** IN CASE OF THIS TYPE OF ERROR: 
"ValueError: Shape must be rank 1 but is rank 0 for 'bn_conv1/Reshape_4' (op: 'Reshape') with input shapes: [1,1,1,64], []."
We found that this error is some backend issue in keras 2.2.4
You need to move to 'lib/python3.6/site-packages/keras/backend/tensorflow_backend.py' and update the batch_normalization method in line 1876
This is the link for the same. 
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / sqrt(var + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        axis: Integer, the axis that should be normalized.
            (typically the features axis).
        epsilon: Fuzz factor.

    # Returns
        A tensor.
    """
    if ndim(x) == 4:
        # The CPU implementation of FusedBatchNorm only support NHWC
        if axis == 1 or axis == -3:
            tf_data_format = 'NCHW'
        elif axis == 3 or axis == -1:
            tf_data_format = 'NHWC'
        else:
            tf_data_format = None

        if tf_data_format == 'NHWC' or tf_data_format == 'NCHW' and _has_nchw_support():
            # The mean / var / beta / gamma may be processed by broadcast
            # so it may have extra axes with 1, it is not needed and should be removed
            if ndim(mean) > 1:
                mean = tf.reshape(mean, [-1])    # here was the issue [-1] instead of (-1)
            if ndim(var) > 1:
                var = tf.reshape(var, [-1])     # here was the issue [-1] instead of (-1)
            if beta is None:
                beta = zeros_like(mean)
            elif ndim(beta) > 1:
                beta = tf.reshape(beta, [-1])   # here was the issue [-1] instead of (-1)
            if gamma is None:
                gamma = ones_like(mean)
            elif ndim(gamma) > 1:
                gamma = tf.reshape(gamma, [-1])   # here was the issue [-1] instead of (-1)
            y, _, _ = tf.nn.fused_batch_norm(
                x,
                gamma,
                beta,
                epsilon=epsilon,
                mean=mean,
                variance=var,
                data_format=tf_data_format,
                is_training=False
            )
            return y
            
******************************************
FOR TESTING RPN LAYER:
******************************************
PRE-CONDITION: 
Check if there is no folder named: Final_Results. 
If it does, delete it. 

Run the Script: 
python test_end2end.py -p test_images

POST-CONDITION:
- "time.txt" contains execution time of each file. 
- "Final_Results" folder contains the final results post classification and Localization. 

END OF README. 


















