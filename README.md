## Airbus Image Segmentation Kaggle Competition

Information about ships location is encoded in csv file in two value format for each : starting point and run length.  
As we want to use U-net architecture, the output of NN must be represented as a mask with the same shape as input image.  

### Solution :  
1. Open csv file, delete all pictures that don't have any ships and group ships from the same image.  
2. Create function that convert labels into full-size mask.
3. Create data pipeline with tf.data.Dataset to preprocess images and masks.  
4. Initialize dice loss function and metric for performance monitoring.  
5. Create upsampling and downsampling blocks for U-net.
6. Initialize and compile U-net model and fit it.