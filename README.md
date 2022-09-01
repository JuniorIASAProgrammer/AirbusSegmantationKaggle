## Airbus Image Segmentation Kaggle Competition

Information about ships location is encoded in csv file in two value format for each : starting point and run length.  
As we want to use U-net architecture, the output of NN must be represented as a mask with the same shape as input image.  

### Solution :  
1. Open csv file -> delete all pictures that don't have any ships to reduce disbalance -> group ships from the same image.  
2. Create function that convert labels (start point + run length) into full-size mask (768*768 two-values mask).
3. As dataset is large and doesn't fit in memory, create data pipeline using tf.data.Dataset to preprocess images and create masks applying functions in map().  
4. Initialize dice loss function and metric for performance monitoring.  
5. Create upsampling and downsampling blocks for U-net.
6. Initialize U-net model.
7. Compile model and launch learning process.
