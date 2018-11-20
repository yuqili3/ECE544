# ECE544
1. dataset: used STL-10, for each training and testing image, generate num_copy=1 noisy images with noise level = sigma=0.05
2. denoising CNN model: still modifying, for now use dncnn 
3. run denoising: python denoising.py --lr=0.01 --epoch=100 --copy=3 --sigma=0.05 --resume
4. denoised dataset: train_data, train_data_denoised, train_labels
  
