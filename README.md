# ECE544
1. dataset: used CIFAR-10, for each training and testing image, generate num_copy=3 noisy images with noise level = sigma=0.05
2. denoising autoencoder model: still modifying, for now encoder and decoder MLP.
3. run denoising: python denoising.py --lr=0.01 --epoch=100 --copy=3 --sigma=0.05 --resume
  
