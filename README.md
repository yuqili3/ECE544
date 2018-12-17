# ECE544
1. Dataset prepare: 

 dataset: used STL-10, for each training and testing image, generate noisy images with noise level = sigma=0.05,0.1,0.2

2. Pretraining denoiser: denoising CNN model: DnCNN

 Example: python denoising.py --lr=1e-3 --epoch=200 --sigma=0.05 (--resume)

 This will train the denoiser DnCNN with learning rate 1e-3 for 200 epochs. the input noisy image has noise level = 0.05. If training from a saved model, use --resume. This would generate the dataset mentioned above, if it doesn't exist in the specifed data directory, so no need to run dataset.py

3. Traditional pipelin: denoising + classfication using a pretrained Resnet-18 model

 Example: python classifier.py --epoch=30 --sigma=0.05 --netName=BM3D/CNN64 --dataset-type=clean/denoised

 This will train a resnet-18 classification model for 30 epochs. The input images are either clean, or denoised images using BM3D or CNN64 (DnCNN model with 64 filters) when the noise level = 0.05.
 
4. Joint training--fine tuning

Example: python MTL_denoising_classification.py --lr=1e-5 --epoch=30 --sigma=0.05 --reg_term=0.1 --netType=CNN64

This will train a joint denoising and classification model with learning rate 1e-5 for 30 epochs. It will load a pretrained denoiser model and a classifier model of resnet-18. The input noisy image has noise level=0.05, and the denoiser is DnCNN model with 64 filters. The loss function consisting of 0.1* MSE_loss + 0.9* CE_loss.

5. Joint training--random initialization

Example MTL_denoising_classification_random_init.py --lr=1e-5 --epoch=150 --sigma=0.05 --reg_term=0.1 --netType=CNN64

The setup is the same as 4. Except that the parameters in denoiser is randomly initialized. 

6. other notes:

In lots of py file, we hard-coded the data directory (training was done on different servers).
