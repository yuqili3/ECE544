# ECE544
Dataset prepare: 
1. dataset: used STL-10, for each training and testing image, generate noisy images with noise level = sigma=0.05,0.1,0.2

Pretraining denoiser:\\
2. denoising CNN model: DnCNN\\
E.g. python denoising.py --lr=1e-3 --epoch=200 --sigma=0.05 (--resume)
This will train the denoiser DnCNN with learning rate 1e-3 for 200 epochs, the input noisy image has noise level=0.05. If training from a saved model, use --resume. This would generate the dataset mentioned above, if it doesn't exist in the specifed data directory, so no need to run dataset.py

Traditional pipelin: denoising + classfication using a pretrained Resnet-18 model
3. python classifier.py --epoch=30 --sigma=0.05 --netName=BM3D/CNN64 --dataset-type=
 
In lots of py file, we hard-coded the data directory (training was done on different servers).
