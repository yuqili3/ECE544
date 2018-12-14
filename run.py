import numpy as np

netType_list = {'CNN32','CNN64','CNN64_5','CNN128'}
sigma = 0.1
epoch = 30
reg_term = 0.5
#for netType in netType_list:    
#    print('python MTL_denoising_classification.py \
#--netType=%s --epoch=%d --sigma=%.2f --reg_term=%.2f \
#|& tee -a ../logs/denoise_class_log.txt'
#          %(netType,epoch,sigma,reg_term))
#
#
#
#netType = 'CNN64'
#sigma_list = [0.05, 0.2]
#for sigma in sigma_list:
#    print('python MTL_denoising_classification.py \
#--netType=%s --epoch=%d --sigma=%.2f --reg_term=%.2f \
#|& tee -a ../logs/denoise_class_log.txt'
#          %(netType,epoch,sigma,reg_term))
    

epoch=20
netType = 'CNN64'
reg_term_list = [0,0.1,0.2,0.5,0.7,0.9]
sigma_list=[0.05,0.1,0.2]
for sigma in sigma_list:
    for reg_term in reg_term_list:
        print('python MTL_denoising_classification.py \
    --netType=%s --epoch=%d --sigma=%.2f --reg_term=%.2f \
    |& tee -a ./logs/denoise_class_log.txt'
              %(netType,epoch,sigma,reg_term))
    
    
'''
python MTL_denoising_classification.py --netType=CNN32 --epoch=30 --sigma=0.10 --reg_term=0.50 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN128 --epoch=30 --sigma=0.10 --reg_term=0.50 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN64 --epoch=30 --sigma=0.10 --reg_term=0.50 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN64_5 --epoch=30 --sigma=0.10 --reg_term=0.50 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN64 --epoch=30 --sigma=0.05 --reg_term=0.50 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN64 --epoch=30 --sigma=0.20 --reg_term=0.50 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN64 --epoch=30 --sigma=0.10 --reg_term=0.10 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN64 --epoch=30 --sigma=0.10 --reg_term=0.20 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN64 --epoch=30 --sigma=0.10 --reg_term=0.80 |& tee -a ../logs/denoise_class_log.txt
python MTL_denoising_classification.py --netType=CNN64 --epoch=30 --sigma=0.10 --reg_term=0.90 |& tee -a ../logs/denoise_class_log.txt
'''
'''
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.05 --reg_term=0.00     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.05 --reg_term=0.10     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.05 --reg_term=0.20     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.05 --reg_term=0.50     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.05 --reg_term=0.70     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.05 --reg_term=0.90     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.10 --reg_term=0.00     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.10 --reg_term=0.10     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.10 --reg_term=0.20     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.10 --reg_term=0.50     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.10 --reg_term=0.70     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.10 --reg_term=0.90     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.20 --reg_term=0.00     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.20 --reg_term=0.10     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.20 --reg_term=0.20     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.20 --reg_term=0.50     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.20 --reg_term=0.70     |& tee -a ./logs/denoise_class_log.txt
python MTL_denoising_classification.py     --netType=CNN64 --epoch=0 --sigma=0.20 --reg_term=0.90     |& tee -a ./logs/denoise_class_log.txt

'''