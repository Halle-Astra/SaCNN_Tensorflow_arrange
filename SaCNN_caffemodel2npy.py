# -*- coding: utf-8 -*-

#此代码用于将caffemodel文件转为npy以便tensorflow运行
import numpy as np
import caffe
import os
import sys

# 指定caffe的编译目录
caffe_root = 'D:\\Medium\\Codes\\Caffe_build\\caffe/'  
model_def = 'model_caffe/deploy.prototxt'
model_weights = 'model_caffe/ShanghaiTech_part_B.caffemodel'
sys.path.insert(0, caffe_root + 'python')

if os.path.isfile(model_weights):
    print('CaffeNet found.')
else:
    print('CaffeNet Not found.')
    
caffe.set_mode_cpu()
net = caffe.Net(model_def,model_weights,caffe.TEST)

key_use = net.params.keys()
SaCNN_model = {}
for layer_name in key_use:
    layer_data_caffe = net.params[layer_name][0].data
    layer_data_caffe = np.swapaxes(layer_data_caffe,0,2)#[n_out, n_in, h,    w]     caffe
    layer_data_caffe = np.swapaxes(layer_data_caffe,1,3)#[kh,    kw,   n_in, n_out] tf
    layer_data_caffe = np.swapaxes(layer_data_caffe,2,3)
    if 'conv_concat' in layer_name:
        layer_data_caffe = np.swapaxes(layer_data_caffe,2,3)
        print(layer_data_caffe.shape)
        SaCNN_model[layer_name] = [layer_data_caffe]
    else:
        SaCNN_model[layer_name] = [layer_data_caffe,net.params[layer_name][1].data]

if not os.path.exists('weights'):
    os.mkdir('weights')
np.save('weights/sacnn_tf',SaCNN_model)
print('Model have been generated.')