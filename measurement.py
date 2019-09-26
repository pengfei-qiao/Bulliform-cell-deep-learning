# run models on new images and measure bulliform cell traits

import os
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from skimage.io import imread
from skimage import img_as_bool
import pickle
import numpy as np
import time
import tensorflow as tf

# models to load
models = ['model1.h5','model2.h5','model3.h5','model4.h5','model5.h5']


img_w = 480 # image width
img_h = 480 # image height
img_c = 1 # image channels


# image ids
X_ids = sorted(os.listdir('./resized'))
X_ids = [i for i in X_ids if '.png' in i]


# define metrics
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# define model fitting
def fit_model(input,model):
    model = load_model(model, custom_objects={'dice_coef': dice_coef})
    preds = model.predict(input, verbose=1)
    return preds


# define get bulliform cell stats
# False is black, True is white (bulliform cells)
def get_bc_area(input):
    return float(np.sum(input))/(input.shape[0]*input.shape[1]*input.shape[2])

def get_bc_1d_cols(input, true_threshold=3, false_threshold=5): # input is 1-d array
    counter = 0
    true_counter = 0
    false_counter = false_threshold
    new_input = np.pad(input, (false_threshold+1,), 'constant', constant_values=(0,0)) # to count bulliform cells that start from the margin
    for i in new_input:
        if i:
            true_counter += 1
        else:
            if true_counter >= true_threshold and false_counter >= false_threshold: # number of pixels for thresholding a bulliform cell
                counter += 1
                true_counter = 0
                false_counter = 0
            else:
                true_counter = 0 
                false_counter += 1
    # for i in new_input:
    #     if i:
    #         true_counter += 1
    #     if true_counter >= true_threshold and not i and false_counter >= false_threshold: # number of pixels for thresholding a bulliform cell
    #         counter += 1
    #         true_counter = 0
    #         false_counter = 0
    #     if not i and false_counter < false_threshold:
    #         true_counter = 0 
    #     if true_counter == 0 and not i:
    #         false_counter += 1
    return counter

def get_bc_3d_cols(input, true_threshold=3, false_threshold=5): # input is 3-d array, with last shape equal to 1
    rows = input.shape[0]
    counter = 0
    for row in xrange(rows):
        counter += get_bc_1d_cols(input[row,].ravel(),true_threshold, false_threshold)
    return float(counter)/rows

def get_ave_bc_width(input):
    if get_bc_3d_cols(input) == 0:
        return float(0)
    else:
        return float(np.sum(input))/get_bc_3d_cols(input)

def get_ave_bc_spacing_width(input):
    return float(np.sum(1-input))/(get_bc_3d_cols(input)+1)

# run on all images
for j in range(60):
    cur_time = time.time()
    X = np.array([1 - img_as_bool(imread('./resized/' + i, as_gray = True)[:480,:480].reshape(img_h, img_w, img_c)) for i in X_ids[1000*j:(1000*j+1000)] if '.png' in i])
    X_mean = np.mean(X,axis=0)
    X = X - X_mean
    print X.shape
    preds_all = np.mean(np.array([fit_model(X,i) for i in models]), axis=0)
    preds_all_t = (preds_all > 0.5).astype(np.bool)
    print preds_all_t.shape
    outfile = open('preds_all_%d' %(j+1),'wb')
    pickle.dump(preds_all,outfile)
    outfile.close()
    outfile = open('preds_all_t_%d' %(j+1),'wb')
    pickle.dump(preds_all_t,outfile)
    outfile.close()
    outfile = open('preds_all_label_%d' %(j+1),'wb')
    pickle.dump(X_ids[1000*j:(1000*j+1000)],outfile)
    outfile.close()
    outfile = open('preds_all_label_%d.txt' %(j+1), 'w')
    outfile.write('line\tbc_area\tbc_col_num\tave_bc_width\tave_spacing_width\n')
    for i in range(1000):
        line = X_ids[1000*j:(1000*j+1000)][i]
        preds_line_t = preds_all_t[i]
        outfile.write('%s\t%.6f\t%.2f\t%.6f\t%.6f\n' %(line,get_bc_area(preds_line_t),get_bc_3d_cols(preds_line_t),get_ave_bc_width(preds_line_t),get_ave_bc_spacing_width(preds_line_t)))
    outfile.close()
    print 'Finished %d-th subset, taking %.2f seconds' %((j+1),time.time()-cur_time)

# for the last fewer than 1000 images    
j = 60
cur_time = time.time()
X = np.array([1 - img_as_bool(imread('./resized/' + i, as_gray = True)[:480,:480].reshape(img_h, img_w, img_c)) for i in X_ids[1000*j:(1000*j+1000)] if '.png' in i])
X_mean = np.mean(X,axis=0)
X = X - X_mean
preds_all = np.mean(np.array([fit_model(X,i) for i in models]), axis=0)
preds_all_t = (preds_all > 0.5).astype(np.bool)
print preds_all_t.shape

outfile = open('preds_all_%d' %(j+1),'wb')
pickle.dump(preds_all,outfile)
outfile.close()

outfile = open('preds_all_t_%d' %(j+1),'wb')
pickle.dump(preds_all_t,outfile)
outfile.close()

outfile = open('preds_all_label_%d' %(j+1),'wb')
pickle.dump(X_ids[1000*j:],outfile)
outfile.close()

outfile = open('preds_all_label_%d.txt' %(j+1), 'w')
outfile.write('line\tbc_area\tbc_col_num\tave_bc_width\tave_spacing_width\n')
for i in range(len(X_ids[1000*j:])):
    line = X_ids[1000*j:][i]
    preds_line_t = preds_all_t[i]
    outfile.write('%s\t%.6f\t%.2f\t%.6f\t%.6f\n' %(line,get_bc_area(preds_line_t),get_bc_3d_cols(preds_line_t),get_ave_bc_width(preds_line_t),get_ave_bc_spacing_width(preds_line_t)))
outfile.close()
print 'Finished %d-th subset, taking %.2f seconds' %((j+1),time.time()-cur_time)
