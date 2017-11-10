import bcolz, itertools, json, math, numpy as np, pandas as pd, os.path, random, scipy, sys, threading, time, PIL

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.optimizers import RMSprop, Adam
from keras.utils.np_utils import to_categorical
from importlib import reload
from matplotlib import pyplot as plt
from PIL import Image
from resnet50 import Resnet50
from sklearn.neighbors import NearestNeighbors, LSHForest
from tqdm import tqdm
from vgg16bn import Vgg16BN

def save_array(fname, arr): 
    c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
    
def load_array(fname): 
    return bcolz.open(fname)[:]

def vgg(output_size, dropout=0.5):
    model = Vgg16BN((224,224), True, dropout=dropout).model #NB: batch normalization added
    model.pop()
    model.add(Dense(output_size, activation='softmax'))
    return model

def resnet(output_size, dropout=0.5):
    resnet = Resnet50((224,224), False)
    x = resnet.model.layers[-1].output
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)   
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(output_size, activation='softmax')(x)
    return Model(resnet.img_input, x)
 
def last_conv_idx(model):
    return [index for index,layer in enumerate(model.layers) if type(layer) is Conv2D][-1]

def conv_model(model):
    return Sequential(model.layers[:last_conv_idx(model) + 1])

def resnet_conv_model(model):
    return Sequential(model.layers[:-3])

def vgg_fc_model(model, output_size, dropout=0.5, output_activation='softmax'):
    fc_model = Sequential([
            MaxPooling2D(input_shape=conv_model(model).output_shape[1:]),
            Flatten(),
            Dense(4096, activation='relu'),
            BatchNormalization(input_shape=(4096,)),
            Dropout(dropout),
            Dense(4096, activation='relu'),
            BatchNormalization(input_shape=(4096,)),
            Dropout(dropout),
            Dense(output_size, activation=output_activation)
            ])
    
    fc_layers = model.layers[last_conv_idx(model) + 1:]
    for l1,l2 in zip(fc_model.layers, fc_layers): 
        l1.set_weights(l2.get_weights())
    
    return fc_model

def resnet_fc_model(model, output_size, dropout=0.5):
    resnet = Resnet50((224,224), False)
    conv_input = Input(resnet.model.layers[-1].output_shape[1:]) 
    x = conv_input
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(output_size, activation='softmax')(x)
    fc_model = Model(conv_input, x)
    
    for i in range(6): fc_model.layers[i+1].set_weights(model.layers[-6 + i].get_weights())
    
    return fc_model

def fits_w_batches(model, params, trn_batches, val_batches, batch_size=64, model_filename=None):
    for param in params:
        fit_with_batches(model, param[0], param[1], trn_batches, val_batches)
        if model_filename: model.save_weights(model_filename)

def fit_with_batches(model, optimizer, epochs, trn_batches, val_batches, batch_size=64, loss='categorical_crossentropy'):
    trn_batches.reset()
    val_batches.reset()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = model.fit_generator(trn_batches,  
                        steps_per_epoch=steps(trn_batches, batch_size),
                        epochs=epochs,
                        validation_data=val_batches,
                        validation_steps=steps(val_batches, batch_size))
    log_action(str(history.history["val_acc"]))
    fit_hist_plot(history)
   
def fits_w_features(model, params, trn_features, trn_labels, val_features, val_labels, batch_size=64, model_filename=None):
    for param in params:
        fit_with_features(model, param[0], param[1], trn_features, trn_labels, val_features, val_labels, batch_size=64)
        if model_filename: model.save_weights(model_filename)
    
def fit_with_features(model, optimizer, epochs, trn_features, trn_labels, val_features, val_labels, batch_size=64, loss='categorical_crossentropy'):
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = model.fit(trn_features, 
              trn_labels, 
              epochs=epochs, 
              validation_data=(val_features, val_labels), 
              batch_size=batch_size)
    log_action(str(history.history["val_acc"]))
    fit_hist_plot(history)

def fit_hist_plot(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['trn', 'val'], loc='upper left')
    plt.show()
    
def log_action(str):
    file = open('log_file.txt', 'a+')
    message = time.strftime("%Y-%m-%d %H:%M") + ' -> ' + str + '\n'
    file.write(message)
    file.close()
    print(message)
    
def clear_logs():
    file = open('log_file.txt', 'w+').close()

def ensemble_predict(model, model_path, batches, size=5):
    for layer in model.layers: layer.trainable=True
    ens_preds=[]
    
    for i in range(size): 
        batches.reset()
        model.load_weights(base_dir + model_path + str(i) + '.h5')
        preds = model.predict_generator(batches, steps(batches, 64), verbose=1)
        ens_preds.append(preds)
        
    return np.stack(ens_preds).mean(axis=0)

def conv_features(model, batches):
    batches.reset()
    return conv_model(model).predict_generator(batches, steps(batches, 64), verbose=1)

def steps(batches, batch_size):
    return int(np.ceil(batches.samples/batch_size))

def ceil(x):
    return int(math.ceil(x))

def plot_paths(img_paths, count=None, columns=4):
    count = count if count else len(img_paths)
    for i in range(ceil(count/columns)):
        start = i*columns
        end = min(count,(i+1)*columns)
        imgs = [image.load_img(p) for p in img_paths[start:end]]
        plot_row(imgs, titles=img_paths[start:end])

def plot(imgs, titles=None, count=None, columns=4):
    count = count if count else len(imgs)
    for i in range(ceil(count/columns)):
        start = i*columns
        end = min(count,(i+1)*columns)
        plot_row(imgs[start:end], titles=titles[start:end] if titles else None)
            
def plot_row(imgs, titles=None):
    if type(imgs[0]) is np.ndarray:
        imgs = np.array(imgs).astype(np.uint8)
        if (imgs.shape[-1] != 3): imgs = imgs.transpose((0,2,3,1))
    
    f = plt.figure(figsize=(24,12))
    for i in range(len(imgs)):
        sp = f.add_subplot(1, len(imgs), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(imgs[i])
        
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def neighbor_averaged_preds(nearest_neighbors, preds, neighbors_weight=0.9):
    weights=[1-neighbors_weight, neighbors_weight]

    weighted_preds=[]
    for i in range(len(preds)):
        nearest_tuples = nearest_neighbors[i]
        nearest_idxs = np.array([int(tup[0]) for tup in nearest_tuples])
        mean_nearest_preds = np.array([preds[idx] for idx in nearest_idxs]).mean(axis=0)
        weighted_preds.append(np.average(np.vstack((preds[i],mean_nearest_preds)), axis=0, weights=weights))
        
    return np.array(weighted_preds)

def to_plot(img) :
    return np.rollaxis(img, 0, 3).astype(np.uint8)

def to_keras(img):
    return np.rollaxis(img, 2).astype(np.uint8)

def from_keras(img):
    return np.moveaxis(img, 0, 2).astype(np.uint8)

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[0], bb[1]), bb[2], bb[3], color=color, fill=False, lw=3)

def show_bb(img, bb):
    plt.imshow(to_plot(img))
    plt.gca().add_patch(create_rect(bb))
    
def show_bb_pred(img, bb, bb_pred):
    plt.figure(figsize=(6,6))
    plt.imshow(to_plot(img))
    ax=plt.gca()
    ax.add_patch(create_rect(bb_pred, 'yellow'))
    ax.add_patch(create_rect(bb))