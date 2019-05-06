import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
marks = pd.read_csv('/home/papa/ROBT407_Project/airbus-ship-detection/train_ship_segmentations_v2.csv') # Markers for ships
images = os.listdir('/home/papa/ROBT407_Project/airbus-ship-detection/train') # Images for training
os.chdir("/home/papa/ROBT407_Project/airbus-ship-detection/train")
#Encodes mask to matrix
def mask_part(pic):
    back = np.zeros(768**2) #create a numpy array with 0s and size of the image(without channels)
    starts = pic.split()[0::2] #find the starting pixel index
    lens = pic.split()[1::2] #find how many of them in a row come together
    for i in range(len(lens)): 
        back[(int(starts[i])-1):(int(starts[i])-1+int(lens[i]))] = 1 #"paint" the ship loaction
    return np.reshape(back, (768, 768, 1))

def is_empty(key):
    
    #Function that checks if there is a ship in image. 
    
    df = marks[marks['ImageId'] == key].iloc[:,1] #create dataframe of the specific image (key) with the mask encodings
    if len(df) == 1 and type(df.iloc[0]) != str and np.isnan(df.iloc[0]): #if the size is 1, the image has no ships 
        return True
    else:
        return False
    
def masks_all(key):
    #Merges together all the ship markers corresponding to a single image
    
    df = marks[marks['ImageId'] == key].iloc[:,1] #create dataframe of the specific image (key) with the mask encodings
    masks= np.zeros((768,768,1)) #numpy array of 0s with 3 dimentions
    if is_empty(key):
        return masks
    else:
        for i in range(len(df)): #take the mask of each ship
            masks += mask_part(df.iloc[i]) #get 1-s where there are ships (masks) 
        return np.transpose(masks, (1,0,2))
        
def transform(X, Y):
    '''
    Function for augmenting images. Do that to both X and Y (image and mask)
    '''

    x = np.copy(X) #do not affect the original image
    y = np.copy(Y) #do not affect the original mask
    x[:,:,0] = x[:,:,0] + np.random.normal(loc=0.0, scale=0.01, size=(768,768)) # add some blur to each rgb channel
    x[:,:,1] = x[:,:,1] + np.random.normal(loc=0.0, scale=0.01, size=(768,768)) 
    x[:,:,2] = x[:,:,2] + np.random.normal(loc=0.0, scale=0.01, size=(768,768))
    # Adding Gaussian noise on each rgb channel; this way we will NEVER get two completely same images.
    # Note that this transformation is not performed on Y (does not make any sense to do so) 
    #Preprocess
    x[np.where(x<0)] = 0 
    x[np.where(x>1)] = 1
    #Do transformation randomly with 0.5 chance for each
    if np.random.rand()<0.5: #0.5 chances for this transformation to occur (same for two below)
        x = np.swapaxes(x, 0,1) 
        y = np.swapaxes(y, 0,1)
    #Do horizontal and vertical flips
    if np.random.rand()<0.5:
        x = np.flip(x, 0)
        y = np.flip(y, 0)

    if np.random.rand()<0.5:
        x = np.flip(x, 1)
        y = np.flip(y, 1)
    return x, y  
def plot_transformed(file):
    '''
    Plotting five augmentations of one image using trivial method
    (five images created transforming original with function 'transform()')
    '''
    X, Y = plt.imread(file), masks_all(file)
    plt.figure(figsize = (19,8))
    plt.subplot(253, title ='Original Image')
    X, Y = plt.imread(file)/255, masks_all(file)
    plt.imshow(X)
    plt.axis('off')
    plt.subplot(256, title ='Transformed Image')
    plt.imshow(transform(X,Y)[0])
    plt.axis('off')
    plt.subplot(257, title ='Transformed Image')
    plt.imshow(transform(X,Y)[0])
    plt.axis('off')    
    plt.subplot(258, title ='Transformed Image')
    plt.imshow(transform(X,Y)[0])
    plt.axis('off')
    plt.subplot(259, title ='Transformed Image')
    plt.imshow(transform(X,Y)[0])
    plt.axis('off')    
    plt.subplot(2,5,10, title ='Transformed Image')
    plt.imshow(transform(X,Y)[0])
    plt.axis('off')
    plt.suptitle(file,x=0.3, y=0.7, verticalalignment ='top', fontsize = 22)
    plt.show()
plot_transformed('0270d7317.jpg')    
    
def make_batch(files, batch_size):
    '''
    Creates batches of mask along with images 
    '''
    X = np.zeros((batch_size, 768, 768, 3)) # Create numpy array of zeros for the background 
    Y = np.zeros((batch_size, 768, 768, 1)) # Create numpy array of zeros for the mask
    for i in range(batch_size):
        ship = np.random.choice(files)
        X[i] = (io.imread(ship))/255.0 # Normalize the images to be between 0-1
        Y[i]= masks_all(ship) #Mask every ship in the given image
    return X, Y

def Generator(files, batch_size):
    '''
    This function generates batches of images and their masks
    '''
    while True:
        X, Y = make_batch(files, batch_size) # Create the batches and do transformations one by one
        for i in range(batch_size):
            X[i], Y[i] = transform(X[i], Y[i]) #do the random transformations
        yield X, Y
        
# Intersection over as the measure of accuracy
def IoU(y_true, y_pred, tresh=1e-10): 
    Intersection = K.sum(y_true * y_pred, axis=[1,2,3]) #Use bitwise AND operation to find the intersection area
    Union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - Intersection  #Use biwise OR operation similarly
    return K.mean( (Intersection + tresh) / (Union + tresh), axis=0) #add some tresh as noise
# Intersection over Union for Background could be found similarly
def back_IoU(y_true, y_pred):
    return IoU(1-y_true, 1-y_pred)
# Loss function is defined to be 1 - IoU which is intuitive. When IoU approaches one, the loss would be close to 0 as expected
def IoU_loss(in_gt, in_pred):
    return 1 - IoU(in_gt, in_pred)

  
def fit():
    seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])
    
    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
    aug_gen = create_aug_gen(make_image_gen(train_df))
    loss_history = [seg_model.fit_generator(aug_gen,
                                 steps_per_epoch=step_count,
                                 epochs=MAX_TRAIN_EPOCHS,
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list,
                                workers=1 # the generator is not very thread safe
                                           )]
    return loss_history

while True:
    loss_history = fit()
    if np.min([mh.history['val_loss'] for mh in loss_history]) < -0.2:
        break
        
        
# Create the U-Net model's smaller version   
inputs = Input((768, 768, 3))

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)

u5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)
u5 = concatenate([u5, c3])
c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (u5)
c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c2])
c6 = Conv2D(16, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(16, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c1], axis=3)
c7 = Conv2D(8, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(8, (3, 3), activation='relu', padding='same') (c7)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss= IoU_loss, metrics=[IoU, back_IoU])
model.summary()



weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                   patience=1, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=0, min_lr=1e-8)

early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                      patience=20) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]




#results = model.fit_generator(Generator(images,  = 200), steps_per_epoch = 500, epochs = 30)
results = model.fit(Generator(images, batch_size = 8), steps_per_epoch = 14500, epochs = 3)



def show_loss(loss_history):
    epochs = np.concatenate([mh.epoch for mh in loss_history])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')
    
    _ = ax2.plot(epochs, np.concatenate([mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history]), 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('Binary Accuracy (%)')

show_loss(loss_history)


seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')
