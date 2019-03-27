# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:05:25 2019

@author: Admin
"""
import numpy as np
if __name__ == "__main__":
#    from __future__ import absolute_import, division, print_function  
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from keras.models import load_model

    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    
    
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    
    train_labels = train_labels.reshape(train_labels.shape[0], 1).astype('float32')
    test_labels = test_labels.reshape(test_labels.shape[0], 1).astype('float32')
    
    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.
    
    print(train_images.shape)
    print(train_labels.shape)
    def testplot(im, label):
        first_image = np.array(im, dtype='float32')
    #    first_image = np.expand_dims(first_image, axis = 0)
        pixels = first_image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()
        print('Label: {}'.format(int(label)))
        
    testplot(train_images[1], train_labels[1])



#%%
def noisybin(corr, image, white = True, corruptioncount = False):
    '''
    Tar slumpvisa coordinater och sätter dem till 1 eller 0 beroende på white = True eller ej.
    Funkar för icke binariserade bilder.
    Ska jag verkligen invertera pixlar?
    Koordinater kan dyka upp fler än en gång -> blir inte riktigt talande för procent.
    '''
    out = np.copy(image)
    num_of_corr = np.ceil(corr * image.size)
    x = np.array([])   
    coords = [np.random.randint(0, i, int(num_of_corr)) for i in image.shape[:-1]]
#    print(coords)
#    unicoords = np.array(np.unique((coords[0],coords[1]), axis = 0))
#    print(unicoords)
#    x = out[coords]
#    y = np.nonzero(x >= .5)
#    x[x < .5] = 1.
#    x[y] = 0.
    if white == True:
        out[coords] = 1.
    else:
        out[coords] = 0.
        
#    out[coords] = x
    
    if corruptioncount == True:
        corrpixels = np.count_nonzero(out - image)
        print('\n{0:.0f}% corruption applied.'.format(corr*100))
        print('{0:.0f} corrupted pixels found.'.format(corrpixels))
        print('{0:.0f} % corruption applied in practice. \n'.format(100*(corrpixels)/784))
    return out

if __name__ == "__main__":
    corr = 0.5
    x_noisy = noisybin(corr, train_images[2], white = 1, corruptioncount = 1)
    
    testplot(train_images[2], train_labels[2])
    testplot(x_noisy, train_labels[2])

#%% 

def binarize(ims):
    """Binarisera kopior"""
    out = np.copy(ims)
    out[out >= .5] = 1.
    out[out < .5] = 0.
    return out


def add_gauss_noise(nf, image):
    "''The standard gaussian noise, works with binarized as well as non binarized noise, nf = scale factor (standard deviation)''"
    out = np.copy(image)  
    noise = np.random.normal(loc=0.0, scale=1., size=image.shape)
    out_noisy = out + nf*noise
    out_noisy = np.clip(out_noisy, 0., 1.)
    return out_noisy

if __name__ == "__main__":
    corr = 1
    x_noisy = add_gauss_noise(corr, train_images[2])
    binarize(x_noisy)    
    testplot(train_images[2], train_labels[2])
    testplot(x_noisy, train_labels[2])
#%%
'''
Gammal, använd ej.
Ska jag verkligen invertera pixlar?
Koordinater kan dyka upp fler än en gång -> blir inte riktigt talande för procent.
'''
#def random_choice_noreplace(m, axis=-1):
#    # m, n are the number of rows, cols of output
#    return np.random.randint(0, 28, size = (m,m))
##unique = True
#corr = 0.5
#def noisybin(corr, image):
#    out = np.copy(image)
#    num_of_corr = np.ceil(corr * image.size)
#    x = np.array([])   
#    coords = random_choice_noreplace(20)#[np.random.randint(0, i, int(num_of_corr)) for i in image.shape[:-1]]
#    print(coords)
##    unicoords = np.array(np.unique((coords[0],coords[1]), axis = 0))
##    print(unicoords)
#    x = out[coords]
#    y = np.nonzero(x >= .5)
#    x[x < .5] = 1.
#    x[y] = 0.
#
#    out[coords] = x
#    print('{0:.0f}% corruption applied.'.format(corr*100))
#    doublets = np.count_nonzero(image - out)
#    print('{0:.0f} doublets found.'.format(doublets))
#    print('{0:.0f} % corruption applied in practice.'.format(100*(num_of_corr-doublets)/image.size))
#    return out

#if __name__ == "__main__":
#    corr = 0.4
#    x_noisy = noisylinesvertical(corr, train_images[2])
#    
#    testplot(train_images[2], train_labels[2])
#    testplot(x_noisy, train_labels[2])


#%%
def noisybinfillervertical(corr, image, white = True, corruptioncount = False):
    '''För att ta bort en del av bilden, vertikal split uppifrån, funkar för ickebinariserat'''
    out = np.copy(image)

    numcorr = int(corr * 28)
#    num_of_corr = np.ceil(corr * image.size)
#    x = np.array([])
#    coords = out[:numcorr]#np.random.randint(0, numcorr, size = (28, 28))
#    coords = [np.random.randint(0, i, int(num_of_corr)) for i in image.shape[:-1]]
#    print(coords)
#    unicoords = np.array(np.unique((coords[0],coords[1]), axis = 0))
#    print(unicoords)
#    x = out[coords]
#    y = np.nonzero(x >= .5)
#    x[x < .5] = 1.
#    x[y] = 0.

#    out[coords] = 1
    if white == True:
        out[:numcorr] = 1
    else:
        out[:numcorr] = 0
    if corruptioncount == True:
        corrpixels = np.count_nonzero(out - image)
        print('\n{0:.0f}% corruption applied.'.format(corr*100))
        print('{0:.0f} corrupted pixels found.'.format(corrpixels))
        print('{0:.0f} % corruption applied in practice. \n'.format(100*(corrpixels)/784))
    return out

if __name__ == "__main__":
    corr = 0.4
    x_noisy = noisybinfillervertical(corr, train_images[2])
    
    testplot(train_images[2], train_labels[2])
    testplot(x_noisy, train_labels[2])

#%%

def noisybinfillerhorizontal(corr, image, white = True, corruptioncount = False):
    '''Täcker halva bilden med vitt horizontellt om white = True, svart annars'''
    out = np.copy(image.T)
    num_of_corr = np.ceil(corr * image[1].size)
    if white == True:
        out[0][:int(num_of_corr)] = 1
    else:
        out[0][:int(num_of_corr)] = 0
    if corruptioncount == True:
        corrpixels = np.count_nonzero(out - image)
        print('\n{0:.0f}% corruption applied.'.format(corr*100))
        print('{0:.0f} corrupted pixels found.'.format(corrpixels))
        print('{0:.0f} % corruption applied in practice. \n'.format(100*(corrpixels)/784))
        
    return out.T
    
if __name__ == "__main__":
    corr = 0.6
    x_noisy = noisybinfillerhorizontal(corr, train_images[2], white = 0)
    
    testplot(train_images[2], train_labels[2])
    testplot(x_noisy, train_labels[2])


#%%


def noisylineshorizontal(corr, image, corruptioncount = False):
    '''Horizontella linjer, utan replacement. Funkar på icke binariserad data '''
    out = np.copy(image)
#    print(image[0].size)
    num_of_corr = np.ceil(corr * image[0].size)
#    x = np.array([])
#    coords = np.random.randint(0, 28, int(num_of_corr))
    coords2 = np.random.choice(28,int(num_of_corr), replace = False)
#    x = out[:][coords2]
#    y = np.nonzero(x >= .5)
#    x[x < .5] = 1.
#    x = 1.
#    x[y] = 0.

    out[:][coords2] = 1
    if corruptioncount == True:
        corrpixels = np.count_nonzero(out - image)
        print('\n{0:.0f}% corruption applied.'.format(corr*100))
        print('{0:.0f} corrupted pixels found.'.format(corrpixels))
        print('{0:.0f} % corruption applied in practice. \n'.format(100*(corrpixels)/784))
    return out

if __name__ == "__main__":
    corr = 0.4
    x_noisy = noisylineshorizontal(corr, train_images[2])
    
    testplot(train_images[2], train_labels[2])
    testplot(x_noisy, train_labels[2])
#x_test_noisy = np.array([noisylineshorizontal(corr, noisy_im) for noisy_im in test_images]).astype('float32')
#testplot(x_test_noisy[4], test_labels[4])
#%%
def noisylinesvertical(corr, image, corruptioncount = False):
    '''Vertikala linjer, utan replacement.. Funkar på icke binariserad data '''
    out = np.copy(image.T)
    num_of_corr = np.ceil(corr * image[1].size)
#    x = np.array([])
    coords = np.random.randint(0, 28, int(num_of_corr))
#    print(coords)
#    x = out[0][coords]
#    y = np.nonzero(x >= .5)
#    x[x < .5] = 1.
#    x[y] = 0.

    out[0][coords] = 1.
    if corruptioncount == True:
        corrpixels = np.count_nonzero(out - image.T)
        print('\n{0:.0f}% corruption applied.'.format(corr*100))
        print('{0:.0f} corrupted pixels found.'.format(corrpixels))
        print('{0:.0f} % corruption applied in practice. \n'.format(100*(corrpixels)/784))
    return out.T

if __name__ == "__main__":
    corr = 0.7
    x_noisy = noisylinesvertical(corr, train_images[2], True)
    
    testplot(train_images[2], train_labels[2])
    testplot(x_noisy, train_labels[2])
#x_train_noisy = np.array([noisybin(corr, noisy_im) for noisy_im in train_images]).astype('float32')
#x_test_noisy = np.array([noisybin(corr, noisy_im) for noisy_im in test_images]).astype('float32')
#
#testplot(x_train_noisy[1], train_labels[1])
#testplot(x_noisy, train_labels[1])