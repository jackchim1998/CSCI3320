import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import imageio
from sklearn.decomposition import PCA
from PIL import Image

def load_data(digits = [0], num = 200):
    '''
    Loads all of the images into a data-array.

    The training data has 5000 images per digit,
    but loading that many images from the disk may take a while.  So, you can
    just use a subset of them, say 200 for training (otherwise it will take a
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.

    '''
    totalsize = 0
    for digit in digits:
        try:
            totalsize += min([len(next(os.walk('train%d' % digit))[2]), num])
        except StopIteration:
            pass
    print('We will load %d images' % totalsize)
    X = np.zeros((totalsize, 784), dtype = np.uint8)   #784=28*28
    for index in range(0, len(digits)):
        digit = digits[index]
        print('\nReading images of digit %d' % digit)
        for i in range(num):
            pth = os.path.join('train%d' % digit,'%05d.pgm' % i)
            image = imageio.imread(pth).reshape((1, 784))
            X[i + index * num, :] = image
        print('\n')
    return X

def plot_mean_image(X, digits = [0]):
    ''' example on presenting vector as an image
    '''
    plt.close('all')
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.imshow(np.reshape(meanrow,(28,28)))
    plt.title('Mean image of digit ' + str(digits))
    plt.gray(), plt.xticks(()), plt.yticks(()), plt.show()

def main():
    digits = [0, 1, 2]
    # load handwritten images of digit 0, 1, 2 into a matrix X
    # for each digit, we just use 500 images
    # each row of matrix X represents an image
    X = load_data(digits, 500)
    # plot the mean image of these images!
    # you will learn how to represent a row vector as an image in this function
    plot_mean_image(X, digits)

    ####################################################################
    # plot the eigen images, eigenvalue v.s. the order of eigenvalue, POV
    # v.s. the order of eigenvalue
    # you need to
    #   1. do the PCA on matrix X;
    #
    #   2. plot the eigenimages (reshape the vector to 28*28 matrix then use
    #   the function ``imshow'' in pyplot), save the images of eigenvectors
    #   which correspond to largest 9 eigenvalues. Save them in a single file
    #   ``eigenimages.jpg''.
    #
    #   3. plot the POV (the Portion of variance explained v.s. the number of
    #   components we retain), save the figure in file ``digit_pov.jpg''
    #
    #   4. report how many dimensions are need to preserve 0.95 POV, describe
    #   your answers and your undestanding of the results in the plain text
    #   file ``description.txt''
    #
    #   5. remember to submit file ``eigenimages.jpg'', ``digit_pov.jpg'',
    #   ``description.txt'' and ``ex2.py''.
    # YOUR CODE HERE!
    pca_sk=PCA()
    pca_sk.fit(X)
    fig = plt.gcf()
    fig.suptitle("The Largest 9 eigenimages", fontsize=14)
    for i in range(9):
        plt.subplot(330+i+1)
        plt.imshow(np.reshape(pca_sk.components_[i,:],(28,28)))
        plt.gray(), plt.xticks(()), plt.yticks(()), plt.xlabel(str(i))
    plt.savefig('eigens_fig.png')
    im = Image.open('eigens_fig.png')
    rgb_im = im.convert('RGB')
    rgb_im.save('eigens_fig.jpg')
    plt.show()
    sum=0
    POV_per=0.95
    POV_per_index=None
    POV=[]
    for eig_val in pca_sk.explained_variance_:
        sum += eig_val
        POV.append(sum)
    for i in range(len(POV)):
        POV[i] = POV[i]/sum
        if POV_per_index is None and POV[i]>POV_per:
            POV_per_index=i
    print("we need to preserve at least "+str(POV_per_index+1)+" dimensions to achieve POV larger than "+str(POV_per))
    plt.xlabel('index of eigenvalue')
    plt.ylabel('POV')
    plt.title('eigenvalues graph')
    plt.plot(POV)
    plt.savefig('digit_pov.png')
    im = Image.open('digit_pov.png')
    rgb_im = im.convert('RGB')
    rgb_im.save('digit_pov.jpg')
    plt.show()
    ####################################################################


if __name__ == '__main__':
    main()
