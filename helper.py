import re
import random
import numpy as np
import os.path
# import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

import cv2

from numpy.random import rand
from numpy.random import randn
from numpy.random import randint


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def truncated_normal(n, sigma=1, mu=0):
    x = randn(n)*sigma
    x[x>2*sigma]=2*sigma
    x[x<-2*sigma]=-2*sigma
    return x + mu

def random_shadow(img):
    # This function directly comes from my project: SDC_ND_P3_Behavioral_Cloning

    # The idea for shadow augmentation came from
    # https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

    # Add wedge like shadow
    xb = rand(2)*img.shape[1]
    G = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    mask = np.zeros(img.shape[0:2])==0
    mask[((G[1]-img.shape[1])*img.shape[0] - np.diff(xb)*(G[0]-img.shape[0]))>=0] = 1

    if rand(1)>0.5:
        mask = mask==1
    else:
        mask = mask==0

    brightnessFactor = rand(1)
    for i in range(3):
        img[:,:,i][mask] = img[:,:,i][mask]*(brightnessFactor+0.2*rand(1))*0.5

    # Add shadow to random square
    idx0 = np.sort(randint(0,img.shape[0],2))
    idx1 = np.sort(randint(0,img.shape[1],2))

    img[idx0[0]:idx0[1],idx1[0]:idx1[1],:] = img[idx0[0]:idx0[1],idx1[0]:idx1[1],:]*(rand(1) + 0.2*rand(1,1,3))*0.7
    return img


def augment_image(X, Y, rot=15., trans=5.):
    # This function is a re-write of the same function from my project: SDC_ND_P3_Behavioral_Cloning


    # ::: Ignore translate, just rotate

    # X translation direction
    trans_option = np.random.randint(0,3,1) # (left, center, right)
    trans_offset = truncated_normal(1,3.,(trans_option-1.)*trans) # in the range of [-6-trans,6+trans]

    # Rotation direction
    # rot_option = np.random.randint(0,3,1) # (cw, center, ccw)
    # rot_offset = truncated_normal(1,3.,(rot_option-1.)*rot)

    # Random x translation
    tx = truncated_normal(1,5.,trans_offset) # [-16-trans,16+trans]
    ty = truncated_normal(1,5.,trans_offset)

    # Translate image
    M = np.float32([[1.,0.,tx],[0.,1.,ty]])
    Xt = cv2.warpAffine(X,M,(X.shape[0:2]),borderMode=cv2.BORDER_REPLICATE)
    Xt = cv2.transpose(Xt)

    Yt = cv2.warpAffine(Y,M,(Y.shape[0:2]),borderMode=cv2.BORDER_REPLICATE)
    Yt = cv2.transpose(Yt)

    # Xt = X
    # Yt = Y

    # Rotate image
    # center = tuple(np.array(X.shape[0:2])/2)
    # th = truncated_normal(1,7.,rot_offset)
    #
    # rot_mat = cv2.getRotationMatrix2D(center,th,1.)
    # Xtr = cv2.warpAffine(Xt,rot_mat,Xt.shape[0:2],borderMode=cv2.BORDER_REPLICATE)
    # Xtr = cv2.transpose(Xtr)
    #
    # Ytr = cv2.warpAffine(Yt,rot_mat,Yt.shape[0:2],borderMode=cv2.BORDER_REPLICATE)
    # Ytr = cv2.transpose(Ytr)

    # Add shadow
    Xt = random_shadow(Xt)

    return Xt, Yt

def gen_batch_function(data_folder, image_shape, augment=False):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_shape = (image_shape[1],image_shape[0])

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([0, 0, 255])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = cv2.resize(cv2.imread(image_file), image_shape)
                gt_image = cv2.resize(cv2.imread(gt_image_file), image_shape)

                if augment:
                    image = random_shadow(image)
                    # image, gt_image = augment_image(image, gt_image)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            # Convert to numpy arrays
            X = np.array(images)
            Y = np.array(gt_images)

            if augment:
                # Randomly flip each image
                # (This code can be vectorized, which is why it is not in
                # augment_image)
                flip_selector = rand(X.shape[0]) > 0.5
                X[flip_selector,:] = np.flip(X[flip_selector,:],axis=2)
                Y[flip_selector,:] = np.flip(Y[flip_selector,:],axis=2)

            yield X, Y
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = cv2.resize(cv2.imread(image_file), (image_shape[1],image_shape[0]))

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = (im_softmax > 0.5)
        overlay = np.dot(segmentation, np.array([[0, 128, 0]]))
        image[:,:,1][mask] = image[:,:,1][mask]/2
        street_im = image + overlay

        # street_im = np.dot(image + mask, np.array([[[1,0.5,1]]]))

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        cv2.imwrite(os.path.join(output_dir, name), image)
