import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

LEARNING_RATE = 1e-4
KEEP_PROB = 0.5

# Using conda environment: carnd-term1-gpu


# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    name_input = 'image_input:0'
    name_keep_prob = 'keep_prob:0'
    name3 = 'layer3_out:0'
    name4 = 'layer4_out:0'
    name7 = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(name_input)
    keep_prob = tf.get_default_graph().get_tensor_by_name(name_keep_prob)
    layer3_out = tf.get_default_graph().get_tensor_by_name(name3)
    layer4_out = tf.get_default_graph().get_tensor_by_name(name4)
    layer7_out = tf.get_default_graph().get_tensor_by_name(name7)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def conv2d_1x1(layer_in, num_classes):
    return tf.layers.conv2d(layer_in, num_classes, 1,
        padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

def upSample(layer_in, num_classes, kern_size, stride):
    return tf.layers.conv2d_transpose(layer_in, num_classes, kern_size,
        strides= stride,
        padding= 'same',
        kernel_initializer= tf.truncated_normal_initializer(stddev=0.01),
        activation=tf.nn.relu,
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    N = num_classes

    # vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    # vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    # vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)

    # scale layers
    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001)
    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.001)

    # apply 1x1 convolutions to all of the input layers
    l7_1x1 = conv2d_1x1(vgg_layer7_out,N)
    l4_1x1 = conv2d_1x1(vgg_layer4_out,N)
    l3_1x1 = conv2d_1x1(vgg_layer3_out,N)

    # Start upsampling from layer7 input and add in skip connections
    output = tf.add(upSample(l7_1x1, N, 4, 2), l4_1x1) # up(7) + 4
    output = tf.add(upSample(output, N, 4, 2), l3_1x1) # up(up(7)+4) + 3
    output = upSample(output, N, 16, 8) # up(up(up(7)+4) + 3)

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # define loss function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    l2_const = 5e-3;
    loss = cross_entropy_loss + l2_const * sum(reg_losses)

    train_op = optimizer.minimize(loss)

    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0;
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = {input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE})
            total_loss += loss
        print("Epoch: {}/{}, Loss: {:.4f}, Total-loss: {:.4f}, dt: {}".format(epoch+1,epochs,loss,total_loss,time.time()-start))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    # image_shape = (64, 230)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 1
    batch_size = 2

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches with image augmentation (rotation, random shadows)
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape, augment=False)

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes]) # placeholder
        learning_rate = tf.placeholder(tf.float32) # placeholder
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
