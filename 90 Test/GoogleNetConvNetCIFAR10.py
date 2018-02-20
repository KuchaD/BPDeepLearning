import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import time
from datetime import timedelta
import os
import cifar10

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

def TestData():
    print("Size of:")
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(images_train)))
    print("- Test-set:\t\t{}".format(len(images_test)))

from cifar10 import img_size, num_channels, num_classes
img_size_cropped = 24



def plot_images(images, cls_true, cls_pred=None, smooth=True):
    
    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
        
        numberClass = cls_true[i]
        # Name of the true class.
        cls_true_name = class_names[numberClass]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Get the first images from the test-set.
images = images_test[0:9]
# Get the true classes for those images.
cls_true = cls_test[0:9]
# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=False)
plot_images(images=images, cls_true=cls_true, smooth=True)


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images

#distorted_images = pre_process(images=x, training=False)

def main_network(x_images):  
#Google Net
    network = x_images
    conv1_7_7 = tf.layers.conv2d(network, filters = 64, kernel_size = [7,7] , strides=2, activation=tf.nn.relu, name = 'conv1_7_7_s2')
    pool1_3_3 = tf.layers.max_pooling2d(conv1_7_7, pool_size = [3,3],strides=2,padding="same")
    pool1_3_3 = tf.nn.local_response_normalization(pool1_3_3)
    conv2_3_3_reduce = tf.layers.conv2d(pool1_3_3, filters = 64,kernel_size = [1,1], activation=tf.nn.relu,name = 'conv2_3_3_reduce',padding="same")
    conv2_3_3 = tf.layers.conv2d(conv2_3_3_reduce, filters = 192,kernel_size=[3,3], activation=tf.nn.relu, name='conv2_3_3',padding="same")
    conv2_3_3 = tf.nn.local_response_normalization(conv2_3_3)
    pool2_3_3 = tf.layers.max_pooling2d(conv2_3_3,pool_size=[3,3], strides=2, name='pool2_3_3_s2',padding="same")


    inception_3a_1_1 = tf.layers.conv2d(pool2_3_3, filters = 64, kernel_size = [1,1],padding="same", activation=tf.nn.relu, name='inception_3a_1_1')
    inception_3a_3_3_reduce = tf.layers.conv2d(pool2_3_3, filters =96,kernel_size = [1,1],padding="same", activation=tf.nn.relu, name='inception_3a_3_3_reduce')
    inception_3a_3_3 = tf.layers.conv2d(inception_3a_3_3_reduce,filters = 128,kernel_size =[3,3],padding="same",activation=tf.nn.relu, name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = tf.layers.conv2d(pool2_3_3,filters = 16, kernel_size=[1,1],padding="same",activation=tf.nn.relu, name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = tf.layers.conv2d(inception_3a_5_5_reduce, filters = 32, kernel_size = [5,5],padding="same", activation=tf.nn.relu, name= 'inception_3a_5_5')
    inception_3a_pool = tf.layers.max_pooling2d(pool2_3_3, pool_size=[3,3],strides=1,padding="same" )
    inception_3a_pool_1_1 = tf.layers.conv2d(inception_3a_pool, filters = 32, kernel_size = [1,1], activation=tf.nn.relu, name='inception_3a_pool_1_1',padding="same")

    #inception_3a__
    inception_3a_output = tf.concat([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],3, name='inception_3a_output')

    inception_3b_1_1 = tf.layers.conv2d(inception_3a_output, filters = 128,kernel_size = [1,1],activation=tf.nn.relu, name= 'inception_3b_1_1',padding="same" )
    inception_3b_3_3_reduce = tf.layers.conv2d(inception_3a_output, filters =128, kernel_size =[1,1], activation=tf.nn.relu, name='inception_3b_3_3_reduce',padding="same")
    inception_3b_3_3 = tf.layers.conv2d(inception_3b_3_3_reduce, filters =192, kernel_size=[3,3],  activation=tf.nn.relu,name='inception_3b_3_3',padding="same")
    inception_3b_5_5_reduce = tf.layers.conv2d(inception_3a_output, filters =32, kernel_size=[1,1], activation=tf.nn.relu, name = 'inception_3b_5_5_reduce',padding="same")
    inception_3b_5_5 = tf.layers.conv2d(inception_3b_5_5_reduce, filters =96, kernel_size = [5,5],  name = 'inception_3b_5_5',padding="same")
    inception_3b_pool = tf.layers.max_pooling2d(inception_3a_output, pool_size=[3,3], strides=1,  name='inception_3b_pool',padding="same")
    inception_3b_pool_1_1 = tf.layers.conv2d(inception_3b_pool, filters =64, kernel_size=[1,1],activation=tf.nn.relu, name='inception_3b_pool_1_1',padding="same")

    #inception_3b
    inception_3b_output = tf.concat([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],3, name='inception_3b_output')

    pool3_3_3 = tf.layers.max_pooling2d(inception_3b_output, pool_size=[3,3], strides=2, name='pool3_3_3',padding="same")
    inception_4a_1_1 = tf.layers.conv2d(pool3_3_3, filters =192, kernel_size = [1,1], activation=tf.nn.relu, name='inception_4a_1_1',padding="same")
    inception_4a_3_3_reduce = tf.layers.conv2d(pool3_3_3, filters =96, kernel_size = [1,1], activation=tf.nn.relu, name='inception_4a_3_3_reduce',padding="same")
    inception_4a_3_3 = tf.layers.conv2d(inception_4a_3_3_reduce, filters =208, kernel_size = [3,3],  activation=tf.nn.relu, name='inception_4a_3_3',padding="same")
    inception_4a_5_5_reduce = tf.layers.conv2d(pool3_3_3, filters =16, kernel_size = [1,1], activation=tf.nn.relu, name='inception_4a_5_5_reduce',padding="same")
    inception_4a_5_5 = tf.layers.conv2d(inception_4a_5_5_reduce, filters =48, kernel_size = [5,5],  activation=tf.nn.relu, name='inception_4a_5_5',padding="same")
    inception_4a_pool = tf.layers.max_pooling2d(pool3_3_3, pool_size=[3,3], strides=1,  name='inception_4a_pool',padding="same")
    inception_4a_pool_1_1 = tf.layers.conv2d(inception_4a_pool, filters =64, kernel_size = [1,1], activation=tf.nn.relu, name='inception_4a_pool_1_1',padding="same")

    #inception_4a
    inception_4a_output = tf.concat([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],3, name='inception_4a_output')


    inception_4b_1_1 = tf.layers.conv2d(inception_4a_output, filters =160, kernel_size = [1,1], activation=tf.nn.relu, name='inception_4b_1_1',padding="same")
    inception_4b_3_3_reduce = tf.layers.conv2d(inception_4a_output, filters =112, kernel_size = [1,1], activation=tf.nn.relu, name='inception_4b_3_3_reduce',padding="same")
    inception_4b_3_3 = tf.layers.conv2d(inception_4b_3_3_reduce, filters =224, kernel_size = [3,3], activation=tf.nn.relu, name='inception_4b_3_3',padding="same")
    inception_4b_5_5_reduce = tf.layers.conv2d(inception_4a_output,filters = 24, kernel_size = [1,1], activation=tf.nn.relu, name='inception_4b_5_5_reduce',padding="same")
    inception_4b_5_5 = tf.layers.conv2d(inception_4b_5_5_reduce, filters =64, kernel_size = [5,5],  activation=tf.nn.relu, name='inception_4b_5_5',padding="same")
    inception_4b_pool = tf.layers.max_pooling2d(inception_4a_output, pool_size=[3,3], strides=1,  name='inception_4b_pool',padding="same")
    inception_4b_pool_1_1 = tf.layers.conv2d(inception_4b_pool, filters =64, kernel_size = [1,1], activation=tf.nn.relu, name='inception_4b_pool_1_1',padding="same")

    #inception_4b    
    inception_4b_output = tf.concat([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],3, name='inception_4b_output')


    inception_4c_1_1 = tf.layers.conv2d(inception_4b_output, 128, 1, activation=tf.nn.relu,name='inception_4c_1_1',padding="same")
    inception_4c_3_3_reduce = tf.layers.conv2d(inception_4b_output, 128, 1, activation=tf.nn.relu, name='inception_4c_3_3_reduce',padding="same")
    inception_4c_3_3 = tf.layers.conv2d(inception_4c_3_3_reduce, 256,  3, activation=tf.nn.relu, name='inception_4c_3_3',padding="same")
    inception_4c_5_5_reduce = tf.layers.conv2d(inception_4b_output, 24, 1, activation=tf.nn.relu, name='inception_4c_5_5_reduce',padding="same")
    inception_4c_5_5 = tf.layers.conv2d(inception_4c_5_5_reduce, 64,  5, activation=tf.nn.relu, name='inception_4c_5_5',padding="same")
    inception_4c_pool = tf.layers.max_pooling2d(inception_4b_output, pool_size=3, strides=1,padding="same")
    inception_4c_pool_1_1 = tf.layers.conv2d(inception_4c_pool, 64, 1, activation=tf.nn.relu, name='inception_4c_pool_1_1',padding="same")

    #inception_3c
    inception_4c_output = tf.concat([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],3, name='inception_4c_output')

    inception_4d_1_1 = tf.layers.conv2d(inception_4c_output, 112, 1, activation=tf.nn.relu, name='inception_4d_1_1',padding="same")
    inception_4d_3_3_reduce = tf.layers.conv2d(inception_4c_output, 144, 1, activation=tf.nn.relu, name='inception_4d_3_3_reduce',padding="same")
    inception_4d_3_3 = tf.layers.conv2d(inception_4d_3_3_reduce, 288, 3, activation=tf.nn.relu, name='inception_4d_3_3',padding="same")
    inception_4d_5_5_reduce = tf.layers.conv2d(inception_4c_output, 32, 1, activation=tf.nn.relu, name='inception_4d_5_5_reduce',padding="same")
    inception_4d_5_5 = tf.layers.conv2d(inception_4d_5_5_reduce, 64, 5,  activation=tf.nn.relu, name='inception_4d_5_5',padding="same")
    inception_4d_pool = tf.layers.max_pooling2d(inception_4c_output, pool_size=3, strides=1,  name='inception_4d_pool',padding="same")
    inception_4d_pool_1_1 = tf.layers.conv2d(inception_4d_pool, 64, 1, activation=tf.nn.relu, name='inception_4d_pool_1_1',padding="same")


    #inception_4d
    inception_4d_output = tf.concat([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],3, name='inception_4d_output')

    inception_4e_1_1 = tf.layers.conv2d(inception_4d_output, 256, 1, activation=tf.nn.relu, name='inception_4e_1_1',padding="same")
    inception_4e_3_3_reduce = tf.layers.conv2d(inception_4d_output, 160, 1, activation=tf.nn.relu, name='inception_4e_3_3_reduce',padding="same")
    inception_4e_3_3 = tf.layers.conv2d(inception_4e_3_3_reduce, 320, 3, activation=tf.nn.relu, name='inception_4e_3_3',padding="same")
    inception_4e_5_5_reduce = tf.layers.conv2d(inception_4d_output, 32, 1, activation=tf.nn.relu, name='inception_4e_5_5_reduce',padding="same")
    inception_4e_5_5 = tf.layers.conv2d(inception_4e_5_5_reduce, 128,  5, activation=tf.nn.relu, name='inception_4e_5_5',padding="same")
    inception_4e_pool = tf.layers.max_pooling2d(inception_4d_output, pool_size=3, strides=1,  name='inception_4e_pool',padding="same")
    inception_4e_pool_1_1 = tf.layers.conv2d(inception_4e_pool, 128, 1, activation=tf.nn.relu, name='inception_4e_pool_1_1',padding="same")

    #inception_4e
    inception_4e_output = tf.concat([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],3, name='inception_4e_output')

    pool4_3_3 = tf.layers.max_pooling2d(inception_4e_output, pool_size=3, strides=2, name='pool_3_3',padding="same")


    inception_5a_1_1 = tf.layers.conv2d(pool4_3_3, 256, 1, activation=tf.nn.relu, name='inception_5a_1_1',padding="same")
    inception_5a_3_3_reduce = tf.layers.conv2d(pool4_3_3, 160, 1, activation=tf.nn.relu, name='inception_5a_3_3_reduce',padding="same")
    inception_5a_3_3 = tf.layers.conv2d(inception_5a_3_3_reduce, 320, 3, activation=tf.nn.relu, name='inception_5a_3_3',padding="same")
    inception_5a_5_5_reduce = tf.layers.conv2d(pool4_3_3, 32, 1, activation=tf.nn.relu, name='inception_5a_5_5_reduce',padding="same")
    inception_5a_5_5 = tf.layers.conv2d(inception_5a_5_5_reduce, 128, 5,  activation=tf.nn.relu, name='inception_5a_5_5',padding="same")
    inception_5a_pool = tf.layers.max_pooling2d(pool4_3_3, pool_size=3, strides=1,  name='inception_5a_pool',padding="same")
    inception_5a_pool_1_1 = tf.layers.conv2d(inception_5a_pool, 128, 1,activation=tf.nn.relu, name='inception_5a_pool_1_1',padding="same")
            
    inception_5a_output = tf.concat([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],3, name='inception_5a_output')

    inception_5b_1_1 = tf.layers.conv2d(inception_5a_output, 384, 1,activation=tf.nn.relu, name='inception_5b_1_1',padding="same")
    inception_5b_3_3_reduce = tf.layers.conv2d(inception_5a_output, 192, 1, activation=tf.nn.relu, name='inception_5b_3_3_reduce',padding="same")
    inception_5b_3_3 = tf.layers.conv2d(inception_5b_3_3_reduce, 384,  3,activation=tf.nn.relu, name='inception_5b_3_3',padding="same")
    inception_5b_5_5_reduce = tf.layers.conv2d(inception_5a_output, 48, 1, activation=tf.nn.relu, name='inception_5b_5_5_reduce',padding="same")
    inception_5b_5_5 = tf.layers.conv2d(inception_5b_5_5_reduce,128, 5,  activation=tf.nn.relu, name='inception_5b_5_5',padding="same")
    inception_5b_pool = tf.layers.max_pooling2d(inception_5a_output, pool_size=3, strides=1,  name='inception_5b_pool',padding="same")
    inception_5b_pool_1_1 = tf.layers.conv2d(inception_5b_pool, 128, 1, activation=tf.nn.relu, name='inception_5b_pool_1_1',padding="same")

    #inception_5b
    inception_5b_output = tf.concat([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],3, name='inception_5b_output')

    pool5_7_7 = tf.layers.average_pooling2d(inception_5b_output, pool_size=[7,7], strides=1,padding="same")

    net = tf.contrib.layers.flatten(pool5_7_7)
    net = tf.layers.dense(inputs=net, name='layer_dense',units=256, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, name='layer_dense',units=128, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, name='layer_fc_out',units=num_classes, activation=None)
    logits = net

    y_pred = tf.nn.softmax(logits=logits)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    
    return y_pred, loss

def create_network():
    images = x
    y_pred, loss = main_network(x_images = images)

    return y_pred, loss

y_pred, loss = create_network()
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
save_dir = 'save/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64
best_validation_accuracy = 0.0
last_improvement = 0

# Zastaveni pokud jiz dlouho nebyl lepsi vysledek
require_improvement = 3000
total_iterations = 0

def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

def optimize(num_iterations):
    # přistup k globalnim promenym
    global total_iterations
    global best_validation_accuracy
    global last_improvement
    
    start_time = time.time()

    for i in range(num_iterations):
        total_iterations += 1
        # Nacteni dat a label
        x_batch, y_true_batch = random_batch()

        # ulozeni do feed_dict
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        #Spusteni optimalizeru 
        session.run(optimizer, feed_dict=feed_dict_train)
        # Vytisteni informaci po kazdých 100 iteracích
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):
            # Spocitani presnosti z train baličku sady
            acc_train = session.run(accuracy, feed_dict=feed_dict_train) 
            # Spocitani presnosti z cele validační sady
            acc_validation, _ = validation_accuracy()
            #  Validacni sada je lepsi jak nejlepší sada 
            if acc_validation > best_validation_accuracy:
                
                best_validation_accuracy = acc_validation         
                # Ulozeni posledni zmeny
                last_improvement = total_iterations
                # Ulozi session
                saver.save(sess=session, save_path=save_path)
                improved_str = ' - Save'
            else:
                improved_str = ''
            
            # Stav
            msg = "Iterace: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
            # Tisk + tisk do log
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))
            

        # Dlouho neproběhlo lepsi nacitani
        if total_iterations - last_improvement > require_improvement:
            print("Stop Training")
            break

    end_time = time.time()
    time_dif = end_time - start_time
    
    # Cas zpracovani.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    
def restoreSession():
    saver.restore(sess=session, save_path=save_path)


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

batch_size = 256

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    
    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()

optimize(num_iterations=1)


optimize(num_iterations=99) 
print_test_accuracy(True,True)
optimize(num_iterations=3000)
 
print_test_accuracy(True,True)
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()
    
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)