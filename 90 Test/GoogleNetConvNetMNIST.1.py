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
img_shape = (img_size, img_size)
img_size_flat = img_size * img_size

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


#Nastaveni TF variables
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#Google Net

network = x
conv1_7_7 = tf.layers.conv2d(network, filters = 64, kernel_size = [7,7] , strides=2, activation=tf.nn.relu, name = 'conv1_7_7_s2',padding="same")
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

#inception_5a       
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

flat = tf.contrib.layers.flatten(pool5_7_7)
dense = tf.layers.dense(inputs=flat, name='layer_dense',units=256, activation=tf.nn.relu)
dense = tf.layers.dense(inputs=dense, name='layer_dense2',units=128, activation=tf.nn.relu)
dense = tf.layers.dense(inputs=dense, name='layer_fc_out',units=num_classes, activation=None)
logits = dense

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
save_dir = 'save/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(session.run(tf.global_variables_initializer()))

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

    
#Testovaci funkce
def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

#validacni 
def predict_cls_validation():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

#spocita spravnost
def cls_accuracy(correct):
    
    correct_sum = correct.sum()
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

def validation_accuracy():
   
    correct, _ = predict_cls_validation()
    return cls_accuracy(correct)

def restoreSession():
    saver.restore(sess=session, save_path=save_path)


def plot_example_errors(cls_pred, correct):
 
    incorrect = (correct == False)
    images = images_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = cls_test[incorrect]
    
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):

    cls_true = cls_test
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)

    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')


    plt.show()

test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):


    num_test = len(images_test)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = images_test[i:j, :, :, :]
        labels = labels_test[i:j, :]
        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = cls_test

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

batch_size = 256

def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :, :, :],
                     y_true: labels[i:j, :]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred

optimize(num_iterations=1)


optimize(num_iterations=99) 
print_test_accuracy(True,True)
optimize(num_iterations=5000)
 
print_test_accuracy(True,True)
def plot_image(image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2)

    # References to the sub-plots.
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]

    # Show raw and smoothened images in sub-plots.
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')

    # Set labels.
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

image1 = images_test[0]

plot_image(image1)