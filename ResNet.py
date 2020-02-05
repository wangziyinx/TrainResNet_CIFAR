from __future__ import print_function
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import scipy.io
import os.path
import scipy.misc
import random
import threading
from Support_Functions import *



def resblock(input, pSize, nkernels,  resinitializer, training, stride = (1,1), name = None, BN = False, prj = False):
    if BN:
        resconv1 = tf.layers.conv2d(input, nkernels, (pSize, pSize), activation=None, name=name+ '_1',
                                  strides=stride, kernel_initializer=resinitializer, padding='SAME')
        resconv1_BN =  tf.nn.leaky_relu(tf.layers.batch_normalization(resconv1, training = training))



        resconv2 = tf.layers.conv2d(resconv1_BN,
                                    nkernels, (3, 3), activation=None,
                                    name=name + '_2',
                                    kernel_initializer=resinitializer, padding='SAME')
        # resconv2_BN = tf.nn.leaky_relu(tf.layers.batch_normalization(resconv2, training=training))
        resconv2_BN = tf.layers.batch_normalization(resconv2, training=training)
        if prj:
            input_prj = tf.layers.conv2d(input, nkernels, (1, 1), activation=None, name=name+ '_prj',
                                         strides=stride, kernel_initializer=resinitializer, padding='SAME')
            return input_prj + resconv2_BN
        else:
            return input + resconv2_BN
    else:
        resconv1 = tf.layers.conv2d(input, nkernels, (pSize, pSize),
                                    activation=tf.nn.leaky_relu, name=name + '_1',
                                    kernel_initializer=resinitializer, padding='SAME')
        resconv2 = tf.layers.conv2d(resconv1, nkernels, (3, 3), activation=None,
                                    name=name + '_2',
                                    kernel_initializer=resinitializer, padding='SAME')
        if prj:
            input_prj = tf.layers.conv2d(input, nkernels, (pSize, pSize), activation=None,
                                         name=name + '_prj',
                                         kernel_initializer=resinitializer, padding='SAME')
            return input_prj + resconv2
        else:
            return input + resconv2

def train_one_epoch( training_data, train_eval = [], train = True, mini_batch = 64, lr = 0.01,
                     sess = [], ll = [], Img = [], logits = [], resize = False, aug = False):

    x = training_data[0]
    y = training_data[1]
    # n_img = x.shape[0]
    n_img = x.shape[0] - x.shape[0]%mini_batch
    x = x[:n_img, :,:,:]
    y = y[:n_img]
    loss_epoch = 0
    train_cycle = 0
    predictions = -1*np.ones(n_img,dtype=np.int32)
    for i in range(0, n_img - 1, mini_batch):
        last_indx = i + mini_batch
        if (last_indx >= x.shape[0]):
            break


        batch_labels = y[i: last_indx]
        if train == True:
            ImgArray = augment_batch(x[i: last_indx, :, :, :], 32, resize=resize, aug=aug)
            train_op = train_eval[0]
            sess.run(train_op, feed_dict={Img: ImgArray, ll: batch_labels, learning_rate: lr, training: True})
        else:
            ImgArray = augment_batch(x[i: last_indx, :, :, :], 32, resize=False, aug=False)
            loss = train_eval[0]
            accuracy = train_eval[1]
            loss_epoch = loss_epoch + sess.run(loss, feed_dict={Img: ImgArray, ll: batch_labels, training: False})
            logit_batch = logits.eval(feed_dict={Img: ImgArray, ll: batch_labels, training: False})
            predictions_batch = predict(logit_batch)
            predictions[i:last_indx] =  predictions_batch.astype(np.int32)
            train_cycle = train_cycle + 1
    if train == False:
        return loss_epoch/train_cycle, compute_acc(predictions, y), compute_mAP(predictions, y)
        # return loss_epoch/train_cycle, acc_train/train_cycle, mAP/train_cycle


###########################################################################
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
num_class = 10

y_train = y_train[:,0]
y_test = y_test[:,0]
#

x_train, y_train = shuffle_data(x_train, y_train)
x_test, y_test = shuffle_data(x_test, y_test)
# x_train = np.concatenate((x_train, x_test[:4000]), axis = 0)
# y_train = np.concatenate((y_train, y_test[:4000]), axis = 0)
###########################################################################

mini_batch = 128
lr = 0.001

tf.reset_default_graph()
# resinitializer = tf.initializers.random_uniform(minval = -1e-5, maxval = 1e-5)
resinitializer = None
batch_norm = True
training = tf.placeholder(tf.bool, name="is_train")
Img = tf.placeholder(tf.float32, shape = (mini_batch, 32, 32, 3), name = "mini_batch_imgs")

#-----------------------------------------------Layer 1_1---------------------------------------------------------------
# Conv 1_1 layer
conv1_0 = tf.layers.conv2d(Img, 16, (3, 3), activation=tf.nn.leaky_relu, name = 'conv1', padding='SAME')
BN_conv1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1_0, training = training))

res1   = tf.nn.leaky_relu( resblock(BN_conv1, 3, 16,  resinitializer, training, stride = (1,1), name = 'res_block_1_1', BN = batch_norm))
res1_2 = tf.nn.leaky_relu( resblock(res1, 3, 16,  resinitializer, training, stride = (1,1), name = 'res_block_1_2', BN = batch_norm))
res1_3 = tf.nn.leaky_relu( resblock(res1_2, 3, 16,  resinitializer, training, stride = (1,1), name = 'res_block_1_3', BN = batch_norm))

res2 = tf.nn.leaky_relu( resblock(res1_3, 3, 32,  resinitializer, training, stride = (2,2), name = 'res_block_2_1', BN = batch_norm, prj = True))
res2_2 = tf.nn.leaky_relu( resblock(res2, 3, 32,  resinitializer, training, stride = (1,1), name = 'res_block_2_2', BN = batch_norm))
res2_3 = tf.nn.leaky_relu( resblock(res2_2, 3, 32,  resinitializer, training, stride = (1,1), name = 'res_block_2_3', BN = batch_norm))

res3 = tf.nn.leaky_relu( resblock(res2_3, 3, 64,  resinitializer, training, stride = (2,2), name = 'res_block_3_1', BN = batch_norm, prj = True))
res3_2 = tf.nn.leaky_relu( resblock(res3, 3, 64,  resinitializer, training, stride = (1,1), name = 'res_block_3_2', BN = batch_norm))
res3_3 = tf.nn.leaky_relu( resblock(res3_2, 3, 64,  resinitializer, training, stride = (1,1), name = 'res_block_3_3', BN = batch_norm))

avgpool = tf.layers.average_pooling2d ( inputs=res3_3, pool_size=[8, 8],  strides=1)
pool_Flat = tf.contrib.layers.batch_norm(tf.layers.flatten(avgpool))

logits = tf.layers.dense(pool_Flat, num_class, name = 'logits', activation=tf.nn.leaky_relu)

# with tf.variable_scope("fc1", reuse=True):
#     w1 = tf.get_variable("kernel")
#     b1 = tf.get_variable("bias")
with tf.variable_scope("logits", reuse=True):
    w2 = tf.get_variable("kernel")
    b2 = tf.get_variable("bias")

ll = tf.placeholder (tf.int32, name = "mini_batch_labels", shape = mini_batch)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits (labels=ll, logits = logits)
loss = tf.reduce_mean(xentropy, name = "loss")

# with tf.name_scope ("train_local"):
# train_list =  fc1_varlist + logits_valist
learning_rate = tf.placeholder(tf.float32, shape=[])
# optimizer = tf.contrib.opt.AdamWOptimizer(1e-4,  epsilon=1e-4, learning_rate = learning_rate)
optimizer = tf.train.AdamOptimizer(epsilon=1e-5, learning_rate = learning_rate)
train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.group([train_op, update_ops])


with tf.name_scope ("eval"):
    correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), ll, 1)
    accuracy = tf.reduce_mean(tf.cast (correct, tf.float32))

result_file_name='Res_recording_'+str(num_class)+'.txt'
result_file = open(result_file_name,'w+')
result_file.close()

best_val = 0
saver= tf.train.Saver()
best_count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, 'C:\\Users\\wangz\\PycharmProjects\\Memory_CNN CIFAR\\results'+str(num_class)+'.ckpt')
    # save_path = saver.save(sess, 'C:\\Users\\wangz\\PycharmProjects\\Memory_CNN CIFAR\\results' + str(
    #     num_class) + '.ckpt')
    print('------------------------------------------------------------------------')
    for epoch in range(120):
        if epoch >= 80 and lr >1e-4:
            lr = lr/10
            print(lr)
        train_one_epoch([x_train, y_train], train_eval=[train_op], train=True,
                            mini_batch=mini_batch, lr=lr,sess=sess, ll=ll, Img=Img, logits=logits , resize = True, aug = True)

        x_train, y_train = shuffle_data(x_train, y_train)
        x_test, y_test = shuffle_data(x_test, y_test)

        loss_epoch_train, acc_train, mAP_train = train_one_epoch([x_train, y_train],
                                                                 train_eval=[loss, accuracy],
                                                                 train=False, mini_batch=mini_batch, lr=lr,
                                                                 sess=sess, ll=ll, Img=Img, logits=logits)
        loss_epoch_val, acc_val, mAP_val = train_one_epoch([x_test, y_test],
                                                           train_eval=[loss, accuracy],
                                                           train=False, mini_batch=mini_batch, lr=lr,
                                                           sess=sess, ll=ll, Img=Img, logits=logits)

        result_file = open(result_file_name, 'a+')
        epoch_result = str(epoch) + " " + str(acc_train * 100) + " " + str(loss_epoch_train) + " " + str(
            mAP_train * 100) + " " + str(acc_val * 100) + " " + str(loss_epoch_val) + " " + str(mAP_val * 100) + '\n'
        result_file.write(epoch_result)
        result_file.close()

        print(epoch, " ",
              (acc_train * 100), " ", loss_epoch_train, " ", (mAP_train * 100), " ",
              (acc_val * 100), " ", loss_epoch_val, " ", (mAP_val * 100))
        if acc_val > best_val:
            save_path = saver.save(sess, 'C:\\Users\\wangz\\PycharmProjects\\Memory_CNN CIFAR\\results'+str(num_class)+'.ckpt')


