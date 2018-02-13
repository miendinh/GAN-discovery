import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import DataSet
import sys

batch_size = 100;
sample_size = 100;
epochs = 3001;
steps = 1000;
Z_dimension = 100;
dataset = DataSet(test_prob=0, one_hot=False)

def conv2d(X, filters, kernel_size = 2, strides = 2, padding = 'same', is_training = True):
    X = tf.layers.conv2d(X, filters, kernel_size, strides = strides, padding = padding )
    X = tf.layers.batch_normalization(X, training = is_training)
    X = tf.nn.leaky_relu(X)
    return X

def deconv(Z, filters, kernel_size = 2, strides = 2, padding = 'same', is_training = True):
    Z = tf.layers.conv2d_transpose(Z, filters, kernel_size, strides = strides, padding= padding)
    Z = tf.layers.batch_normalization(Z, training = is_training)
    Z = tf.nn.relu(Z)
    return Z

def Generator(Z, reuse = False, is_training = False):
    with tf.variable_scope("Generator") as scope:
        if reuse:
            scope.reuse_variables()

        Z = tf.layers.dense(Z, units=3*3*128)
        Z = tf.layers.batch_normalization(Z, training = is_training)
        Z = tf.nn.relu(Z)
        Z = tf.reshape(Z, shape=[-1, 3, 3, 128])
        Z = deconv(Z, 128, kernel_size = 3, padding = 'valid')
        Z = deconv(Z, 64)
        Z = deconv(Z, 1)
        Z = tf.nn.tanh(Z)
        return Z

def Discriminator(X, reuse = False, is_training = False):
    with tf.variable_scope("Discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        X = conv2d(X, 32, is_training = is_training)
        X = conv2d(X, 64, is_training = is_training)
        X = conv2d(X, 128, kernel_size = 3, padding = 'valid', is_training = is_training)
        X = tf.reshape(X, shape=[-1, 3*3*128])
        X = tf.layers.dense(X, 128)
        X = tf.layers.batch_normalization(X, training = is_training)
        X = tf.layers.dense(X, 1)

        return tf.nn.sigmoid(X), X

Z = tf.placeholder(tf.float32, shape=(None, 100), name = 'Z')
X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name = 'X')

G = Generator(Z)
Dx_s, Dx = Discriminator(X)
Dg_s, Dg = Discriminator(G, reuse = True)

Dx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx_s)))
Dg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg_s)))
D_loss = Dx_loss + Dg_loss
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg_s)))

tf.summary.scalar('Dx_loss', Dx_loss)
tf.summary.scalar('Dg_loss', Dg_loss)
tf.summary.scalar('D_loss',  D_loss)
tf.summary.scalar('G_loss',  G_loss)

summary = tf.summary.merge_all()

G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

with tf.variable_scope(tf.get_variable_scope(), reuse = False) as scope:
    Dx_trainner = tf.train.AdamOptimizer(0.0001).minimize(Dx_loss, var_list = D_vars)
    Dg_trainner = tf.train.AdamOptimizer(0.0001).minimize(Dg_loss, var_list = D_vars)
    G_trainner = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list = G_vars)

sample_noises = np.random.uniform(-1., 1., size = [sample_size, Z_dimension])
sample_images, _ = dataset.next_batch(batch_size)
sample_images = sample_images.reshape((batch_size, 28, 28, 1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs', sess.graph)

    for e in range(epochs):
        for i in range(steps):
            images, _ = dataset.next_batch(batch_size)
            if len(images) < batch_size:
                continue
            images = images.reshape((batch_size, 28, 28, 1))

            Z_random_noises = np.random.uniform(-1., 1., [batch_size, Z_dimension])

            _, _, Dx_loss_, Dg_loss_, summary_ = sess.run([Dx_trainner, Dg_trainner, Dx_loss, Dg_loss, summary], {X : images, Z : Z_random_noises })

            _, G_loss_ = sess.run([G_trainner, G_loss], feed_dict={Z : Z_random_noises})
            _, G_loss_ = sess.run([G_trainner, G_loss], feed_dict={Z : Z_random_noises})

            print("Epoch:%04d Step: %04d G_loss: %.8f Dx_loss: %.8f Dg_loss: %.8f" % (e, i, G_loss_, Dx_loss_, Dg_loss_))

        writer.add_summary(summary_, e + 1)

        if e % 1 == 0:
            Z_gen = sess.run(G, feed_dict={Z: sample_noises})
            Z_gen = np.array(Z_gen.reshape((sample_size, 28, 28)) + 1) * 127.5
            big_image = Z_gen.reshape(10, 10, 28, 28).swapaxes(1,2).reshape(10*28, 10*28)
            plt.imshow(big_image, cmap='gray')
            plt.show()
            cv2.imwrite("generated/{0:05d}.jpg".format(e), big_image)

    saver = tf.train.Saver()
    save_path = saver.save(sess, "./gan_discover.ckpt")
    print('done!')
