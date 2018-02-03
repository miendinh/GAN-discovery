import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import next_batch

batch_size = 10;
epochs = 3000;
k = 21;
Z_dimension = 100;

def conv2d(X, filters, kernel_size = 5, strides = 2, padding = 'same', is_training = True):
    X = tf.layers.conv2d(X, filters, kernel_size, strides = strides, padding = padding )
    X = tf.layers.batch_normalization(X, training = is_training)
    X = tf.nn.leaky_relu(X)
    return X

def deconv(Z, filters, kernel_size = 5, strides = 2, padding = 'same', is_training = True):
    Z = tf.layers.conv2d_transpose(Z, filters, kernel_size, strides = strides, padding= padding)
    Z = tf.layers.batch_normalization(Z, training = is_training)
    Z = tf.nn.relu(Z)
    return Z

def Generator(Z, reuse = False, is_training = True):
    with tf.variable_scope("Generator") as scope:
        if reuse:
            scope.reuse_variables()

        Z = tf.layers.dense(Z, units=4*4*1024)
        Z = tf.layers.batch_normalization(Z, training = is_training)
        Z = tf.nn.relu(Z)
        Z = tf.reshape(Z, shape=[-1, 4, 4, 1024])
        Z = deconv(Z, 1024)
        Z = deconv(Z, 512)
        Z = deconv(Z, 256)
        Z = deconv(Z, 128)
        Z = deconv(Z, 3)
        Z = tf.nn.tanh(Z)
        return Z

def Discriminator(X, reuse = False, is_training = True):
    with tf.variable_scope("Discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        X = conv2d(X, 32, is_training = is_training)
        X = conv2d(X, 64, is_training = is_training)
        X = conv2d(X, 128, is_training = is_training)
        X = conv2d(X, 256, is_training = is_training)
        X = conv2d(X, 512, is_training = is_training)
        X = conv2d(X, 1024, is_training = is_training)

        X = tf.reshape(X, shape=[-1, 2*2*1024])
        X = tf.layers.dense(X, 1024)
        X = tf.layers.batch_normalization(X, training = is_training)
        X = tf.layers.dense(X, 1)

        return X

Z = tf.placeholder(tf.float32, shape=(None, 100), name = 'Z')
X = tf.placeholder(tf.float32, shape=(None, 128, 128, 3), name = 'X')

G = Generator(Z)
Dx = Discriminator(X)
Dg = Discriminator(G, reuse = True)

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))
Dx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size, 1], 1.)))
Dg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

D_loss = Dx_loss + Dg_loss

G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

with tf.variable_scope(tf.get_variable_scope(), reuse = False) as scope:
    Dx_trainner = tf.train.AdamOptimizer(0.001).minimize(Dx, var_list = D_vars)
    Dg_trainner = tf.train.AdamOptimizer(0.001).minimize(Dg, var_list = D_vars)
    G_trainner = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list = G_vars)

noises = np.random.uniform(-1., 1., size = [batch_size, Z_dimension])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        Z_random_noises = np.random.uniform(-1., 1., [batch_size, Z_dimension])
        for i in range(k):
            images = next_batch(0, batch_size)
            _, _, D_loss_ = sess.run([Dx_trainner, Dg_trainner, D_loss], {X : images, Z : Z_random_noises }) #is_training : True

        _, G_loss_ = sess.run([G_trainner, G_loss], feed_dict={Z : Z_random_noises}) #is_training : True

        print("G_loss = ", G_loss_, "D_loss = ", D_loss_)

        if e % 100 == 0 or e == 1:
            Z_gen, Dl, Gl = sess.run([G, D_loss, G_loss], {X: images, Z: noises})
            plt.imshow(Z_gen[0,:,:,:].reshape(128, 128, 3))
            plt.show()
