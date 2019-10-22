'''
Created on Mar 20, 2019

@author: Adrien Papaioannou
'''

import time
import os
import scipy.io
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_cut_test(grid_spec, x, actual, pred, cut=0.25, x_lim=[-1.1,1.1], y_lim=[-1.1,1.1]):
        ax = plt.subplot(grid_spec)
        ax.plot(x, actual[cut,:], 'b-', linewidth = 2, label = 'actual', alpha=0.7)       
        ax.plot(x, pred[cut,:], 'r--', linewidth = 1, label = 'pred')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_title('$t = 0.' + str(cut) + '$', fontsize = 10) #TODO hardocoded int cut with t decimal
#         ax.axis('square')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

def plot_cut_test_std(grid_spec, x, actual, pred, pred_min, pred_max, cut=0.25, x_lim=[-1.1,1.1], y_lim=[-1.1,1.1]):
        ax = plt.subplot(grid_spec)
        ax.plot(x, actual[cut,:], 'b-', linewidth = 2, label = 'actual', alpha=0.7)       
        ax.plot(x, pred[cut,:], 'r--', linewidth = 1, label = 'pred')
        ax.fill_between(np.squeeze(x), pred_min[cut,:], pred_max[cut,:], color = 'orange', alpha = 0.4, label = '2-std')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_title('$t = 0.' + str(cut) + '$', fontsize = 10) #TODO hardocoded int cut with t decimal
        ax.axis('square')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

def net_pde_res_burger(u, x, t):
    nu = 0.01 / np.pi
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    f = u_t + u*u_x - nu*u_xx
    return f

def net_pde_res_simple(u, x):
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_squared = u * u
    f = u_xx - u_squared * u_x
    return f

def net_pde_res_buckley_simple(u, x, t):
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    Swc = 0.1
    Sor = 0.
    M = 10
    nu = 0.001 / np.pi# artificial diffusivity
    frac = tf.divide(tf.square(u-Swc),tf.square(u-Swc)+tf.divide(tf.square(1-u-Sor),M))
    frac_u = tf.gradients(frac,u)[0]
    f = u_t + frac_u*u_x - nu * u_xx
    return f

def net_pde_res_buckley_gravity(u, x, t):
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    Swc = 0.1
    Sor = 0
    M = 5
    Ng = 0
    theta = 90
    nu = 0#0.001 / np.pi# artificial diffusivity
    kro = tf.square(tf.divide(1 - u - Sor,1 - Swc - Sor))
    krw = tf.square(tf.divide(u - Swc,1 - Swc - Sor))
#     frac = tf.divide(1 - Ng * np.sin(theta * np.pi / 180) * kro,1 + tf.divide(kro, M * krw))
    frac = tf.square(u - Swc) * (1 - Ng * np.sin(theta * np.pi / 180) * tf.square(1 - u - Sor)/(1 - Sor - Swc)**2) / (tf.square(u - Swc) + tf.square(1 - u - Sor) / M)
    frac_u = tf.gradients(frac,u)[0]
    f = u_t + frac_u*u_x - nu * u_xx
    return f

def net_pde_res_buckley(u, x, t, nu):
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    Swc = 0.1
    Sor = 0.
    M = 10
    # nu = 0.1 / np.pi# artificial diffusivity
    frac = tf.divide(tf.square(u-Swc),tf.square(u-Swc)+tf.divide(tf.square(1-u-Sor),M))
    frac_u = tf.gradients(frac,u)[0]
    f = u_t + frac_u*u_x - nu * u_xx
    return f

def discriminator_loss(logits_real, logits_fake):
    # x = logits, z = labels
    # tf.nn.sigmoid_cross_entropy_with_logits <=> z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.zeros_like(logits_real)))
    dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)))
    dis_loss = dis_loss_real + dis_loss_fake
    return dis_loss

def generator_loss(logits_fake, logits_posterior, pde_residuals, w_posterior_loss, w_pde_loss):
    # x = logits, z = labels
    # tf.nn.sigmoid_cross_entropy_with_logits <=> z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    gen_loss_entropy = tf.reduce_mean(logits_fake)
    gen_loss_posterior = tf.reduce_mean(tf.multiply((w_posterior_loss - 1.0), tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_posterior, labels=tf.ones_like(logits_posterior))))
    gen_loss_pde = w_pde_loss * tf.reduce_mean(tf.square(pde_residuals), name='loss_pde_form')
    gen_loss = gen_loss_entropy + gen_loss_posterior + gen_loss_pde
    return gen_loss, gen_loss_entropy, gen_loss_posterior, gen_loss_pde


class MLPConfig(object):
    def __init__(self, layer_size_lst, activation_lst, main_name):
        self.layer_size_lst = layer_size_lst
        self.activation_lst = activation_lst
        self.main_name = main_name
        self.layer_nb = len(layer_size_lst)

def create_mlp(x, mlp_config, reuse=False):
    mlp = tf.layers.dense(x, mlp_config.layer_size_lst[0], 
                          activation=mlp_config.activation_lst[0],
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          name=mlp_config.main_name + '_layer_0', reuse=reuse)
    for i in range(1, mlp_config.layer_nb):
        mlp = tf.layers.dense(mlp, mlp_config.layer_size_lst[i], 
                              activation=mlp_config.activation_lst[i],
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              name=mlp_config.main_name + '_layer_' + str(i), reuse=reuse)    
    return mlp


