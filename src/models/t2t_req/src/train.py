import os
import time
import math
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from config import FLAGS

from functools import partial

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from utils import encode_dataset, flatten, iter_data, find_trainable_variables, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path

attn_pdrop = 0.1
resid_pdrop = 0.1

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    """
    q: (batch * 2[s1 + s2]) * head_num * ctx_len * new_n_state 
    k: (batch * 2[s1 + s2]) * head_num * new_n_state * ctx_len
    v: (batch * 2[s1 + s2]) * head_num * ctx_len * new_n_state 
    """
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    #w = mask_attn_weights(w)
    w = tf.nn.softmax(w)
    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return w, a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    """
    x : (batch * 2[s1 + s2]) * ctx_len * n_state
    """
    if k:
        #(batch * 2[s1 + s2]) * head_num * new_n_state * ctx_len
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        #(batch * 2[s1 + s2]) * head_num * ctx_len * new_n_state 
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    """
    x : (batch * 2[s1 + s2]) * head_num * ctx_len * new_n_state 
    """
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), w) + b, shape_list(x)[:-1]+[nf])
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    """
    x : (batch * 2[s1 + s2]) * ctx_len * emb
    """
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        w, a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, train=train)
        a = dropout(a, resid_pdrop, train)
        return w, a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns["relu"]
        h = act(conv1d(x, 'c_fc', n_state, train=train))
        h2 = conv1d(h, 'c_proj', nx, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        n_head = 8
        w, a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return w, h

def embed(X, we):
    """
    X : batch * ctx_len
    we : vocab_len * embedding_size
    """
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    return e

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b



