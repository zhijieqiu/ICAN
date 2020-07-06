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

def model(X, M, P, Y, train=False, reuse=False, pred_top_k = 10):
    with tf.variable_scope('model', reuse=reuse):
        batch_size = tf.shape(X)[0] 
        we = tf.get_variable("we", [n_vocab, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        pe = tf.get_variable("pe", [n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)
        pe = dropout(pe, embd_pdrop, train)

        X = tf.reshape(X, [-1, n_ctx])
        P = tf.reshape(P, [-1, n_ctx])
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we) + embed(P, pe)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        lm_h = tf.reshape(h, [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)

        #batch_softmax = tf.nn.softmax(lm_logits)
        _, top_k = tf.nn.top_k(lm_logits, pred_top_k)
        predict_item = tf.reshape(top_k, [batch_size, -1, pred_top_k])

        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(Y, [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]])
        lm_losses = tf.reduce_sum(lm_losses * M, 1) / tf.reduce_sum(M, 1)
        return lm_losses, lm_logits, predict_item, Y

def print_sample(gold, topk, i2w, sample_n, eos):
  res = ""
  gold = gold[:, :, np.newaxis]
  i = 0
  for (gold_seq, pred_seq) in zip(gold, topk):
    if i < sample_n:
      for (gold_item, pred_items) in zip(gold_seq, pred_seq):
        pred_list = []
        for pred_item in pred_items:
          pred_list.append(i2w[pred_item])
        res += "{} <-> {}\n".format(" ".join([i2w[x] for x in gold_item]), " ".join(pred_list))
        if i2w[gold_item[0]] == eos:
          break
      res += "-------------------------------\n"
    i += 1
  return res

def mgpu_train(*xs):
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            train_losses,_,_,_ = model(*xs, train=True, reuse=do_reuse)
            params = find_trainable_variables("model")
            grads = tf.gradients(train_losses, params)
            grads = list(zip(grads, params))
            gpu_grads.append(grads)
            gpu_ops.append([train_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    grads = average_grads(gpu_grads)
    grads = [g for g, p in grads]
    train = opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), 2000, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)
    return [train]+ops

def mgpu_predict(*xs):
    gpu_ops = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
            lm_losses = model(*xs, train=False, reuse=True)
            gpu_ops.append([lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    return ops

def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        x13 = [start]+x1[:max_len]+[delimiter]+x3[:max_len]+[clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def iter_apply(Xs, Ms, Ys):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    for xmb, mmb, ymb in iter_data(n_batch_train, False, True, float("inf"), Xs, Ms, Ys):
        n = len(xmb)
        if n == n_batch_train:
            res = sess.run([eval_mgpu_logits, eval_mgpu_clf_loss], {X_train:xmb, M_train:mmb, Y_train:ymb})
        else:
            res = sess.run([eval_logits, eval_clf_loss], {X:xmb, M:mmb, Y:ymb})
        res = [r*n for r in res]
        results.append(res)
    results = zip(*results)
    return [fn(res) for res, fn in zip(results, fns)]

def iter_predict(Xs, Ms):
    logits = []
    for xmb, mmb in iter_data(n_batch_train, False, True, float("inf"), Xs, Ms):
        n = len(xmb)
        if n == n_batch_train:
            logits.append(sess.run(eval_mgpu_logits, {X_train:xmb, M_train:mmb}))
        else:
            logits.append(sess.run(eval_logits, {X:xmb, M:mmb}))
    logits = np.concatenate(logits, 0)
    return logits

#def save(path):
#    ps = sess.run(params)
#    joblib.dump(ps, make_path(path))

def log():
    global best_score
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost/len(trY[:n_valid])
    va_cost = va_cost/n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1))*100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1))*100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f'%(n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            save(os.path.join(save_dir, desc, 'best_params.jl'))

argmax = lambda x:np.argmax(x, 1)

pred_fns = {
    'rocstories':argmax,
}

filenames = {
    'rocstories':'ROCStories.tsv',
}

label_decoders = {
    'rocstories':None,
}

def predict():
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))

def neg_sampling_loss(neg_num, hidden_out, trg_seq, trg_embedding, biases = False):
  hidden_size = hidden_out.shape[-1]
  output = tf.reshape(hidden_out, [-1, hidden_size])
  trg_flat = tf.reshape(trg, [-1])
  inputs_append = tf.expand_dims(output, 1)
  target_append = tf.expand_dims(trg_flat, -1)
  nb_classes, real_batch_size = tf.shape(trg_embedding)[0], tf.shape(target_append)[0]
  negative_sample = tf.random_uniform([real_batch_size, neg_num], 0, nb_classes, dtype=tf.int32)
  random_sample = tf.concat([target_append, negative_sample], axis = -1)
  #sampled_weights : batch * (1 + nb_negative) * embedding size
  sampled_weights = tf.nn.embedding_lookup(trg_embedding, random_sample)
  if biases:
    sampled_biases = tf.nn.embedding_lookup(biases, random_sample)
    sampled_logits = tf.matmul(inputs_append, sampled_weights, transpose_b=True)[:,0,:] + sampled_biases
  else:
    sampled_logits = tf.matmul(inputs_append, sampled_weights, transpose_b=True)[:,0,:]
  #sampled_logits : batch * (1 + nb_negative)
  sampled_labels = tf.zeros([real_batch_size], dtype=tf.int32)
  return sampled_labels, sampled_logits

if __name__ == '__main__':
    print(FLAGS)
    globals().update(FLAGS.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **FLAGS.__dict__)
    vocab_i2w = load_vocab(vocab_path, split_str)
    n_vocab = len(vocab_i2w)

    train_data = build_data(train_path, n_batch, n_ctx, True, n_iter)
    tr_iter = train_data.make_one_shot_iterator()
    X, Y, M, P = tr_iter.get_next()

    test_data = build_data(test_path, n_batch, n_ctx, False, n_iter)
    #te_iter = test_data.make_one_shot_iterator()
    #teX, teY, teM, teP = va_iter.get_next()

    #train_losses,_,_ = model(X, M, P, Y, train=True)
    train, train_losses = mgpu_train(X, M, P, Y)
    lm_loss = tf.reduce_mean(train_losses)
    _, hidden_logit, sample_out, gold = model(X, M, P, Y, train=False, reuse = True)
    _, te_hidden_logit, te_sample_out, te_gold = model(X, M, P, Y, train=False, reuse = True)
    #train = tf.train.AdamOptimizer().minimize(lm_loss)

    params = find_trainable_variables('model')
    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in xrange(2):
        print "epochs: " + str(i)
        print sess.run([train, lm_loss])
        if i % 100 == 1:
          sample_out_, gold_  = sess.run([sample_out, gold])
          #print print_sample(gold_, sample_out_, vocab_i2w, 10, '<EOS>')
      saver.save(sess, os.path.join(chpt_dir, 'params.ckpt'))
      print "model saved"
