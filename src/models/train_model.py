import sys
from ICAN import CrossModel
from ModelConfig import ModelConfig
import tensorflow as tf
from data_helper import batch_iter_multi

import numpy as np
import os
import time
import datetime
import sys

from t2t_req.src.config import FLAGS

if __name__ == "__main__":
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        model_config = ModelConfig()
        model_config.history_aggregate_mode = FLAGS.history_aggregate_mode
        model_config.use_residual = FLAGS.use_residual
        model_config.use_group_attention = FLAGS.use_group_attention
        model_config.user_activation = FLAGS.user_activation
        model_config.doc_activation = FLAGS.doc_activation
        model_config.use_position = FLAGS.use_position
        model = CrossModel(model_config)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        timestamp = str(int(time.time()))
        out_dir = FLAGS.log_path
        if out_dir=="None":
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", model.loss)


        # Train Summaries
        train_summary_op = loss_summary
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        if tf.gfile.Exists(train_summary_dir) and FLAGS.mode == "train":
            tf.gfile.DeleteRecursively(train_summary_dir)
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        def train_step(features, writer=None):
            """
            Evaluates model on a dev set
            """
            # print("begin ......")
            sex = features['use_sex']
            for i in range(len(sex)):
                if sex[i]>=5:
                    sex[i]=1
            feed_dict = {
                model.label_placeholder: features['label'], 
                model.business_placeholder: features['business_type'],
                model.user_sex_placeholder: sex,
                model.user_age_placeholder: features['user_age'],
                model.user_provice_placeholder: features['user_province'],
                model.user_city_placeholder: features['user_city'],
                model.short_video_placeholder: features['video_docids'],
                model.mini_video_placeholder: features['mini_docids'],
                model.mp_placeholder: features['mp_docids'],
                model.doc_vid_placeholder: features['doc_docid'],
                model.short_video_tag_placeholder: features["short_tag"],
                model.short_video_cate_placeholder: features["short_cate"],
                model.mini_video_tag_placeholder: features["mini_tag"],
                model.mini_video_cate_placeholder: features["mini_cate"],
                model.mp_docid_tag_placeholder: features["mp_tag"],
                model.mp_docid_cate_placeholder: features["mp_cate"],
                model.is_train : 1
            }

            _, step, summaries, loss, sample_logits = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.sampled_logits],
               feed_dict=feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            if step % 10 == 0:
                print("{}: step {}, loss {}".format(time_str, step, loss))
            if writer and step % 10 == 0 :
                writer.add_summary(summaries, step)

        if FLAGS.mode != "predict":
            for features in batch_iter_multi(FLAGS.training_file_path, batch_size=FLAGS.batch_size, epoch=1):
                train_step(features,train_summary_writer)
                current_step = tf.train.global_step(sess, global_step)
            saver.save(sess, checkpoint_prefix, global_step=current_step)
