import numpy as np
import sys
import tensorflow as tf
import re


def libsvm_field_data_iterator_multi(input_file, batch_size, num_epochs):
  # Read TFRecords files for training
  filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(input_file),
      num_epochs=num_epochs)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  batch_serialized_example = tf.train.shuffle_batch(
      [serialized_example],
      batch_size=batch_size,
      num_threads=8,
      capacity=100000,
      min_after_dequeue=10000 + 3*batch_size)

  features={
      #'uin': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.float32),
      'use_sex': tf.FixedLenFeature([], tf.int64),
      'business_type': tf.FixedLenFeature([], tf.int64),
      'user_age': tf.FixedLenFeature([], tf.int64),
      'user_province': tf.FixedLenFeature([], tf.int64),
      'user_city': tf.FixedLenFeature([], tf.int64),
      'video_docids': tf.FixedLenFeature([5], tf.int64),
      'mini_docids': tf.FixedLenFeature([5], tf.int64),
      'mp_docids': tf.FixedLenFeature([5], tf.int64),
      'doc_docid': tf.FixedLenFeature([], tf.int64),
      'short_tag':tf.FixedLenFeature([20], tf.int64),
      'short_cate':tf.FixedLenFeature([5], tf.int64),
      'mini_tag':tf.FixedLenFeature([20], tf.int64),
      'mini_cate':tf.FixedLenFeature([5], tf.int64),
      'mp_tag':tf.FixedLenFeature([20], tf.int64),
      'mp_cate':tf.FixedLenFeature([5], tf.int64),
  }

  features = tf.parse_example(batch_serialized_example, features=features)
  return features

def libsvm_field_data_iterator2(input_file, batch_size, num_epochs):
  # Read TFRecords files for training
  filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(input_file),
      num_epochs=num_epochs)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  batch_serialized_example = tf.train.shuffle_batch(
      [serialized_example],
      batch_size=batch_size,
      num_threads=8,
      capacity=100000,
      min_after_dequeue=10000 + 3*batch_size)

  features={
      'label': tf.FixedLenFeature([], tf.int64),
      'use_sex': tf.FixedLenFeature([], tf.int64),
      'user_age': tf.FixedLenFeature([], tf.int64),
      'user_province': tf.FixedLenFeature([], tf.int64),
      'user_city': tf.FixedLenFeature([], tf.int64),
      'user_category': tf.FixedLenFeature([30], tf.int64),
      'user_tag': tf.FixedLenFeature([40], tf.int64),
      'user_docid': tf.FixedLenFeature([5], tf.int64),
      'user_ctype': tf.FixedLenFeature([10], tf.int64),
      'user_recent_category': tf.FixedLenFeature([10], tf.int64),
      'user_recent_tag': tf.FixedLenFeature([20], tf.int64),
      'user_profile_category': tf.FixedLenFeature([30], tf.int64),
      'user_profile_ctype': tf.FixedLenFeature([10], tf.int64),
      'user_profile_tag': tf.FixedLenFeature([40], tf.int64),
      'doc_docid': tf.FixedLenFeature([], tf.int64),
      'doc_tag': tf.FixedLenFeature([10],tf.int64),
      'doc_category': tf.FixedLenFeature([],tf.int64)
  }

  features = tf.parse_example(batch_serialized_example, features=features)
  return features

short_tag = np.random.randint(1,110000,(2500000,5))
short_cate = np.random.randint(1,240,2500000)
mp_tag = np.random.randint(1,603192,(1021160,5))
mp_cate = np.random.randint(1,480,1021160)
def collect_tags(docids, tag_list):
    tag_map = dict()
    for docid in docids:
        tags = tag_list[docid]
        for t in tags:
            tag_map[t] = tag_map.get(t,0)+1
    tag_ret = tag_map.keys()[:20]
    tag_ret.extend([0]*(20-len(tag_ret)))
    return tag_ret
def collect_category(docids,cate_list):
    tag_map = dict()
    for docid in docids:
        t = cate_list[docid]
        tag_map[t] = tag_map.get(t,0)+1
    tag_ret = tag_map.keys()[:5]
    tag_ret.extend([0]*(5-len(tag_ret)))
    return tag_ret
def extend_features(features):
    return
    batch_size = len(features.get("doc_docid",[]))
    #features["short_tag"] = [[0]*20]*batch_size
    #features["short_cate"] = [[0]*5]*batch_size
    #features["mini_tag"] = [[0]*20]*batch_size
    #features["mini_cate"] = [[0]*5]*batch_size
    #features["mp_docid_tag"] = [[0]*20]*batch_size
    #features["mp_docid_cate"] = [[0]*5]*batch_size
    features["short_tag"] = []
    features["short_cate"] = []
    features["mini_tag"] = []
    features["mini_cate"] = []
    features["mp_docid_tag"] = []
    features["mp_docid_cate"] = []
    for i in range(batch_size):
        docid = features["video_docids"][i]
        features["short_tag"].append(collect_tags(docid, short_tag))
        features["short_cate"].append(collect_category(docid, short_cate))
        docid = features["mini_docids"][i]
        features["mini_tag"].append(collect_tags(docid, short_tag))
        features["mini_cate"].append(collect_category(docid, short_cate))
        docid = features["mp_docids"][i]
        features["mp_docid_tag"].append(collect_tags(docid, mp_tag))
        features["mp_docid_cate"].append(collect_category(docid, mp_cate))

       



def batch_iter_multi(tf_record_file_name,batch_size, epoch=1):
    #write_tf(train_file, r'tf.records',max_length)
    #feature_batch = DataHelper.test(tf_record_file_name, batch_size)
    feature_batch = libsvm_field_data_iterator_multi(tf_record_file_name, batch_size, epoch)
    with tf.Session() as sess:
        i = 0
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            mini_features = dict()
            shortvideo_features = dict()
            mp_features = dict()
            while not coord.should_stop() :
                features = sess.run(feature_batch)
                for i in range(len(features["doc_docid"])):
                    if features["business_type"][i] == 1:
                        a_dict = mini_features
                    elif features["business_type"][i] == 2:
                        a_dict = shortvideo_features
                    elif features["business_type"][i] == 3:
                        a_dict = mp_features
                    for key,value in features.items():
                        if key not in a_dict:
                            a_dict[key] = []
                        a_dict[key].append(value[i])
                if mini_features and len(mini_features["doc_docid"])>=batch_size:
                    extend_features(mini_features)
                    yield mini_features
                    mini_features.clear()
                if shortvideo_features and len(shortvideo_features["doc_docid"])>=batch_size:
                    extend_features(shortvideo_features)
                    yield shortvideo_features
                    shortvideo_features.clear()
                if mp_features and len(mp_features["doc_docid"])>=batch_size:
                    extend_features(mp_features)
                    yield mp_features
                    mp_features.clear()
                i+=1
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)

                   
        

if __name__ == "__main__":
    #data = load_origin_data(r'/data1/searchusers/hdpjackqiu/user_embed/mapreduce/generate_traindata_pos_neg/part-00000')
    #print data["label"][:10]
    #print data["use_sex"][:10]
    #print data["user_age"][:10]
    #print data["user_province"][:10]
    #print data["user_city"][:10]
    #print data["user_category"][:10]
    #print data["user_tag"][:10]
    #print data["user_docid"][:10]
    #print data["user_recent_tag"][:10]
    #print data["user_recent_category"][:10]
    #print data["doc_tag"][:10]
    #print data["doc_category"][:10]
    #print data["doc_docid"][:10]
    #exit(0)
    #get_all_doc_info_from_traindata("/mnt/yardcephfs/mmyard/g_wxg_fd_search/jackqiu/user_embed_data/b.tfrecords","20180805")
    #exit(0)

    index = 0
    keys = ["label", "business_type","use_sex","user_province", "user_age", "user_city", "video_docids", "mini_docids", "mp_docids", "doc_docid", "short_tag", "short_cate", "mini_tag", "mini_cate", "mp_docid_tag", "mp_docid_cate"]

    for f in batch_iter_multi(r"./train_data/part*", 100):
        for i in range(len(f["label"])):
            instance = list()
            for key in keys:
                if key in f:
                    if type(f[key][i]) ==np.ndarray or type(f[key][i]) == type(list()):
                        instance.append("\003".join([str(x) for x in f[key][i]]))
                    else:
                        instance.append(str(f[key][i]))
                else:
                    instance.append("")
            print("\t".join(instance))
