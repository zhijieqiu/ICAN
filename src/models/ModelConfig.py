
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

class ModelConfig(object) :
    batch_size = 256
    trainable = False

    video_subcategory_size = 482
    video_subcategory_embed_size = 64
    video_tag_size = 125297
    video_tag_embed_size = 64

    vocab_size=2417330
    vid_embed_size=64
    
    mp_tag_size = 603192
    mp_tag_embed_size = 64
    mp_subcate_size = 492
    
    mp_vocab_size = 1021160

    sex_size = 4
    sex_embed_size = 64

    age_size=120
    age_embed_size =64

    province_size = 1500
    province_embed_size = 64

    city_size = 7000
    city_embed_size=64
    
    attention_size = 64


    max_doc_tag = 10
    max_user_subcategory = 15
    max_user_ctype = 10
    max_user_tag = 15
    max_user_vid = 5
    max_user_recent_subcategory = 10
    max_user_recent_tag = 20

    user_dims = [128, 64, 64]
    doc_dims = [128, 64, 64]

    learning_rate = 0.1
    l2_reg_lambda = 0.02
    time_windows = 5
    user_pretrained_vid_embedding = False

    use_residual = False
    use_group_attention = False
    use_position = False
    
    history_aggregate_mode = 0 #0 is default average pooling 1:sum pooling 2:attention_pooling 3:rnn_pooling 
    user_activation = "tanh"
    doc_activatin = "tanh"
    #cnn config
    filter_size = 3
    num_filters = 64
    mini_video_count = 30000
    use_fm = False

