import tensorflow as tf
import sys
from tensorflow.contrib import layers
import numpy as np
from ModelConfig import ModelConfig
from modules import encode
from t2t_req.src.train import gelu,attn


def full_connected_layer_auto_reuse(input, W_size, b_size, w_name, b_name):
    with tf.variable_scope("full_connected_layer", reuse=tf.AUTO_REUSE):
        W = tf.get_variable(name=w_name, shape=W_size, initializer=tf.contrib.layers.xavier_initializer())
        # W = tf.Variable(tf.random_uniform(W_size, -1, 1))
        b = tf.Variable(tf.constant(0.1, shape=b_size),name=b_name)
        output = tf.nn.xw_plus_b(input, W, b)
    return W, b, output

def full_connect_layer(input,output_size,name_prefix="full_connect"):
    with tf.variable_scope("full_connected_layer", reuse=tf.AUTO_REUSE):
        input_shape=input.shape
        input_size=int(input_shape[-1])
        origin_input_shape = tf.shape(input)
        w_shape = [input_size, output_size]
        W = tf.get_variable(name=name_prefix+"_w", shape=w_shape, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name=name_prefix+"_b") 
        new_input = tf.reshape(input, [-1,input_size])
        output = tf.nn.xw_plus_b(new_input, W, b)
        output = tf.reshape(output, tf.concat([origin_input_shape[:-1], [output_size]], 0))
        return output
def full_connect_layer(variable_map, input,output_size,name_prefix="full_connect"):
    with tf.variable_scope("full_connected_layer", reuse=tf.AUTO_REUSE):
        input_shape=input.shape
        input_size=int(input_shape[-1])
        origin_input_shape = tf.shape(input)
        w_shape = [input_size, output_size]
        W = tf.get_variable(name=name_prefix+"_w", shape=w_shape, initializer=tf.contrib.layers.xavier_initializer())
        variable_map[name_prefix+"_w"] = W 
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name=name_prefix+"_b") 
        variable_map[name_prefix+"_b"] = b 
        new_input = tf.reshape(input, [-1,input_size])
        output = tf.nn.xw_plus_b(new_input, W, b)
        output = tf.reshape(output, tf.concat([origin_input_shape[:-1], [output_size]], 0)) 
        return output

def cosine(input_a,input_b):
    square1 = tf.sqrt(tf.reduce_sum(tf.square(input_a), axis=1))
    square2 = tf.sqrt(tf.reduce_sum(tf.square(input_b), axis=1))
    inner_product = tf.reduce_sum(tf.multiply(input_a, input_b), axis=1)
    inner_product = tf.divide(inner_product, square1)
    inner_product = tf.divide(inner_product, square2)
    return inner_product

class CrossModel(object):

    def _embedding_group__(self, embeddings, inputs, position_inputs=None):
        with tf.device("/cpu:0"):
            input_embeddings = tf.nn.embedding_lookup(embeddings, inputs) #b*m*n
            if position_inputs is not None:
                pos_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_inputs)
                input_embeddings = tf.add(input_embeddings, position_embeddings)
        with tf.device("/cpu:0"):
            input_weights = tf.to_float(tf.abs(inputs) > 0) #b*m
            weight_sum_sequence = tf.reduce_sum(input_weights, axis=1, keep_dims=True)+0.01 #b*1

            input_weights = tf.expand_dims(input_weights,-1)# b*m*1

            input_embeddings = tf.multiply(input_embeddings, input_weights) #b*m*n
            input_embeddings = tf.reduce_sum(input_embeddings, axis=1) #b*n
            input_embeddings = tf.div(input_embeddings, weight_sum_sequence)
        return input_embeddings
    
    def attention_aggregate(self, embeddings, inputs, name_prefix="", position_inputs=None):
        with tf.device("/cpu:0"):
            input_embeddings = tf.nn.embedding_lookup(embeddings, inputs) #b*m*n
            if position_inputs is not None:
                pos_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_inputs)
                input_embeddings = tf.add(input_embeddings, pos_embeddings)
        with tf.device("/cpu:0"):
            with tf.name_scope("attention_aggregate"):
                input_weights = tf.expand_dims(tf.to_float(tf.abs(inputs)>0),-1) #b*m*1
                u_context = tf.Variable(tf.truncated_normal([self.config.attention_size]),name=name_prefix+"_u_context")
                h = full_connect_layer(input_embeddings, self.config.attention_size, name_prefix=name_prefix)
                h = tf.nn.tanh(h)
                self.h = h
                hu_sum = tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True)
                exp = tf.exp(hu_sum)
                exp_adapt = tf.multiply(exp, input_weights)
                exp_adapt_sum = tf.reduce_sum(exp_adapt, axis=1, keep_dims=True)+0.001
                  
                alpha = tf.div(exp_adapt, exp_adapt_sum)

                # [batch_size, embedding_size]
                attention_output = tf.reduce_sum(tf.multiply(input_embeddings, alpha), axis=1)
        return attention_output
    def context_attention(self, inputs, train_weight_reshape,group_type_inputs, business_type_input, name_prefix="", aggregate_mode="sum"):
        with tf.device("/cpu:0"):
            with tf.name_scope("attention_layer"):
                bus_embeddings = tf.nn.embedding_lookup(self.business_type_embeddings, business_type_input) #b*n
                bus_embeddings = tf.expand_dims(bus_embeddings,[1]) #b*1*n
                group_type_embed = inputs
                u_context = tf.Variable(tf.truncated_normal([self.config.attention_size]), name=name_prefix+"_u_context")
                new_inputs_part1 = group_type_embed-bus_embeddings
                new_inputs_part2 = group_type_embed*bus_embeddings
                bus_tile = tf.tile(tf.squeeze(bus_embeddings, 1), [1, inputs.get_shape().as_list()[1]])
                bus_tile = tf.reshape(bus_tile, [-1, inputs.get_shape().as_list()[1], inputs.get_shape().as_list()[-1]])
                new_inputs = tf.concat([bus_tile, group_type_embed, new_inputs_part1,new_inputs_part2],axis=2)
                
                h = full_connect_layer(self.variable_map, new_inputs, self.config.attention_size, name_prefix=name_prefix+"f1")
                h = tf.nn.tanh(h)
                h = full_connect_layer(self.variable_map, h, 32 , name_prefix=name_prefix+"f2")
                h = tf.nn.tanh(h)
                h = full_connect_layer(self.variable_map, h, 1, name_prefix=name_prefix+"f3")
                h = tf.nn.tanh(h)
                exp = tf.exp(h)
                exp_adapt = tf.multiply(exp, train_weight_reshape)
                exp_adapt_sum = tf.reduce_sum(exp_adapt, axis=1, keep_dims=True)+0.001
                alpha = tf.div(exp_adapt, exp_adapt_sum)
                self.group_alpha = alpha
                if aggregate_mode=="sum":
                    attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
                else:
                    attention_output = tf.multiply(inputs, alpha)


        return attention_output  
     
    def _create_rnn_cell(self, hidden_size, n_layers, cell_type):
        w_initializer = tf.random_uniform_initializer(-0.05, 0.05)
        b_initializer = tf.random_uniform_initializer(-0.01, 0.01)        
    	if cell_type == "GRU":
    	    self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size, kernel_initializer = w_initializer, bias_initializer = b_initializer) for _ in range(n_layers)])
    	elif cell_type == "LSTM":
    	    self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_size, kernel_initializer = w_initializer, bias_initializer = b_initializer) for _ in range(n_layers)])
    	else:
    	    raise NotImplementedError
    def rnn_aggregate(self, embeddings, inputs, name_prefix="", position_inputs=None):
        with tf.device("/cpu:0"):
            input_embeddings = tf.nn.embedding_lookup(embeddings, inputs) #b*m*n
            if position_inputs is not None:
                pos_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_inputs)
                input_embeddings = tf.add(input_embeddings, position_embeddings)

        with tf.device("/cpu:0"):
            input_weights = tf.to_float(tf.abs(inputs) > 0) #b*m
            weight_sum_sequence = tf.reduce_sum(input_weights, axis=1)#b*1 seq_len
            input_weights = tf.expand_dims(input_weights,-1)# b*m*1
            input_embeddings = tf.multiply(input_embeddings, input_weights) #b*m*n
            with tf.device("/cpu:0"): 
            	outputs, state = tf.nn.dynamic_rnn(self.cell, input_embeddings, weight_sum_sequence, dtype=tf.float32) 
        return outputs[:,-1,:]
    def cnn_aggregate(self, embeddings, inputs, name_prefix="", sequence_length=5, filter_size=3, num_filters=64, position_inputs=None):
        with tf.device("/cpu:0"):
        	input_embeddings = tf.nn.embedding_lookup(embeddings, inputs) #b*m*n
        	embedding_size = int(input_embeddings.shape[-1])
            #position embedding size should be the same with docid_embeddings
        	if position_inputs is not None:
        	    pos_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_inputs)
        	    input_embeddings = tf.add(input_embeddings, position_embeddings)
        	embed_inputs_expands = tf.expand_dims(input_embeddings,-1) #b*m*n*1
        	with tf.variable_scope("conv_layer"):
        	    filter_shape = [filter_size,embedding_size,1,num_filters]
        	    W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name=name_prefix+"conv_W")
        	    b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name=name_prefix+'conv_b')
        	    conv_output = tf.nn.conv2d(embed_inputs_expands, W, [1, 1, 1, 1], padding="VALID", name=name_prefix+"conv1")
        	    conv_hidden = tf.nn.relu(tf.nn.bias_add(conv_output, b))
        	with tf.variable_scope("max_pooling_layer"):
        	    pooled_query = tf.nn.max_pool(conv_hidden,ksize=[1,sequence_length-filter_size+1,1,1],\
        	                                  strides=[1,1,1,1], padding="VALID",name=name_prefix+'pool')
        	pooled_query_flat = tf.reshape(pooled_query, [-1, num_filters])
        	return pooled_query_flat
        
            
    def attention_layer(self, inputs, train_weight_reshape, name_prefix="", aggregate_mode="sum"):
        with tf.device("/cpu:0"):
            with tf.name_scope("attention_layer"):
                u_context = tf.Variable(tf.truncated_normal([self.config.attention_size]), name=name_prefix+"_u_context")

                h = full_connect_layer(inputs, self.config.attention_size, name_prefix=name_prefix)
                h = tf.nn.tanh(h)

                hu_sum = tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True)
                exp = tf.exp(hu_sum)
                exp_adapt = tf.multiply(exp, train_weight_reshape)
                exp_adapt_sum = tf.reduce_sum(exp_adapt, axis=1, keep_dims=True)+0.001
                alpha = tf.div(exp_adapt, exp_adapt_sum)

                # [batch_size, embedding_size]
                if aggregate_mode=="sum":
                    attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
                else:
                    attention_output = tf.multiply(inputs, alpha)
        return attention_output 
    
    def residual_layer(self, x, inputs, name="residual"):
        return tf.add(x,inputs,name=name)

    def _embedding__(self, embeddings, inputs):
        with tf.device("/cpu:0"):
            return tf.nn.embedding_lookup(embeddings,inputs)
			
    def random_softmax_loss(self, nb_negative, inputs, targets, biases=None, business_type=1):
        weights = self.vid_embeddings
        if business_type == 3:
            weights = self.mp_docid_embeddings
        nb_classes, real_batch_size = tf.shape(weights)[0], tf.shape(targets)[0]
        targets = tf.cast(targets,dtype=tf.int32)
        negative_sample = tf.random_uniform([real_batch_size, nb_negative], 1,self.config.mini_video_count+1 , dtype=tf.int32)
        if business_type == 2:
            negative_sample = tf.random_uniform([real_batch_size, nb_negative], self.config.mini_video_count+1,self.config.vocab_size , dtype=tf.int32)
        elif business_type == 3:
            negative_sample = tf.random_uniform([real_batch_size, nb_negative], 1,self.config.mp_vocab_size + 1 , dtype=tf.int32)
        random_sample = tf.concat([targets, negative_sample], axis=1)
        sampled_weights = tf.nn.embedding_lookup(weights, random_sample)
        with tf.device("/cpu:0"):
            if biases:
                sampled_biases = tf.nn.embedding_lookup(biases, random_sample)
                self.sampled_logits = tf.matmul(inputs, sampled_weights, transpose_b=True)[:,0,:] + sampled_biases
            else:
                self.sampled_logits = tf.matmul(inputs, sampled_weights, transpose_b=True)[:,0,:]
                sampled_labels = tf.zeros([real_batch_size], dtype=tf.int32)
                loss_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.sampled_logits, labels=sampled_labels))
        return loss_mean
			
    def _build_user_graph(self, config, input, mode='user', activation="tanh"):
        shape_list = input.get_shape().as_list()
        init_dim = shape_list[-1]
        dims_description = config.user_dims
        if mode=='doc':
            dims_description = config.doc_dims
        final_output = input
        self.user_W = []
        self.user_b = []
        self.doc_W = []
        self.doc_b = []

        for i in range(len(dims_description)):
            dim = dims_description[i]
            last_dim = init_dim
            if i > 0:
                last_dim = dims_description[i-1]
            W_size = [last_dim, dim]
            b_size = [dim]
            W, b, output = full_connected_layer_auto_reuse(final_output, W_size=W_size, b_size=b_size,
                                                           w_name = mode+"_layer_w_"+str(i),
                                                           b_name = mode+"_layer_b_"+str(i))
            if mode == "user":
                self.user_W.append(W)
                self.user_b.append(b)
            elif mode == "doc":
                self.doc_W.append(W)
                self.doc_b.append(b)
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            if activation == "tanh":
                final_output = tf.nn.tanh(output)
            elif activation == "relu":
                final_output = tf.nn.relu(output)
            else:
                final_output = output

        return final_output

    def docid_aggregate(self, vid_embeddings, inputs, position_inputs, name_prefix="prefix"):
        if self.config.history_aggregate_mode == 2: #attention_pooling
            user_docid_embedding = self.attention_aggregate(vid_embeddings, inputs, name_prefix=name_prefix,
                 position_inputs = position_inputs)
        elif self.config.history_aggregate_mode == 1: #sum pooling
            user_docid_embedding = self._embedding_group__(vid_embeddings, inputs)
        elif self.config.history_aggregate_mode == 3: #rnn 
            user_docid_embedding = self.rnn_aggregate(vid_embeddings, inputs, name_prefix=name_prefix, \
                position_inputs = position_inputs)
        elif self.config.history_aggregate_mode == 4:
            #embeddings, inputs, name_prefix="", filter_size=3, num_filters=64, position_inputs=None)
            user_docid_embedding = self.cnn_aggregate(vid_embeddings, inputs, name_prefix=name_prefix, \
               sequence_length=self.config.max_user_vid, filter_size=self.config.filter_size, \
               num_filters=self.config.num_filters, position_inputs=position_inputs) 
        else:
            user_docid_embedding = self._embedding_group__(vid_embeddings, inputs)
        return user_docid_embedding


    def __init__(self,config):

        self.label_placeholder = tf.placeholder(tf.float32, [None], name='label_placeholder')
        #self.label_placeholder = tf.placeholder(tf.int64, [None], name='label_placeholder')

        #user_information
        self.user_sex_placeholder = tf.placeholder(tf.int64, [None], name='sex_placeholder')
        self.business_placeholder = tf.placeholder(tf.int64, [None], name='business_placeholder')
        self.user_age_placeholder = tf.placeholder(tf.int64, [None], name='age_placeholder')
        self.user_provice_placeholder = tf.placeholder(tf.int64, [None], name='province_placeholder')
        self.user_city_placeholder = tf.placeholder(tf.int64, [None], name='city_placeholder')
        self.video_subcategory_placeholder = tf.placeholder(tf.int64, [None, config.max_user_subcategory], name = 'video_subcategory')
        self.video_tag_placeholder = tf.placeholder(tf.int64, [None, config.max_user_tag], name='video_tag_placeholder')
        self.video_longtag_placeholder = tf.placeholder(tf.int64, [None, config.max_user_tag], name='video_longtag_placeholder')
        self.video_shorttag_placeholder = tf.placeholder(tf.int64, [None, config.max_user_tag], name='video_shorttag_placeholder')
        self.mp_shorttag_placeholder = tf.placeholder(tf.int64, [None, config.max_user_tag], name='mp_shorttag_placeholder')
        self.mp_longtag_placeholder = tf.placeholder(tf.int64, [None, config.max_user_tag], name='mp_longtag_placeholder')
        self.short_video_placeholder = tf.placeholder(tf.int64, [None, config.max_user_vid], name='short_video_placeholder')
        self.mini_video_placeholder = tf.placeholder(tf.int64, [None, config.max_user_vid], name='mini_video_placeholder')
        self.mp_placeholder = tf.placeholder(tf.int64, [None, config.max_user_vid], name='mp_placeholder')
        self.short_video_tag_placeholder = tf.placeholder(tf.int64, [None, config.max_user_recent_tag], name='shortvideo_tag')
        self.short_video_cate_placeholder = tf.placeholder(tf.int64, [None, config.max_user_vid], name='shortvideo_cate')
        self.mini_video_tag_placeholder = tf.placeholder(tf.int64, [None, config.max_user_recent_tag], name='minivideo_tag')
        self.mini_video_cate_placeholder = tf.placeholder(tf.int64, [None, config.max_user_vid], name='minivideo_cate')
        self.mp_docid_tag_placeholder = tf.placeholder(tf.int64, [None, config.max_user_recent_tag], name='mp_docid_tag')
        self.mp_docid_cate_placeholder = tf.placeholder(tf.int64, [None, config.max_user_vid], name='mp_docid_cate')



        self.is_train = tf.placeholder(tf.int64, None, name="is_train")
        self.variable_map = dict()

        
        self.config = config

        # keeping track of l2 regularization loss
        self.l2_loss = tf.constant(0.0)

        #doc_information
        self.doc_vid_placeholder = tf.placeholder(tf.int64, [None], 'doc_vid')

        self.vid_embeddings = tf.Variable(tf.random_uniform([config.vocab_size + 1, config.vid_embed_size], -0.05, 0.05))
        #with tf.device
        
        with tf.name_scope(name="embedding_init"), tf.device("/cpu:0"):
        	#rnn cell
            self._create_rnn_cell(config.vid_embed_size, 1, "GRU")
            self.video_subcate_embeddings = tf.Variable(
                tf.random_uniform([config.video_subcategory_size + 1, config.video_subcategory_embed_size], -0.05, 0.05))
            self.video_tag_embeddings = tf.Variable(
                tf.random_uniform([config.video_tag_size + 1, config.video_tag_embed_size], -0.05, 0.05))
            self.age_embeddings = tf.Variable(
                tf.random_uniform([config.age_size + 1, config.age_embed_size], -0.05, 0.05))
            self.sex_embeddings = tf.Variable(
                tf.random_uniform([config.sex_size + 1, config.sex_embed_size], -0.05, 0.05))
            self.mp_tag_embeddings = tf.Variable(
                tf.random_uniform([config.mp_tag_size + 1, config.mp_tag_embed_size], -0.05, 0.05))
            self.mp_cate_embeddings = tf.Variable(tf.random_uniform([512, 64], -0.05, 0.05))
            self.province_embeddings = tf.Variable(
                tf.random_uniform([config.province_size + 1, config.province_embed_size], -0.05, 0.05))
            self.city_embeddings = tf.Variable(
                tf.random_uniform([config.city_size + 1, config.city_embed_size], -0.05, 0.05))
            self.position_embeddings = tf.Variable(
                tf.random_uniform([config.time_windows + 1, config.vid_embed_size], -0.05, 0.05))
            self.mp_docid_embeddings = tf.Variable(
                tf.random_uniform([config.mp_vocab_size + 1, config.vid_embed_size], -0.05, 0.05))
            self.business_type_embeddings = tf.Variable(
                tf.random_uniform([5, config.vid_embed_size], -0.05, 0.05))
            self.group_type_embeddings = tf.Variable(
                tf.random_uniform([20, config.vid_embed_size], -0.05, 0.05))
             

        with tf.name_scope(name="embedding_init"):
            user_sex_embedding = self._embedding__(self.sex_embeddings, self.user_sex_placeholder)
            user_age_embedding = self._embedding__(self.age_embeddings, self.user_age_placeholder)
            user_province_embedding = self._embedding__(self.province_embeddings, self.user_provice_placeholder)
            user_city_embedding = self._embedding__(self.city_embeddings, self.user_city_placeholder)
            #user_docid_embedding = self._embedding_group__(self.vid_embeddings, self.user_history_vid_placeholder)
            if self.config.use_position: 
                self.position_inputs = tf.tile(tf.expand_dims(tf.range(1,config.time_windows+1,dtype=tf.int64),[0]) , [tf.shape(user_sex_embedding)[0],1])
            else:
                self.position_inputs = None
            
            self.group_type_inputs = tf.tile(tf.expand_dims(tf.range(1,15,dtype=tf.int64),[0]) , [tf.shape(user_sex_embedding)[0],1])
            short_video_docid_embedding = self.docid_aggregate(self.vid_embeddings, self.short_video_placeholder, self.position_inputs, "video_history")
            mini_video_docid_embedding = self.docid_aggregate(self.vid_embeddings, self.mini_video_placeholder, self.position_inputs, "mini_history")
            mp_docid_embedding = self.docid_aggregate(self.mp_docid_embeddings, self.mp_placeholder, self.position_inputs, name_prefix="mp_history")
            short_video_tag_embedding = self._embedding_group__(self.video_tag_embeddings, self.short_video_tag_placeholder)
            short_video_cate_embedding = self._embedding_group__(self.video_subcate_embeddings, self.short_video_cate_placeholder)
            mini_video_tag_embedding = self._embedding_group__(self.video_tag_embeddings, self.mini_video_tag_placeholder)
            mini_video_cate_embedding = self._embedding_group__(self.video_subcate_embeddings, self.mini_video_cate_placeholder)
            mp_docid_tag_embedding = self._embedding_group__(self.mp_tag_embeddings, self.mp_docid_tag_placeholder)
            mp_docid_cate_embedding = self._embedding_group__(self.mp_cate_embeddings, self.mp_docid_cate_placeholder)
            doc_vid_embedding = self._embedding__(self.vid_embeddings, self.doc_vid_placeholder)

        
        with tf.device("/cpu:0"):
            user_embedding_concat_part1 = tf.concat([user_sex_embedding,
                                               user_age_embedding,
                                               user_province_embedding,
                                               user_city_embedding,
                                               ],
                                            axis= -1, name="concat_user_profile_item")
            user_embedding_concat_part2 = tf.concat([
                                               mini_video_docid_embedding,
                                               short_video_docid_embedding,
                                               mp_docid_embedding,
                                               mp_docid_tag_embedding,
                                               mp_docid_cate_embedding,
                                               short_video_tag_embedding,
                                               short_video_cate_embedding,
                                               mini_video_tag_embedding,
                                               mini_video_cate_embedding
                                               ],
                                            axis= -1, name="concat_user_profile_item")
            if self.config.use_group_attention: 
            	group_size = user_embedding_concat_part2.get_shape().as_list()[-1]/64
            	user_embedding_group=tf.reshape(user_embedding_concat_part2, [-1,group_size,64])
            	batch_size = tf.shape(user_embedding_group)[0]
            	group_weight = tf.ones([batch_size,group_size,1])
                business_type = self.business_placeholder
                user_embedding_attention = self.context_attention(user_embedding_group, group_weight,self.group_type_inputs, business_type, name_prefix="group_attention", aggregate_mode="none")
                attentin_weights ,user_embedding_attention = attn(user_embedding_attention, "test", 64, 8, False)
            	user_embedding_attention = tf.reshape(user_embedding_attention,[-1,group_size*64]) 
                user_embedding_concat = tf.concat([user_embedding_concat_part1, user_embedding_attention], axis=-1, name="test_part")

            doc_embedding_concat = doc_vid_embedding							  
            with tf.name_scope("user_doc_embedding"):
                user_final_embedding = self._build_user_graph(config, user_embedding_concat, 'user', self.config.user_activation)
                doc_final_embedding = doc_vid_embedding

            

            with tf.name_scope("loss"):
                user_final_embedding = tf.expand_dims(user_final_embedding, 1)
                labels = tf.expand_dims(self.doc_vid_placeholder,-1)
                self.loss = self.random_softmax_loss(20, user_final_embedding, labels, None, self.business_placeholder[0])







