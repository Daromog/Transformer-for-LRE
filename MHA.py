

#_________________________________________________________________________________________________________

# This code implements the entire transformer architecture, and  sliding attention windows

#___________________________________________________________________________________________________________




#Librerias ---------------------------

import Load_Database
import tensorflow as tf
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True) #Configuracion para la GPU
tf.random.set_seed(6789) #Se establece un random seed para obtener resultados constantes
TF_ENABLE_GPU_GARBAGE_COLLECTION=False




# Se importa la base de datos ------

print()
print()
print("Getting Database...")

path_to_files=r'C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\KALAKA-3\Allosaurus_Phones\1024_Trigramas_Splitted'
input_tensor_train,labels_train,input_tensor_dev,labels_dev,input_tensor_ev,labels_ev,lang_tokenizer,labels_names_train,labels_names_dev, labels_names_eval=Load_Database.load_dataset(path_to_files)




# Parametros del modelo -------

Save_Prob=13

BUFFER_SIZE=len(input_tensor_train)  #Ejemplos a tomar para hacer un shuffle
BATCH_SIZE=32
EPOCHS=25
input_vocab_size=len(lang_tokenizer.get_vocab())+1
target_classes=6
maximum_position_encoding=1024

num_layers = 1
d_model = 32
num_heads = 1
dropout_rate = 0.4


# Se realiza el batch y el shuffle para la base de datos ------

print()
print("Vocab Size")
print(input_vocab_size)
print()


dataset_train=tf.data.Dataset.from_tensor_slices((input_tensor_train,labels_train,labels_names_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=False)
dataset_dev=tf.data.Dataset.from_tensor_slices((input_tensor_dev,labels_dev,labels_names_dev)).batch(BATCH_SIZE,drop_remainder=False)
dataset_eval=tf.data.Dataset.from_tensor_slices((input_tensor_ev,labels_ev,labels_names_eval)).batch(BATCH_SIZE,drop_remainder=False)


print()
print("Database Loaded")
print()




# Se crea los paths y summaries para guardar y graficar el modelo------

current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir='logs/gradient_tape/' + current_time + '/train'
test_log_dir='logs/gradient_tape/' + current_time + '/test'
eval_log_dir='logs/gradient_tape/' + current_time + '/eval'
train_summary_writer=tf.summary.create_file_writer(train_log_dir)
test_summary_writer=tf.summary.create_file_writer(test_log_dir)
eval_summary_writer=tf.summary.create_file_writer(eval_log_dir)




# Positional Encodings -------

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position,d_model):
	angle_rads = get_angles(np.arange(position)[:,np.newaxis],
							np.arange(d_model)[np.newaxis,:],
							d_model)

	#apply sin to even indices in the array
	angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])

	#apply cos to odd indices in the array
	angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])

	pos_encoding = angle_rads[np.newaxis, ...]

	return tf.cast(pos_encoding,dtype=tf.float32)



# Masking ---------

def create_look_ahead_mask(size1,size2,zeros):
    mask = 1 -tf.linalg.band_part(tf.ones((size1, size2)), 0, zeros)
    return mask  # (seq_len, seq_len)

def create_padding_mask(seq,window,size,init_zeros):

	seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
	seq_2=tf.math.equal(seq, 0)

	#......

	padding_window=tf.zeros([1,size,size])

	for j in range(seq.shape[0]):

		tmp_indices = tf.where(tf.equal(seq[j,:], 1))

		if tf.equal(tf.size(tmp_indices), 0) == False:
			if size-tmp_indices.shape[0]<init_zeros+1:
				mask=tf.repeat(seq[j,:][tf.newaxis,:],repeats=size,axis=0)
			else:
				slice=window[size-(init_zeros+1+tmp_indices.shape[0]):,:]
				cut_ones=tf.ones([slice.shape[0],tmp_indices.shape[0]])
				slice=slice[:,0:size-tmp_indices.shape[0]]
				slice=tf.concat([slice,cut_ones],axis=1)
				slice=slice[0:slice.shape[0]-tmp_indices.shape[0],:]
				slice=tf.concat([slice,tf.repeat(tf.ones([1,size]),repeats=tmp_indices.shape[0],axis=0)],axis=0)
				mask=tf.concat([window[:size-(init_zeros+1+tmp_indices.shape[0]),:],slice],axis=0)
				#mask=tf.concat([seq[j,:][tf.newaxis,:],mask[1:,:]],axis=0)
		else:
			mask=window

		padding_window=tf.concat([padding_window,mask[tf.newaxis,:,:]],0)

	padding_window=padding_window[1:,:,:]

	#.....

	# add extra dimensions to add the padding
	# to the attention logits
	return padding_window[:,tf.newaxis,:,:], seq_2 #(batch_size, 1, size, size)




 # Scaled - Dot Product Attention----------

def scaled_dot_product_attention(q, k, v, mask):

	matmul_qk = tf.matmul(q, k, transpose_b=True) #(... , seq_len_q, seq_len_k)

	#scale matmul_qk
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	#add the mask to the scaled tensor
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)

	# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) #(batch, num_heads, seq_len_q,seq_len_k)
	output = tf.matmul(attention_weights, v) #(..., seq_len_q, depth_v)

	return output,attention_weights




# Multi-Head Attention ------------


class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self,d_model,num_heads):
		super(MultiHeadAttention,self).__init__()

		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0 

		self.depth = d_model // self.num_heads

		self.wqw = tf.keras.layers.Dense(d_model)
		self.wkw = tf.keras.layers.Dense(d_model)
		self.wvw = tf.keras.layers.Dense(d_model)


		self.dense = tf.keras.layers.Dense(d_model)

	def split_heads(self, x, batch_size):
		#Split the last dimension into (num_heads,depth) - multiple heads
		#Transpose the result such (batch_size,num_heads,seq_len,depth)

		x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
		return tf.transpose(x,perm=[0, 2, 1, 3])

	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]

		q = self.wqw(q) #(batch_size, seq_len, d_model)
		k = self.wkw(k) #(batch_size, seq_len, d_model)
		v = self.wvw(v) #(batch_size, seq_len, d_model)


		q = self.split_heads(q, batch_size) #(batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size) #(batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size) #(batch_size, num_heads, seq_len_v, depth)

		#scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		#attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) #(batch_size, seq_len_q, num_heads, depth) - (64, 500, 4, 38)

		concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) #(batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention) #(batch_size, seq_len_q, d_model)

		return output, attention_weights



# Encoder Layer -----------

#def point_wise_feed_forward_network(d_model, dff):
#  return tf.keras.Sequential([
#      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
#      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
#  ])

class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, rate):
		super(EncoderLayer,self).__init__()

		self.mha = MultiHeadAttention(d_model, num_heads)
		#self.ffn = point_wise_feed_forward_network(d_model, 64)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		#self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = tf.keras.layers.Dropout(rate)
		#self.dropout2 = tf.keras.layers.Dropout(rate)

	def call(self, x, training, mask):

		attn_output, attention_weights = self.mha(x, x, x, mask) #(batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output,training=training)
		out1 = self.layernorm1(x + attn_output) #(batch_size, input_seq_len, d_model)

		#ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
		#ffn_output = self.dropout2(ffn_output, training=training)
		#out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

		return out1, attention_weights



# Encoder ------------


class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, input_vocab_size, maximum_position_encoding, rate):
		super(Encoder,self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers

		self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
		#embeddings_initializer=tf.keras.initializers.Constant(init_embeddings)
		self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

		self.enc_layers = [EncoderLayer(d_model, num_heads, rate)
						   for _ in range(num_layers)]

		self.dropout = tf.keras.layers.Dropout(rate)

	def call (self, x, training, mask):

		seq_len = tf.shape(x)[1]

		#adding embedding and position encoding
		x = self.embedding(x) #(batch_size, input_seq_len, d_model)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) #Normalization
		x += self.pos_encoding[:, :seq_len, :]

		x = self.dropout(x, training=training)

		for i in range(self.num_layers):
			x,attention_weights = self.enc_layers[i](x, training, mask)

		return x,attention_weights #(batch_size, input_seq_len, d_model)



# TRANSFORMER ----------

class Transformer(tf.keras.Model):
	def __init__(self, num_layers, d_model, num_heads, input_vocab_size, target_classes, maximum_position_encoding, rate):
		super(Transformer, self).__init__()

		self.encoder = Encoder(num_layers, d_model, num_heads, input_vocab_size, maximum_position_encoding, rate)

		self.final_layer = tf.keras.layers.Dense(target_classes)

		self.pool = tf.keras.layers.GlobalAveragePooling1D()

		self.dropout1 = tf.keras.layers.Dropout(rate)

	def call(self, inp, training, enc_padding_mask, pooling_mask):

		enc_output, attention_weights = self.encoder(inp, training, enc_padding_mask) #(batch_size, inp_seq_len, d_model)

		enc_output = self.dropout1(self.pool(enc_output,mask=pooling_mask),training=training)
		#enc_output = self.dropout1(enc_output[:,0,:],training=training)

		final_output = self.final_layer(enc_output) #(batch_size, target_classes)

		final_output = tf.keras.activations.softmax(final_output,axis=-1)

		return final_output,attention_weights

transformer = Transformer(num_layers, d_model, num_heads,
						  input_vocab_size, target_classes,
						  maximum_position_encoding, dropout_rate)




# Optimizer --------


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps = 4000):
		super(CustomSchedule,self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self,step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)
		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)




# Loss Function and Metrics -------


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

def loss_function(real,pred):
	loss = loss_object(real,pred)
	return loss

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

eval_loss = tf.keras.metrics.Mean(name = 'eval_loss')
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'eval_accuracy')




# Training and Testing Graphs  --------

train_step_signature = [tf.TensorSpec(shape=(None,None), dtype=tf.int32),
						tf.TensorSpec(shape=(None,), dtype=tf.int32),
						tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
						tf.TensorSpec(shape=(None,None), dtype=tf.bool)]

@tf.function(input_signature = train_step_signature)
def train_step(inp, tar,enc_padding_mask,pooling_mask):

	with tf.GradientTape() as tape:
		predictions,attention_weights = transformer(inp, True, enc_padding_mask, pooling_mask)

		loss = loss_function(tar, predictions)

	gradients = tape.gradient(loss, transformer.trainable_variables)
	optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

	train_loss(loss)
	train_accuracy(tar, predictions) 

	return predictions

test_step_signature = [tf.TensorSpec(shape=(None,None), dtype=tf.int32),
						tf.TensorSpec(shape=(None,), dtype=tf.int32),
						tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
						tf.TensorSpec(shape=(None,None), dtype=tf.bool)]

@tf.function(input_signature = test_step_signature)
def test_step(inp, tar,enc_padding_mask,pooling_mask):

	predictions, attention_weights = transformer(inp, False, enc_padding_mask, pooling_mask)

	loss = loss_function(tar, predictions)

	test_loss(loss)
	test_accuracy(tar, predictions) 

	return predictions


eval_step_signature = [tf.TensorSpec(shape=(None,None), dtype=tf.int32),
						tf.TensorSpec(shape=(None,), dtype=tf.int32),
						tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
						tf.TensorSpec(shape=(None,None), dtype=tf.bool)]

@tf.function(input_signature = eval_step_signature)
def eval_step(inp, tar,enc_padding_mask,pooling_mask):

	predictions, attention_weights = transformer(inp, False, enc_padding_mask, pooling_mask)

	loss = loss_function(tar, predictions)

	eval_loss(loss)
	eval_accuracy(tar, predictions) 

	return predictions



# Training and testing loops ---------

# Windows Masking ......
size=1024
init_zeros=64
mid_zeros=init_zeros*2

central_1 = create_look_ahead_mask(size,size,init_zeros)
central=central_1[0:init_zeros+1,init_zeros+1:]
central_1=0
zeros=tf.zeros([init_zeros+1,init_zeros+1])
init=tf.concat([zeros, central], 1)
central=0
zeros=0
central_2 = create_look_ahead_mask(size,size,mid_zeros)
window=tf.concat([init,central_2[1:size-init_zeros,:]],0)
#window=tf.concat([tf.zeros([size,1]),window[:,1:]],axis=1)
init=0
central_2=0
#.......


for epoch in range(EPOCHS):

	train_loss.reset_states()
	train_accuracy.reset_states()
	test_loss.reset_states()
	test_accuracy.reset_states()
	eval_loss.reset_states()
	eval_accuracy.reset_states()


	n_t = []
	p_t = tf.zeros([1,6],tf.float32)
	for (batch,(inp_tr, tar_tr,names_train)) in enumerate(dataset_train):

		#----------------------------------------------------
		#inp_tr = tf.make_ndarray(tf.make_tensor_proto(inp_tr))
		#for i in range(inp_tr.shape[0]):
		#	if inp_tr[i,-1]!=2 and inp_tr[i,-1]!=0:
		#		inp_tr[i,-1]=2
		#inp_tr = tf.convert_to_tensor(inp_tr,dtype=tf.int32)
		#------------------------------------------------------

		enc_padding_mask, pooling_mask = create_padding_mask(inp_tr,window,size,init_zeros)
		predictions_train = train_step(inp_tr,tar_tr,enc_padding_mask,pooling_mask)
		if epoch+1 ==Save_Prob:
			names_train = tf.make_ndarray(tf.make_tensor_proto(names_train)).tolist()
			n_t = n_t + names_train
			predictions_train = tf.make_ndarray(tf.make_tensor_proto(predictions_train))
			p_t = tf.concat([p_t,predictions_train],0)


	n_d = []
	p_d = tf.zeros([1,6],tf.float32)
	for (batch,(inp_dev, tar_dev,names_dev)) in enumerate(dataset_dev):

		#----------------------------------------------------
		#inp_dev = tf.make_ndarray(tf.make_tensor_proto(inp_dev))
		#for i in range(inp_dev.shape[0]):
		#	if inp_dev[i,-1]!=2 and inp_dev[i,-1]!=0:
		#		inp_dev[i,-1]=2
		#inp_dev = tf.convert_to_tensor(inp_dev,dtype=tf.int32)
		#------------------------------------------------------

		enc_padding_mask, pooling_mask = create_padding_mask(inp_dev,window,size,init_zeros)
		predictions_dev = test_step(inp_dev,tar_dev,enc_padding_mask,pooling_mask)
		if epoch+1 ==Save_Prob:
			names_dev = tf.make_ndarray(tf.make_tensor_proto(names_dev)).tolist()
			n_d = n_d + names_dev
			predictions_dev = tf.make_ndarray(tf.make_tensor_proto(predictions_dev))
			p_d = tf.concat([p_d,predictions_dev],0)


	n_e = []
	p_e = tf.zeros([1,6],tf.float32)
	for (batch,(inp_eval, tar_eval,names_eval)) in enumerate(dataset_eval):

		#----------------------------------------------------
		#inp_eval = tf.make_ndarray(tf.make_tensor_proto(inp_eval))
		#for i in range(inp_eval.shape[0]):
		#	if inp_eval[i,-1]!=2 and inp_eval[i,-1]!=0:
		#		inp_eval[i,-1]=2
		#inp_eval = tf.convert_to_tensor(inp_eval,dtype=tf.int32)
		#------------------------------------------------------

		enc_padding_mask, pooling_mask = create_padding_mask(inp_eval,window,size,init_zeros)
		predictions_eval = eval_step(inp_eval,tar_eval,enc_padding_mask,pooling_mask)
		if epoch+1 ==Save_Prob:
			names_eval = tf.make_ndarray(tf.make_tensor_proto(names_eval)).tolist()
			n_e = n_e + names_eval
			predictions_eval = tf.make_ndarray(tf.make_tensor_proto(predictions_eval))
			p_e = tf.concat([p_e,predictions_eval],0)



	with train_summary_writer.as_default():
		tf.summary.scalar('loss',train_loss.result(),step=epoch)
		tf.summary.scalar('accuracy',train_accuracy.result(),step=epoch)

	with test_summary_writer.as_default():
		tf.summary.scalar('loss',test_loss.result(),step=epoch)
		tf.summary.scalar('accuracy',test_accuracy.result(),step=epoch)

	with eval_summary_writer.as_default():
		tf.summary.scalar('loss',eval_loss.result(),step=epoch)
		tf.summary.scalar('accuracy',eval_accuracy.result(),step=epoch)


	if epoch+1 == Save_Prob:
		#n_t = np.asarray(n_t)
		#np.save('names_train',tf.make_ndarray(tf.make_tensor_proto(n_t)))
		#np.save('probabilities_train',tf.make_ndarray(tf.make_tensor_proto(p_t[1:,:])))
		#n_d = np.asarray(n_d)
		#np.save('names_dev',tf.make_ndarray(tf.make_tensor_proto(n_d)))
		#np.save('probabilities_dev',tf.make_ndarray(tf.make_tensor_proto(p_d[1:,:])))
		#n_e = np.asarray(n_e)
		#np.save('names_eval',tf.make_ndarray(tf.make_tensor_proto(n_e)))
		#np.save('probabilities_eval',tf.make_ndarray(tf.make_tensor_proto(p_e[1:,:])))
		#print(tf.cast(tf.argmax(p_e[1:,:], axis=-1), tf.int32))
		#print(tf.one_hot(tf.cast(tf.argmax(p_e[1:,:], axis=-1), tf.int32), 6))
		scipy.io.savemat('probabilities_dev.mat',{"probabilities_dev":tf.make_ndarray(tf.make_tensor_proto(tf.one_hot(tf.cast(tf.argmax(p_d[1:,:], axis=-1), tf.int32), 6)))})
		scipy.io.savemat('probabilities_eval.mat',{"probabilities_eval":tf.make_ndarray(tf.make_tensor_proto(tf.one_hot(tf.cast(tf.argmax(p_e[1:,:], axis=-1), tf.int32), 6)))})


	template='Epoch {}, Perdida: {:.4f}, Exactitud: {:.4f}, Perdida de prueba: {:.4f}, Exactitud de prueba: {:.4f}, Perdida de Eval: {:.4f}, Exactitud de Eval : {:.4f}'
	print(template.format(epoch+1,
						  train_loss.result(),
						  train_accuracy.result()*100,
						  test_loss.result(),
						  test_accuracy.result()*100,
						  eval_loss.result(),
						  eval_accuracy.result()*100))


