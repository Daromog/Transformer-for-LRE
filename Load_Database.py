
#__________________________________________________________________________________________________

# Este programa importa la base de datos de KALAKA-3 para que sea usada por el modelo de Attention

#___________________________________________________________________________________________________




# Librerias --------------------------------------------

import tensorflow as tf
import os
import numpy as np
import io
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing





# Se importan los nombres y los labels de la base de datos original---------------------------

names_train=np.load(r'C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\Names_Labels\names_train.npy')
names_dev = np.load(r'C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\Names_Labels\names_dev.npy')
names_eval = np.load(r'C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\Names_Labels\names_eval.npy')

labels_train = np.load(r'C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\Names_Labels\labels_train.npy')
labels_dev = np.load(r'C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\Names_Labels\labels_dev.npy')
labels_eval = np.load(r'C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\Names_Labels\labels_eval.npy')




#Funciones ---------------------------------------------

# Esta funcion crea listas de los fonemas 

def create_dataset(parent_dir):

	lines_files_train=[]
	lines_files_dev=[]
	lines_files_ev=[]

	ph=["training","devel","eval"]
	labels_names_train=[]
	labels_names_dev=[]
	labels_names_eval=[]

	names_tokenizer=[]

	for i in range(len(ph)):
		phase_folder=os.path.join(parent_dir,ph[i])  #Etapas

		if ph[i] == "training":
			for lan in os.listdir(phase_folder):
				lan_folder=os.path.join(phase_folder,lan) #Lenguajes

				for cor in os.listdir(lan_folder):
					cor_path=os.path.join(lan_folder,cor) #Clean/Noisy

					for name in os.listdir(cor_path):
						names_path=os.path.join(cor_path,name) #Names
						names_tokenizer.append(names_path)

						lines=io.open(names_path,encoding='UTF-8').read().strip().split('\n')

						index=names_train.tolist().index(name.split('.')[0])
						pairs_train=[lines[0],labels_train[index]] 
						labels_names_train.append(str(name.split('.')[0])) #Cuando quiero ver los nombres y labels
						lines_files_train.append(pairs_train)

			print('Cantidad de Archivos de Train:{}'.format(len(lines_files_train)))

		elif ph[i] == "devel":

			for name in os.listdir(phase_folder):
				names_path=os.path.join(phase_folder,name) #Names
				names_tokenizer.append(names_path)

				lines=io.open(names_path,encoding='UTF-8').read().strip().split('\n')

				index=names_dev.tolist().index(name.split('.')[0])
				pairs_dev=[lines[0],labels_dev[index]] 
				labels_names_dev.append(str(name.split('.')[0])) #Cuando quiero ver los nombres y labels
				lines_files_dev.append(pairs_dev) #Guarda el contenido

			print('Cantidad de Archivos de Dev:{}'.format(len(lines_files_dev)))

		elif ph[i] == "eval":

			for name in os.listdir(phase_folder):
				names_path=os.path.join(phase_folder,name) #Names
				names_tokenizer.append(names_path)

				lines=io.open(names_path,encoding='UTF-8').read().strip().split('\n')

				index=names_eval.tolist().index(name.split('.')[0])
				pairs_ev=[lines[0],labels_eval[index]] 
				labels_names_eval.append(str(name.split('.')[0])) #Cuando quiero ver los nombres y labels
				lines_files_ev.append(pairs_ev)  

			print('Cantidad de Archivos de Eval:{}'.format(len(lines_files_ev)))


	return zip(*lines_files_train),zip(*lines_files_dev),zip(*lines_files_ev),labels_names_train, labels_names_dev, labels_names_eval,names_tokenizer





# Funcion para Tokenizar cada unidad ---------------------------------------------------------

def tokenize(lang,names_tokenizer):

	#bert_tokenizer = Tokenizer(WordPiece())
	#bert_tokenizer.pre_tokenizer = WhitespaceSplit()
	#bert_tokenizer.post_processor = TemplateProcessing(
	#	single="[CLS] $A [EOS]",
	#	special_tokens=[("[CLS]", 1),("[EOS]", 2),("[UNK]", 3)],)
	#trainer = WordPieceTrainer(vocab_size=20000, special_tokens=["[UNK]","[PAD]"])
	#bert_tokenizer.train(trainer, names_tokenizer)
	#bert_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

	#files = bert_tokenizer.model.save(r"C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\V3_Window_Attention\Version_1.5_WP_Tokenizer", "vocab")
	#bert_tokenizer.model = WordPiece.from_file(*files, unk_token="[UNK]")
	#bert_tokenizer.save(r"C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\V3_Window_Attention\Version_1.5_WP_Tokenizer/tokenizer-vocab.json")
	bert_tokenizer = Tokenizer.from_file(r"C:\Users\ASUS\Desktop\David\Proyectos\Reconocimiento de Idioma con Transformers\Sistema Fonotactico\V3_Window_Attention\Version_1.5_WP_Tokenizer/tokenizer-vocab.json")


	output = bert_tokenizer.encode_batch(lang)


	tensor=np.zeros((1,1024),dtype=int)
	for i in range(len(list(lang))):
		tensor=np.append(tensor,[output[i].ids[:1024]],axis=0)

	tensor=tf.convert_to_tensor(tensor[1:,:])
	print(tensor)
	print(tensor.shape)
	#print(output[5].tokens)
	#print(output[0].ids)
	#print(len(output[5].tokens))
	#print(len(output[5].ids))
	#print(bert_tokenizer.get_vocab())
	#print(len(bert_tokenizer.get_vocab()))

	return tensor,bert_tokenizer






# Funcion principal -----------------------------------------------------------------------------

def load_dataset(path):

	print()
	print()
	print("Loading ......")

	#Procesa las listas y carga los contenidos de cada archivo

	(sent_train,labels_train),(sent_dev,labels_dev),(sent_ev,labels_ev),labels_names_train, labels_names_dev, labels_names_eval,names_tokenizer=create_dataset(path)


	print()
	print()
	print("Tokenizing.....")

	# Se realiza la tokenizacion de cada unidad 

	input_tensor_train,lang_tokenizer=tokenize(sent_train,names_tokenizer)


	output = lang_tokenizer.encode_batch(sent_dev)
	input_tensor_dev=np.zeros((1,1024),dtype=int)
	for i in range(len(list(sent_dev))):
		input_tensor_dev=np.append(input_tensor_dev,[output[i].ids[:1024]],axis=0)
	input_tensor_dev=tf.convert_to_tensor(input_tensor_dev[1:,:])
	print(input_tensor_dev)
	print(input_tensor_dev.shape)

	output = lang_tokenizer.encode_batch(sent_ev)
	input_tensor_ev=np.zeros((1,1024),dtype=int)
	for i in range(len(list(sent_ev))):
		input_tensor_ev=np.append(input_tensor_ev,[output[i].ids[:1024]],axis=0)
	input_tensor_ev=tf.convert_to_tensor(input_tensor_ev[1:,:])
	print(input_tensor_ev)
	print(input_tensor_ev.shape)



	print('')
	print('IPA VOCABULARY')
	print('')
	print(lang_tokenizer.get_vocab())
	print(len(lang_tokenizer.get_vocab()))
	print('')


    #Se transforma los labels a formato int
	labels_t=[]
	labels_d=[]
	labels_e=[]

	for i in range(len(labels_train)):
		labels_t.append(int(labels_train[i]))

	for i in range(len(labels_dev)):
		labels_d.append(int(labels_dev[i]))

	for i in range(len(labels_ev)):
		labels_e.append(int(labels_ev[i]))

	return input_tensor_train,labels_t,input_tensor_dev,labels_d,input_tensor_ev,labels_e,lang_tokenizer,labels_names_train,labels_names_dev, labels_names_eval
