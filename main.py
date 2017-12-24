import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from options import trainOptions
from layers import *


class DD():

	def initialize(self):
		
		opt = trainOptions().parse()[0]

		self.batch_size = 128
		self.max_epoch = opt.max_epoch
		self.to_test = opt.test
		self.load_checkpoint = False
		self.emb_size = opt.emb_size
		self.num_token = opt.num_token
		self.num_train_questions = 300000
		
		self.tensorboard_dir = "./output/tensorboard"
		self.check_dir = "./output/checkpoints"


	def load_dataset(self):

		self.df = pd.read_csv("data/train.csv", delimiter=",")

		self.df['question1'] = self.df['question1'].apply(lambda x: str(x).strip())
		self.df['question2'] = self.df['question2'].apply(lambda x: str(x).strip())

		# Creating tokens for the given corpus

		self.tk = tf.keras.preprocessing.text.Tokenizer(self.num_token, lower="true", split=' ')
		
		self.questions1 = self.df['question1'].values
		self.questions2 = self.df['question2'].values
		self.y_train = self.df['is_duplicate'].values
		
		self.tk.fit_on_texts(self.questions1 + self.questions2)

		self.num_questions = self.questions1.size
		self.word_index = self.tk.word_index

		
		# Loading test dataset 

		self.df_test = pd.read_csv("data/test.csv", delimiter=",")
		self.num_test_questions = self.df_test.shape[0]
		
		self.df_test['question1'] = self.df_test['question1'].apply(lambda x: str(x).strip())
		self.df_test['question2'] = self.df_test['question2'].apply(lambda x: str(x).strip())


		self.questions1_test = self.df_test['question1'].values
		self.questions2_test = self.df_test['question2'].values

	def pre_process(self):

		# checking if the pre processing has been done alreay or not
		# If not run the code to preprocess otherwise just load the data

		if not os.path.exists(self.check_dir+"/glove1.npy"):

			# Loading dataset and creating embeddings
			self.load_dataset()

			print("Done loading dataset")

			# Loading the glove embedding from the file
			self.glove_emb()

			# print(self.embedding_matrix[1])

			self.question1_glove = np.zeros((self.num_questions, 300), dtype=np.float32)
			self.question2_glove = np.zeros((self.num_questions, 300), dtype=np.float32)

			# Making the glove vector by taking the sum of all glove vector of the words

			i = 0
			
			while(i+1000 < self.num_questions):

				temp = self.tk.texts_to_matrix(self.questions1[i:i+1000], mode="tfidf")
				temp = np.matmul(temp, self.embedding_matrix)
				self.question1_glove[i:i+1000] = temp

				temp = self.tk.texts_to_matrix(self.questions2[i:i+1000], mode="tfidf")
				temp = np.matmul(temp, self.embedding_matrix)
				self.question2_glove[i:i+1000] = temp

				i+=1000
				print i

			temp = self.tk.texts_to_matrix(self.questions1[i:], mode="tfidf")
			temp = np.matmul(temp, self.embedding_matrix)
			self.question1_glove[i:] = temp

			temp = self.tk.texts_to_matrix(self.questions2[i:], mode="tfidf")
			temp = np.matmul(temp, self.embedding_matrix)
			self.question2_glove[i:] = temp

			if not os.path.exists(self.check_dir):
				os.makedirs(self.check_dir)

			# Saving processed data

			np.save(self.check_dir+"/glove1.npy", self.question1_glove)
			np.save(self.check_dir+"/glove2.npy", self.question2_glove)
			np.save(self.check_dir+"/y_train.npy", self.y_train)
			np.save(self.check_dir+"/emb_matrix.npy", self.embedding_matrix)

		else :

			# Loading Processed data

			self.load_dataset()
			self.question1_glove = np.load(self.check_dir+"/glove1.npy")
			self.question2_glove = np.load(self.check_dir+"/glove2.npy")
			self.embedding_matrix = np.load(self.check_dir+"/emb_matrix.npy")



	def glove_emb(self):

		self.glove_embedding = {}

		self.n_words = len(self.word_index)

		print("Loading glove matrix")
		
		f = open("data/glove.42B.300d.txt", "r")
		for lines in f:
			line = lines.split(' ')
			self.glove_embedding[line[0]] = np.asarray(line[1:], dtype=np.float32)

		print("Done loading glove matrix")


		# Creating the embedding matrix of required words

		self.embedding_matrix = np.zeros((self.num_token, 300), dtype=np.float32)

		for words in self.word_index.items():
			temp = self.glove_embedding.get(words[0])
			if(temp is not None and words[1] < self.num_token):
				self.embedding_matrix[words[1]] = temp

		print("Done making embedding matrix")



	def network(self, ques):

		o_l1 = linear1d(ques, self.emb_size, 128, name="l1")
		o_l2 = linear1d(o_l1, 128, 128, name="l2")
		o_l3 = linear1d(o_l2, 128, 128, name="l3")
		o_conc = tf.concat([o_l1, o_l2, o_l3], axis=1)
		out = tf.contrib.layers.batch_norm(o_conc, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

		return out


	def contrast_loss(self, y_pred, y_act):

		return tf.reduce_mean((y_act)*(y_pred) + (1-y_act)*tf.square(tf.maximum(1 - tf.sqrt(y_pred), 0)))

	def model_setup(self):

		with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:

			self.qs1_ph = tf.placeholder(tf.float32, [None, self.emb_size])
			self.qs2_ph = tf.placeholder(tf.float32, [None, self.emb_size])
			self.y_ph = tf.placeholder(tf.float32, [None, 1])

			o_net_1 = self.network(self.qs1_ph)
			o_net_2 = self.network(self.qs2_ph)

			self.pred_y = tf.reduce_sum(tf.squared_difference(o_net_1, o_net_2),axis=1, keep_dims=True)

			self.loss = self.contrast_loss(self.pred_y, self.y_ph)

			optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
			self.loss_optimizer = optimizer.minimize(self.loss)
			self.loss_summ = tf.summary.scalar("loss", self.loss)

	def accuracy(self, y_pred, y_act):
		temp = np.sum(np.absolute( (np.sign(np.sqrt(y_pred)-0.5)+1.0)/2.0  - y_act))
		return temp

	def train(self):

		self.pre_process()
		self.model_setup()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		
		with tf.Session() as sess:

			sess.run(init)
			writer = tf.summary.FileWriter(self.tensorboard_dir)

			for epoch in range(0, self.max_epoch):
				
				for itr in range(self.num_train_questions/self.batch_size):

					feed_dict = { self.qs1_ph:self.question1_glove[itr*self.batch_size: (itr+1)*self.batch_size],
								  self.qs2_ph:self.question2_glove[itr*self.batch_size: (itr+1)*self.batch_size],
								  self.y_ph:np.reshape(self.y_train[itr*self.batch_size: (itr+1)*self.batch_size],(self.batch_size,1))}

					_, loss_str, temp_loss = sess.run([self.loss_optimizer, self.loss_summ, self.loss], feed_dict=feed_dict)

					if(itr%100 == 0):

						# Checking the performance of validation set

						temp_loss = sess.run(self.loss, feed_dict=feed_dict)

						print(epoch, itr, temp_loss)
						# print(epoch, itr, acc)
						# feed_dict = { self.qs1_ph:self.question1_glove[self.num_train_questions:],
						# 		  self.qs2_ph:self.question2_glove[self.num_train_questions:]
						# 		  }

						# temp_y_pred = sess.run(self.pred_y, feed_dict=feed_dict)
						# print(temp_y_pred[0])
						# acc = self.accuracy(temp_y_pred, np.reshape(self.y_train[self.num_train_questions:],(-1,1)))

						# print(epoch, itr, acc)

						# sys.exit()
					
					writer.add_summary(loss_str, epoch*int(self.num_questions/self.batch_size) + itr)

				saver.save(sess,os.path.join(self.check_dir,"dd"),global_step=epoch)

	def test(self):

		self.pre_process()
		self.model_setup()
		saver = tf.train.Saver()

		with tf.Session() as sess:

			chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
			saver.restore(sess,chkpt_fname)

			y_pred_store = np.array((self.y_test.shape[0], 1),dtype=np.float32)

			for itr in range(self.num_test_questions/self.batch_size):

				temp = self.tk.texts_to_matrix(self.questions1[itr*batch_size:(itr+1)*batch_size], mode="tfidf")
				questions1_feed = np.matmul(temp, self.embedding_matrix)
				temp = self.tk.texts_to_matrix(self.questions2[itr*batch_size:(itr+1)*batch_size], mode="tfidf")
				questions2_feed = np.matmul(temp, self.embedding_matrix)

				feed_dict={self.qs1_ph:question1_feed, self.qs2_ph:question2_feed}

				temp_y_pred = sess.run(self.pred_y, feed_dict=feed_dict)
				y_pred_store[itr*batch_size:(itr+1)*batch_size] = temp_y_pred

			self.accuracy(y_pred_store, np.reshape(self.y_test,[-1,1]))
				

def main():

	model = DD()
	model.initialize()
	model.train()

if __name__ == "__main__":
	main()