import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from options import trainOptions


class DD():

	def initialize(self):
		
		opt = trainOptions().parse()[0]

		self.batch_size = 32
		self.max_epoch = opt.max_epoch
		self.to_test = opt.test
		self.load_checkpoint = False
		self.emb_size = opt.emb_size
		self.num_token = opt.num_token
		
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
		
		self.tk.fit_on_texts(self.questions1 + self.questions2)

		self.num_questions = self.questions1.size
		self.word_index = self.tk.word_index
		

	def glove_emb(self):

		self.glove_embedding = {}

		self.n_words = len(self.word_index)

		print("Loading glove matrix")
		
		f = open("data/glove.42B.300d.txt", "r")
		for lines in f:
			line = lines.split(' ')
			self.glove_embedding[line[0]] = np.asarray(line[1:], dtype=np.float32)

		print("Done loading glove matrix")

		# sys.exit()

		self.embedding_matrix = np.zeros((self.num_token, 300), dtype=np.float32)

		for words in self.word_index.items():
			temp = self.glove_embedding.get(words[0])
			if(temp is not None and words[1] < self.num_token):
				self.embedding_matrix[words[1]] = temp

		print("Done making embedding matrix")



	def network(self, ques1):

		o_l1 = linear1d(ques1, self.emb_size, 128, name="l1")
		o_l2 = linear1d(o_l1, 128, 128, name="l2")
		o_l3 = linear1d(o_l2, 128, 128, name="l3")

		o_conc = tf.concat([o_l1, o_l2, o_l3], axis=0)
		out = tf.contrib.layers.batch_norm(o_enc, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

		return out

	def loss_setup(self, pred, act):

		return 0

	def model_setup(self):

		with tf.variable_scope("Model") as scope:

			self.qs1_ph = tf.placeholder(tf.float32, [self.batch_size, self.emb_size])
			self.qs2_ph = tf.placeholder(tf.float32, [self.batch_size, self.emb_size])
			self.y_ph = tf.placeholder(tf.float32, [self.batch_size, 1])

			o_net_1 = self.network(self.qs1_ph)
			o_net_2 = self.network(self.qs2_ph)

			pred_y = tf.sqrt(tf.mean(tf.squared_difference(o_net_1, o_net_2)))

			self.loss = self.loss_setup(pred_y, self.y_ph)

			optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
			self.loss_optimizer = optimizer.minimize(self.loss, var_list=enc_dec_vars)


	def train(self):


		if not os.path.exists(self.check_dir+"/glove1.npy"):

			# Loading dataset and creating embeddings
			self.load_dataset()

			print("Done loading dataset")

			# Loading the glove embedding from the file
			self.glove_emb()

			# print(self.embedding_matrix[1])

			i = 0

			self.question1_glove = np.zeros((self.num_questions, 300), dtype=np.float32)
			self.question2_glove = np.zeros((self.num_questions, 300), dtype=np.float32)

			# Making the glove vector by taking the sum of all glove vector of the words

			while(i+1000 < self.num_questions):

				temp = self.tk.texts_to_matrix(self.questions1[i:i+1000], mode="tqidf")
				temp = np.matmul(temp, self.embedding_matrix)
				self.question1_glove[i:i+1000] = temp

				temp = self.tk.texts_to_matrix(self.questions2[i:i+1000], mode="tqidf")
				temp = np.matmul(temp, self.embedding_matrix)
				self.question2_glove[i:i+1000] = temp

				i+=1000
				print i

			temp = self.tk.texts_to_matrix(self.questions1[i:], mode="tqidf")
			temp = np.matmul(temp, self.embedding_matrix)
			self.question1_glove[i:] = temp

			temp = self.tk.texts_to_matrix(self.questions2[i:], mode="tqidf")
			temp = np.matmul(temp, self.embedding_matrix)
			self.question2_glove[i:] = temp

			if not os.path.exists(self.check_dir):
				os.makedirs(self.check_dir)


			#Saving the vector representations for questions

			np.save(self.check_dir+"/glove1.npy", self.question1_glove)
			np.save(self.check_dir+"/glove2.npy", self.question2_glove)
			self.y_train = self.df['is_duplicate'].values
			np.save(self.check_dir+"/y_train.npy", self.y_train)

		else :

			self.question1_glove = np.load(self.check_dir+"/glove1.npy")
			self.question2_glove = np.load(self.check_dir+"/glove2.npy")
			self.y_train = np.load(self.check_dir+"/y_train.npy")

		self.model_setup()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		
		with tf.Session() as sess:

			sess.run(init)
			writer = tf.summary.FileWriter(self.tensorboard_dir)

			for epoch in range(0, self.max_epoch):
				for itr in range(self.num_questions/self.batch_size):

					feed_dict = { self.qs1_ph:self.question1_glove[itr*self.batch_size: (itr+1)*self.batch_size],
								  self.qs2_ph:self.question2_glove[itr*self.batch_size: (itr+1)*self.batch_size],
								  self.y_ph:self.y_train[itr*self.batch_size: (itr+1)*self.batch_size]}

					_ = sess.run(self.loss_optimizer, feed_dict=feed_dict)

					sys.exit()

def main():

	model = DD()
	model.initialize()
	model.train()

if __name__ == "__main__":
	main()