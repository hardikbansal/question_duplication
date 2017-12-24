import numpy
import os
import sys
import pandas as pd
import tensorflow as tf




class DD():

	def __initialize__(self):
		
		opt = trainOptions().parse()[0]

		self.batch_size = 32
		self.max_epoch = opt.max_epoch
		self.to_test = opt.test
		self.load_checkpoint = False
		self.emb_size = opt.emb_size
		
		self.tensorboard_dir = "./output/" + self.model + "/" + self.dataset + "/tensorboard"
		self.check_dir = "./output/"+ self.model + "/" + self.dataset +"/checkpoints"
		self.images_dir = "./output/" + self.model + "/" + self.dataset + "/imgs"


	def load_dataset(self):

		self.df = pd.read_csv("data/train.csv", delimiter=",")

		self.df['question1'] = self.df['question1'].apply(lambda x: str(x).strip())
		self.df['question2'] = self.df['question2'].apply(lambda x: str(x).strip())


		# Creating tokens for the given corpus

		tk = tf.keras.preprocessing.text.Tokenizer(2000, lower="true", split=' ')
		
		questions1 = self.df['question1'].values
		questions2 = self.df['question2'].values
		
		tk.fit_on_texts(questions1 + questions2)

		self.question1_matrix = tk.texts_to_matrix(questions1, mode="tfidf")
		self.question2_matrix = tk.texts_to_matrix(questions2, mode="tfidf")

		self.word_index = tk.word_index

	def glove_emb(self):

		self.glove_embedding = {}
		
		f = open("data/glove.txt", "r")
		for lines in f:
			lines = lines.split(' ')
			self.glove_embedding[lines[0]] = lines[1]

		n_words = len(self.word_index)

		self.embedding_matrix = np.zeros(n_words + 1, 300)

		for words in self.word_index.items():
			if(words[0] in self.glove_embedding.keys()):
				self.embedding_matrix[words[1]] = self.glove_embedding[words[0]]



	def network(self, ques1):

		o_l1 = linear1d(ques1, self.emb_size, 128, name="l1")
		o_l2 = linear1d(o_l1, 128, 128, name="l2")
		o_l3 = linear1d(o_l2, 128, 128, name="l3")

		o_conc = tf.concat([o_l1, o_l2, o_l3], axis=0)

		return o_conc

	def loss_setup(self, pred, act):

		return 0

	def model_setup(self):

		with tf.variable_scope("Model") as scope:

			qs1_ph = tf.placeholder(tf.float32, [self.batch_size, self.emb_size])
			qs2_ph = tf.placeholder(tf.float32, [self.batch_size, self.emb_size])
			y_ph = tf.placeholder(tf.float32, [self.batch_size, 1])

			o_net_1 = self.network(qs1_ph)
			o_net_2 = self.network(qs2_ph)

			pred_y = tf.sqrt(tf.mean(tf.squared_difference(o_net_1, o_net_2)))

			self.loss = loss_setup(pred_y, 0)

			optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
			self.loss_optimizer = optimizer.minimize(self.loss, var_list=enc_dec_vars)




	def train(self):

		# Loading dataset and creating embeddings
		self.load_dataset()

		sys.exit()
		# Loading the glove embedding from the file
		self.glove_emb()


		self.question1_glove = tf.matmul(self.question1_matrix, self.embedding_matrix)
		self.question2_glove = tf.matmul(self.question2_matrix, self.embedding_matrix)

		self.model_setup()
		self.loss_setup()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		if not os.path.exists(self.images_dir+"/train/"):
			os.makedirs(self.images_dir+"/train/")
		if not os.path.exists(self.check_dir):
			os.makedirs(self.check_dir)

		with tf.Session() as sess:

			sess.run(init)
			writer = tf.summary.FileWriter(self.tensorboard_dir)

			for epoch in range(0, self.max_epoch):
				for itr in range(self.num_questions/self.batch_size):

					feed_dict = { qs1_ph:self.question1_glove[itr*self.batch_size: (itr+1)*self.batch_size],
								  qs2_ph:self.question2_glove[itr*self.batch_size: (itr+1)*self.batch_size]}
					_ = sess.run(self.loss_optimizer, feed_dict=feed_dict)

def main():

	model = DD()
	model.train()

if __name__ == "__main__":
	main()