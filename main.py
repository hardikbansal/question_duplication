import numpy
import os
import sys
import pandas as pd
import tensorflow as tf




class DD():

	def load_dataset(self):

		self.df = pd.read_csv("data/train.csv", delimiter=",")

		self.df['question1'] = self.df['question1'].apply(lambda x: str(x).strip())
		self.df['question2'] = self.df['question2'].apply(lambda x: str(x).strip())


	def tokenizer(self):

		tk = tf.keras.preprocessing.text.Tokenizer(20000, lower="true", split=' ')
		
		questions1 = self.df['question1'].values
		questions2 = self.df['question2'].values
		
		tk.fit_on_texts(questions1 + questions2)

		self.question1_sequences = tk.texts_to_matrix(questions1)
		self.question2_sequences = tk.texts_to_matrix(questions2)

		print(self.question1_sequences[0])
		sys.exit()

		self.word_index = tk.word_index

	def glove_emb(self):

		self.glove_embedding = {}
		
		f = open("data/glove.txt", "r")
		for lines in f:
			lines = lines.split(' ')
			self.glove_embedding[lines[0]] = lines[1]

	def emb_matrix(self):

		n_words = len(self.word_index)
		embedding_matrix = np.zeros(n_words + 1, 300)

		for words in self.word_index.items():

			if(words[0] in self.glove_embedding.keys()):
				embedding_matrix[words[1]] = self.glove_embedding[words[0]]

		return embedding_matrix




def main():

	model = DD()
	model.load_dataset()
	model.tokenizer()

if __name__ == "__main__":
	main()