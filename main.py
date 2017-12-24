import numpy
import os
import sys
import pandas as pd




class DD():

	def load_dataset(self):
		df = panda.read_csv("data/train.csv", delimiter=",")

		df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
		df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))





def main():

	model = DD()
	model.load_dataset()
	# model.initialize()

	
	# if(model.to_test):
	# 	model.test()
	# else:
	# 	model.train()


if __name__ == "__main__":
	main()