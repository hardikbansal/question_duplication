import optparse
import os

class trainOptions():
	def __init__(self):
		self.parser = optparse.OptionParser()
		self.initialized = False

	def initialize(self):
		self.parser.add_option('--batch_size', type='int', default=100, dest='batch_size')
		self.parser.add_option('--max_epoch', type='int', default=50, dest='max_epoch')
		self.parser.add_option('--num_token', type='int', default=20000, dest='num_token')
		self.parser.add_option('--emb_size', type='int', default=300, dest='emb_size')
		self.parser.add_option('--test', action="store_true", default=False, dest="test")
		

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()

		self.opt = self.parser.parse_args()

		return self.opt
