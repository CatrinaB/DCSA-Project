import math
import re
from collections import Counter
import time
import numpy as np
from mrjob.job import MRJob
from mrjob.step import MRStep

unlabeled_data = re.compile(r"\w")

class KNNMapReduce(MRJob):
	
	def mapper_1(self, _, line):
		aux = line.split(',')
		if aux[0].isnumeric():
			yield aux[-1], aux[:-1]
	
	def reducer_1(self, species, features):
		if species == '':
			global unlabeled_data
			unlabeled_data = list(features)
		else:
			yield species, list(features)
	
	def mapper_2(self, species, features):
		for el in unlabeled_data:
			for feature in features:
				yield el[0], (self.euclidean_distance(el[1:], feature[1:]), species)
		# yield species[''], list(features)
	
	def combiner_2(self, uf, distances):
		# list_d = list(distances)
		yield uf, list(distances)
		
	def reducer_2(self, uf, distances):
		list_d = [item for sublist in list(distances) for item in sublist]
		list_d.sort()
		yield uf, list_d[:15]
		
	def mapper_3(self, uf, distances):
		for i in distances:
			yield uf, i[1]
		
	def reducer_3(self, uf, species):
		# c = Counter(species)
		yield uf, Counter(species).most_common(1)[0][0]
		
	# def reducer_3(self, uf, distances):
	
	def steps(self):
		return [
			MRStep(mapper=self.mapper_1,
				   reducer=self.reducer_1),
			MRStep(mapper=self.mapper_2,
				   combiner=self.combiner_2,
				   reducer=self.reducer_2),
			MRStep(mapper=self.mapper_3,
				   reducer=self.reducer_3)
		]
	
	def euclidean_distance (self, arr1, arr2):
		sum = 0
		
		for i in range(len(arr1)):
			sum += (float(arr1[i]) - float(arr2[i])) ** 2
		return math.sqrt(sum)


if __name__ == '__main__':
	KNNMapReduce.run()
	# print('unl', unlabeled_data)
