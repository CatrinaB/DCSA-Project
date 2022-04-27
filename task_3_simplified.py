import math
import time
from collections import Counter
from mrjob.job import MRJob
from mrjob.step import MRStep
import pandas as pd

class KNNMapReduce(MRJob):
	# class variable used to store the feature values of the unlabeled data
	unlabeled_data = []
	
	def mapper_1(self, _, line):
		"""
		Mapper that takes each line of the file and returns a (species, [id, features]) pair
		:param _: None
		:param line: the current line of the file
		:return : (species, [id, feature])
		"""
		# since we read the data from a CSV, we split the line along the commas
		aux = line.split(',')
		# the first line contains the names of the columns, so we add a condition to ignore it
		if aux[0].isnumeric():
			yield aux[-1], aux[:-1]
	
	def reducer_1(self, species, features):
		"""
		Reducer that brings the [species, [id, features]) pairs together and that saves in the global
		variable the values of the '' key, meaning of the unlabeled data.
		:param species: if the data was labeled, the name of the species; if it was unlabeled, ''
		:param features: the [id, features] values of each key
		:return: (species, [id, features]) -- only for labeled data
		"""
		# if the data is unlabeled, we save it in the global variable
		if species == '':
			KNNMapReduce.unlabeled_data = list(features)
		# if it is labeled, we return a (species, [if, features]) pair
		# the features input argument is a generator and cannot be yielded as such, so we transform it into a list
		else:
			yield species, list(features)
	
	def mapper_2(self, species, features):
		"""
		Mapper that takes the previous output, calculates the distance between the unlabeled data's
		features' values and the labeled data's features' values; returns a (unlabeled_set_id, [distance, label]) pair
		:param species: the label of the data, also the key of the previous output
		:param features: the [id, features] value of each key (label)
		:return: (unlabeled_set_id, [distance, label]) pair
		"""
		# for each set of feature values in unlabeled_data, we calculate the distance between it and each
		# labeled feature set; then, we return a (unlabeled_set_id, [distance, label]) pair
		for el in KNNMapReduce.unlabeled_data:
			for feature in features:
				yield el[0], (self.euclidean_distance(el[1:], feature[1:]), species)
	
	def combiner_2(self, unlabeled_set_id, distances):
		"""
		Combiner that partially combines the (unlabeled_set_id, [distance, label]) pairs by key
		:param unlabeled_set_id: list of unlabeled sets' IDs, used as key
		:param distances: generator of [distance, label] elements
		:return: (unlabeled_set_id, [[distance, label]])
		"""
		# just like before, we need to convert the generator to a list
		yield unlabeled_set_id, list(distances)
	
	def reducer_2(self, unlabeled_set_id, distances):
		"""
		Reducer that finishes combining the (unlabeled_set_id, [distance, label]) pairs by key and returns a list of
		the closest 15 elements to each unlabeled_set_id key, together with their labels
		:param unlabeled_set_id: list of unlabeled sets' IDs, used as key
		:param distances: generator of [distance, label] elements
		:return: (unlabeled_set_id, [[distance, label]]) pairs
		"""
		# the generator 'distances' is of form [list 1 from combiner, list 2 from combiner, ...], where "list n from
		# combiner" is of shape (number_elements, 2); in other words, it is 3D-shaped and we need to make it 2D
		list_d = [item for sublist in list(distances) for item in sublist]
		
		# the Python implicit sort function works as follows on matrices: it sorts the elements in increasing order
		# based on the value of the first element, using the second element just to differentiate between elements
		# that have the same first value; this works for us, since our elements are of [distance, label] form
		list_d.sort()
		
		# we return only the first 15 elements of the sorted list, which correspond to the 15 nearest neighbours of
		# each set of unlabeled values
		yield unlabeled_set_id, list_d[:15]
	
	def mapper_3(self, unlabeled_set_id, distances):
		"""
		Mapper that returns a (unlabeled_set_id, label) pair for ease of counting in the reducer.
		:param unlabeled_set_id: the id of the unlabeled feature set, used as key
		:param distances: generator of [distance, label] pairs
		:return: (unlabeled_set_id, label) pair
		"""
		# for each element of form [distance, label] in the generator, we return a (unlabeled_set_id, label) pair
		for i in distances:
			yield unlabeled_set_id, i[1]
	
	def reducer_3(self, unlabeled_set_id, species):
		"""
		Reducer that returns the most common label for each unlabeled_set_id.
		:param unlabeled_set_id: the id of the unlabeled feature set, used as key
		:param species: generator with all the 15 species nearest to the unlabeled_set_id
		:return: (unlabeled_set_id, predicted_label)
		"""
		# because we know the generator has only 15 elements, it makes no sense to use MapReduce to count the number
		# of occurrences of each element; instead, we use the implicit collections.Counter function and its attribute
		# most_common
		# Because Counter(species).most_common(1)[0][0] outputs [[most common label, number of occurrences]], we return
		# the element with the index [0][0], since we are only interested in the label.
		yield unlabeled_set_id, Counter(species).most_common(1)[0][0]
	
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
	
	def euclidean_distance(self, list1, list2):
		"""
		Function that calculates and returns the euclidian distance between two elements represented by an array
		containing their coordinates.
		:param list1: list of coordinates of first element
		:param list2: list of coordinates of second element
		:return: euclidean distance, float
		"""
		sum_el = 0
		
		# we assume the two arrays have the same length, since the elements they represent are situated
		# in the same n-dimensional space
		for i in range(len(list1)):
			# for each pair of i-th elements, we subtract them, we square the result, and we add the square to the sum;
			# because in this program, the arrays we are using store their elements as strings. we need to convert
			# them to float first
			sum_el += (float(list1[i]) - float(list2[i])) ** 2
			
		# finally, we return the square root of the sum
		return math.sqrt(sum_el)


if __name__ == '__main__':
	start = time.time()
	
	# we read the CSV file into a pandas DataFrame, using the already existing IDs as DataFrame IDs
	data = pd.read_csv('Iris.csv', index_col=0)
	
	# we normalize the feature values using the formula
	# x_norm = x / |x_max|
	for col in data.columns[:-1]:
		data[col] = data[col] / data[col].abs().max()
	
	# we write the newly normalized data back into the CSV file, since we will use it as input for the MapReduce
	data.to_csv('Iris_normalized.csv')
	
	KNNMapReduce.run()
	end = time.time()
	print('execution time: ', end - start)
