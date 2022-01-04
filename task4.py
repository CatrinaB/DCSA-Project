import math
import time
from mrjob.job import MRJob
from mrjob.step import MRStep


class FrobeniusNormIndex(MRJob):
	"""
	In this class, I show how we can calculate the Frobenius norm using a key for the MapReduce, in this case
	the index of the matrix line that is being read.
	"""
	# in order to use the index of the line we are reading as key, we need to save it in a variable
	index = -1
	
	def mapper_1(self, _, line):
		"""
		Mapper that reads the file line by line and returns (line_index, squared_absolute_value) for each number.
		:param _: None
		:param line: the current line of the file
		:return: (line_index, squared_absolute_value)
		"""
		# we increase the index for each line we read
		FrobeniusNormIndex.index += 1
		# we split the line along the spaces between the numbers to get the numbers
		for number in line.split(' '):
			# the numbers are read as strings, so we convert them to float; then, we square their absolute value
			# we return a (line_index, squared_absolute_value) pair for each number
			yield self.index, abs(float(number)) ** 2
			
	def reducer_1(self, index, number):
		"""
		Reducer that groups the new values by row and returns their sum.
		:param index: the index of the matrix line the number was read from, used as key; we don't actually use it,
		but we pass it as an input argument so that the reducer knows how to group the numbers
		:param number: the squared absolute value of one element of the matrix
		:return: (None, row_sum = the sum of the numbers on the same one row)
		"""
		# since we are just going to add together all the row sums, we don't need a key
		yield None, sum(number)
		
	def reducer_2(self, _, row_sum):
		"""
		Reducer that takes all the row sums and returns the Frobenius norm.
		:param _: None
		:param row_sum: the sum of all the elements of a row
		:return: (None, Frobenius_norm = sqrt(sum(row_sum)) )
		"""
		yield None, math.sqrt(sum(row_sum))
		
	def steps(self):
		return [
			MRStep(mapper=self.mapper_1,
				   reducer=self.reducer_1),
			MRStep(reducer=self.reducer_2)
		]

class FrobeniusNormNoIndex(MRJob):
	"""
	In this class, I show that we can calculate the Frobenius norm without using a key for the MapReduce.
	"""
	
	def mapper_1(self, _, line):
		"""
		Mapper that reads the file line by line and returns (None, squared_absolute_value) for each number.
		:param _: None
		:param line: the current line of the file
		:return: (None, squared_absolute_value)
		"""
		# we split the line along the spaces between the numbers to get the numbers
		for number in line.split(' '):
			# the numbers are read as strings, so we convert them to float; then, we square their absolute value
			# we return a (None, squared_absolute_value) pair for each number
			yield None, abs(float(number)) ** 2
	
	def reducer_1(self, _, number):
		"""
		Reducer that returns the sum of all the numbers it receives.
		:param _: None
		:param number: the squared absolute value of one element of the matrix
		:return: (None, row_sum = the sum of the numbers on the same one row)
		"""
		# since we are just going to add together all the row sums, we don't need a key
		yield None, sum(number)
	
	def reducer_2(self, _, red_sum):
		"""
		Reducer that takes all the previously calculated sums and returns the Frobenius norm.
		:param _: None
		:param red_sum: the sum of all the elements of a reducer
		:return: (None, Frobenius_norm = sqrt(sum(red_sum)) )
		"""
		yield None, math.sqrt(sum(red_sum))
	
	def steps(self):
		return [
			MRStep(mapper=self.mapper_1,
				   reducer=self.reducer_1),
			MRStep(reducer=self.reducer_2)
		]


if __name__ == '__main__':
	start = time.time()
	FrobeniusNormIndex.run()
	FrobeniusNormNoIndex.run()
	end = time.time()
	print('execution time: ', end - start)
