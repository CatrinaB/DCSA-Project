import re
import time

from mrjob.job import MRJob

WORD_RE = re.compile(r"[\w']+") # match words

class InvertWebLink(MRJob):
	
	'''
	Mapper takes each line of the file and, if it is of [from_id, to_id] form, returns a (to_id, from_id) pair
	:param _: None
	:param line: the current line of the output file
	:return: (to_id, from_id)
	'''
	def mapper(self, _, line):
		aux = []
		
		# in the input file, the connections are written as 'from_id \tab to_id', so using regex is the easiest way
		# to retrieve the IDs
		for word in WORD_RE.findall(line):
			# because we need both IDs, but we read them in turns, we save them in a temporary array
			aux.append(word)
		
		# the input file includes comments; for comment lines, aux[0] is a word, therefore not numeric,
		# so we ignore those lines
		if aux[0].isnumeric():
			# if aux contains IDs, it is of form [from_id, to_id]
			yield aux[1], aux[0]
	
	'''
	Reducer that returns a list of all the page IDs that link to a certain page ID
	:param to_id: ID of the page that is linked to
	:param from_id: generator object with IDs of pages linking to page to_id
	:return: (to_id, [from_id])
	'''
	def reducer(self, to_id, from_id):
		# the generator from_id already contains all the values that have the key to_id;
		# we just need to return it as a serializable object (because of JSON default protocol in MRJob),
		# in this case a list
		yield to_id, list(from_id)


if __name__ == '__main__':
	start = time.time()
	InvertWebLink.run()
	end = time.time()
	print('execution time: ', end - start)
