import time

import nltk
from mrjob.job import MRJob
from mrjob.step import MRStep
from collections import Counter


class MovieGenreKeywords(MRJob):
	
	def mapper_1(self, _, line):
		line = line.strip().lower()
		if line.find('"') > -1:
			
			aux = line.split(',"')
			aux = aux[1]
			[title, genres] = aux.split('",')
		# genres = genres.split('|')
		# words[2] = words[2][1:]
		else:
			words = line.split(',')
			title = words[1]
			genres = words[2]
		
		genres = genres.split('|')
		
		for genre in genres:
			if (genre != 'genres'):
				yield genre, self.remove_title_words(title)
	
	def combiner_1(self, genre, title):
		yield genre, list(title)
	
	def reducer_1(self, genre, title):
		yield genre, list(title)
	
	def mapper_2(self, genre, titles):
		# words = string.split(' ')
		for title in titles:
			words = title.split(' ')
			for word in words:
				yield genre, word
	
	# def combiner_2(self, genre, lista):
	# 	word_list = {}
	# 	for i in lista:
	# 		if i[0] != '':
	# 			if i[0] not in word_list:
	# 				word_list[i[0]] = i[1]
	# 			else:
	# 				word_list[i[0]] += i[1]
	#
	# 	yield genre, word_list
	
	def reducer_2(self, genre, words):
		yield genre, Counter(words).most_common(10)
	
	def steps(self):
		return [
			MRStep(mapper=self.mapper_1,
				   # combiner=self.combiner_1,
				   reducer=self.reducer_1),
			MRStep(mapper=self.mapper_2,
				   # combiner=self.combiner_2,
				   reducer=self.reducer_2)
		]
	
	def remove_title_words(self, title):
		title = title[:title.find('(')]
		text = nltk.word_tokenize(title)
		
		tagged = nltk.pos_tag(text, 'universal')
		
		remove_pos = ['ADP', 'ADV', 'CONJ', 'DET', 'PRT', '.', 'NUM', 'X']
		remove_words = ['a.k.a', '*', 'ii', 'iii', '']
		final = ''
		
		for i in tagged:
			if i[1] not in remove_pos and i[0] not in remove_words:
				final += (i[0] + ' ')
		
		return final[0:-1]


if __name__ == '__main__':
	start = time.time()
	MovieGenreKeywords.run()
	end = time.time()
	print('execution time: ', end - start)