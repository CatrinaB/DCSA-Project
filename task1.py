import time
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from mrjob.job import MRJob
from mrjob.step import MRStep


class MovieGenreKeywords(MRJob):
	
	def mapper_1(self, _, line):
		"""
        Mapper that reads the input file line by line and returns a (genre, partial_title) pair for each genre of the
        movie. The partial_title is the original title without numerals, punctuation marks, adverbs, conjunctions etc.
        ans is obtained using the function remove_title_words.
        :param _: None
        :param line: the line of the file that is being read
        :return: (genre, partial_title)
        """
		# first, I eliminate all possible trailing empty spaces and convert the text to lowercase
		line = line.strip().lower()
		
		# the titles are written in the file in two ways: [id, title, genres] or [id, "title", genres]; the "title"
		# may include commas and quotations marks inside it, so it needs a different delimiter
		if line.find('"') > -1:
			# I use ',"' to separate the index from the rest of the line
			aux = line.split(',"')
			aux = aux[1]
			# I use '",' to separate the title and the genre
			[title, genres] = aux.split('",')
		else:
			# in case of [id, title, genres] U use the comma as a delimiter
			words = line.split(',')
			title = words[1]
			genres = words[2]
		
		# I isolate each genre
		genres = genres.split('|')
		
		for genre in genres:
			# the program also reads the first line of the input file, which contains the names of the columns; I added
			# this condition in order to ignore the first line
			if (genre != 'genres'):
				yield genre, self.remove_title_words(title)
	
	def reducer_1(self, genre, title):
		"""
        Reducer that groups the partial titles by genre.
        :param genre: the genre of a movie
        :param title: generator containing the partial titles of the movies pertaining to one genre
        :return: (genre, [titles])
        """
		# the generator object is not iterable and thus cannot be returned by the reducer,
		# so we transform it into a list
		yield genre, list(title)
	
	def mapper_2(self, genre, titles):
		"""
        Mapper that splits the titles into individual words and returns (genre, [word, 1]) for each word. This will
        allow us to count the number of occurrences of each word by genre.
        :param genre: the genre of a movie
        :param titles: the list of titles that belong to that genre
        :return: (genre, [word, 1])
        """
		for title in titles:
			words = title.split(' ')
			for word in words:
				yield genre, (word, 1)
	
	def combiner_2(self, genre, words):
		"""
        Combiner that groups the [word, 1] lists by genre, creates a dictionary of form
        {word: total_number_of_occurrences} and returns (genre, dictionary)
        :param genre: the genre of a movie
        :param words: list of [word, 1] lists
        :return: (genre, dictionary_total_number_of_occurrences_per_word)
        """
		word_dict = {}
		for word in words:
			# in practice, there are a few genres that had '' as a keyword. While this is most probably an issue related
			# to mapper_1 and the way the titles are split, it is easier to eliminate the '' values in this step
			if word[0] != '':
				# if the word is not already in the dictionary, it is added and initialised with the count 1
				if word[0] not in word_dict:
					word_dict[word[0]] = word[1]
				# if the word is already in the dictionary, its count is increased
				else:
					word_dict[word[0]] += word[1]
		
		yield genre, word_dict
	
	def reducer_2(self, genre, words):
		"""
        Reducer that returns the top 10 most used words for each genre and their counts.
        :param genre: the genre of a movie
        :param words: dictionary of {word: number_of_occurrences} form
        :return: (genre, [word, number of occurrences])
        """
		# the dictionary is read
		dct = next(words)
		# the dictionary is sorted by number of occurrences in decreasing order and the first 10 elements are returned
		# as a (genre, [word, number_of_occurrences]) pair
		yield genre, sorted(dct.items(), key=lambda el: el[1], reverse=True)[:10]
	
	def mapper_3(self, genre, words):
		"""
        Mapper that eliminates the word count from the previous reducer's output.
        :param genre: the genre of a movie
        :param words: a list of [word, number_of_occurrences] lists
        :return: (genre, [top_ten_words])
        """
		for word in words:
			yield genre, word[0]
	
	def reducer_3(self, genre, word):
		"""
        Reducer that groups the keywords by genre; the decreasing order is kept.
        :param genre: the genre of a movie
        :param word: generator containing all the keywords pertaining to the genre
        :return: (genre, top_ten_keywords)
        """
		# the generator object is not iterable and thus cannot be returned by the reducer,
		# so we transform it into a list
		yield genre, list(word)
	
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
	
	def remove_title_words(self, title):
		"""
		Function that takes a title and eliminates numeral, conjunctions, adverbs etc.
		:param title: the original title
		:return: the processed title
		"""
		#  since all the titles end with the year they were produced in, I eliminate the year
		title = title[:title.find('(')]
		
		# using the nltk package, the title is tokenised and then each token (word) is tagged as a part of speech
		text = nltk.word_tokenize(title)
		tagged = nltk.pos_tag(text, 'universal')
		
		# list of undesirable parts of speech (adposition, adverb, conjunction, determiner/article, particle,
		# punctuation marks, other); more info at https://www.nltk.org/book/ch05.html
		remove_pos = ['ADP', 'ADV', 'CONJ', 'DET', 'PRT', '.', 'NUM', 'X']
		# list of words that have appeared in practice in the final top 10 keywords and which shouldn't be keywords;
		# the tagger fails to identify them as undesirable parts of speech (e.g. 'ii' is considered a noun)
		remove_words = ['a.k.a', '*', 'ii', 'iii']
		final = ''
		
		for i in tagged:
			# if the word is not an undesirable part of speech or an undesirable word, we add it to the result and
			# add a space after it, so the string can be properly split later on
			if i[1] not in remove_pos and i[0] not in remove_words:
				final += (i[0] + ' ')
		
		# the processed title is returned without the last space at the end
		return final[0:-1]


if __name__ == '__main__':
	start = time.time()
	MovieGenreKeywords.run()
	end = time.time()
	print('execution time: ', end - start)
