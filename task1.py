import time

import nltk
from mrjob.job import MRJob
from mrjob.step import MRStep
import re

# WORD_RE = re.compile(r"[\w']+")  # match words


class MovieGenreKeywords(MRJob):

    def mapper(self, _, line):
        line = line.strip().lower()
        if line.find('"') > -1:
            words = line.split('"')
            words[2] = words[2][1:]
        else:
            words = line.split(',')

        genres = words[2].split('|')

        for genre in genres:
            yield genre, self.remove_title_words(words[1])

    def combiner(self, genre, title):
        yield genre, ','.join(title)

    def reducer_title_string(self, genre, title):
        yield genre, ','.join(title)

    def mapper_2(self, genre, string):
        words = string.split(' ')
        for word in words:
            yield genre, (word, 1)

    def combiner_2(self, genre, lista):
        word_list = {}
        for i in lista:
            if i[0] in word_list:
                word_list[i[0]] += i[1]
            else:
                word_list[i[0]] = i[1]

        yield genre, word_list

    def reducer_2(self, genre, words):
        dct = next(words)
        yield genre, sorted(dct.items(), key=lambda el: el[1], reverse=True)[:10]

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   combiner=self.combiner,
                   reducer=self.reducer_title_string),
            MRStep(mapper=self.mapper_2,
                   combiner=self.combiner_2,
                   reducer=self.reducer_2)
        ]

    def remove_title_words(self, title):
        title = title[:title.find('(')]
        text = nltk.word_tokenize(title)

        tagged = nltk.pos_tag(text, 'universal')

        remove_pos = ['ADP', 'ADV', 'CONJ', 'DET', 'PRT', '.', 'NUM']
        remove_words = ['a.k.a', '*', 'ii']
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