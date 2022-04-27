import math
import re
import nltk
from collections import Counter
WORD_RE = re.compile(r"[\w']+")
str = "Witchcraft V: Dance with the Devil "

text = nltk.word_tokenize(str)
# print(nltk.pos_tag(text, tagset='universal'))
# print(nltk.pos_tag(text))

tagged = nltk.pos_tag(text, 'universal')

print(tagged)

# print("drama".split('|'))
#
# remove_pos = ['ADP', 'ADV', 'CONJ', 'DET', 'PRT', '.', 'NUM']
# final = ''
#
# for i in tagged:
#     if i[1] not in remove_pos:
#         final += (i[0] + ' ')
#
# print(final[0:-2])
#
# # aa = 'ana'
# # bb = 'ma'
# #
# # print(''.join())

# with open('iris_mini.csv', 'r') as inf:
#     for line in inf:
#         # print(line.split(','))
#         for word in WORD_RE.findall(line):
#             print(word)
# WORD_RE = re.compile(r"[\w']+") # match words
# str = '#	11342'
# for word in WORD_RE.findall(str):
#     print(word)
#     print(word.isnumeric())

# def euclidean_distance(arr1, arr2):
#     sum = 0
#     print(arr1)
#
#     for i in range(len(arr1)):
#         sum += (arr1[i] - arr2[i]) ** 2
#     return math.sqrt(sum)
#
# a = [1, 2, 3]
#
# b = [4, 5, 6]
#
# print(euclidean_distance(a, b))

# print(Counter([(0.09999999999999998, "Iris-setosa", 1), (0.1414213562373093, "Iris-setosa", 1),
# 			   (0.14142135623730964, "Iris-setosa", 1), (0.14142135623730995, "Iris-setosa", 1),
# 			   (0.14142135623730995, "Iris-setosa", 1), (0.17320508075688743, "Iris-setosa", 1),
# 			   (0.17320508075688762, "Iris-setosa", 1), (0.22360679774997896, "Iris-setosa", 1),
# 			   (0.30000000000000016, "Iris-setosa", 1), (0.30000000000000027, "Iris-setosa", 1),
# 			   (0.316227766016838, "Iris-setosa", 1), (0.33166247903553986, "Iris-setosa", 1),
# 			   (0.3605551275463989, "Iris-setosa", 1), (0.37416573867739383, "Iris-setosa", 1),
# 			   (0.3741657386773941, "Iris-setosa", 1)], ))
