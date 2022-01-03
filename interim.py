import re

import nltk

str = "B*A*P*S (1997),Comedy"

text = nltk.word_tokenize(str)
# print(nltk.pos_tag(text, tagset='universal'))
# print(nltk.pos_tag(text))

tagged = nltk.pos_tag(text, 'universal')

print(tagged)

print("drama".split('|'))
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

# with open('movies_mini.csv', 'r') as inf:
#     for line in inf:
#         if(line.find('"') > -1):
#             print(line.split('"'))

WORD_RE = re.compile(r"[\w']+") # match words
str = '#	11342'
for word in WORD_RE.findall(str):
    print(word)
    print(word.isnumeric())