'''
Code from CS244n Class

Time diff between For Loop and Mat Mul
'''

from numpy import random
N = 500  # num of windows to classify
d = 300  # dim of each windows / word vectors
C = 5  # 5 classes
W = random.rand(C, d)  # Weights for the classes

# if it is a list 
wordvectors_list = [random.rand(d, 1) for i in range(N)]

# if it is a matrix 
wordvectors_in_1_matrix = random.rand(d, N)

# %timeit [W.dot(wordvectors_list[i]) for i in range(N)]
# %timeit W.dot(wordvectors_in_1_matrix)


