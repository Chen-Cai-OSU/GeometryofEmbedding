import sys
from ampligraph.common.aux import rel_rank_stat, precision_format, load_data, eigengap, vmatrix, field_d
import numpy as np
period=1001

lists = []
# lists.append([1,2,3,4,5])
# lists.append([10,20,30,40,50])
# lists.append([50,100,150,200,250])
lists.append([100,200,300,400,500,600])

# lists.append([1,2,3,4,5,6])
# lists.append([1,2,4,8,16,32])
# lists.append([1,3,9,27,81,243])
# lists.append([5,7,11,13,17,19])
# lists.append([11,13,17,41,53,59])
# lists.append([1,3,5,7,9,11])
# lists.append([2,4,6,8,10,12])
# lists.append([2,4,8,16,32,64])

for list in lists:
    print(list)
    print(vmatrix(field_d(list, period)))
sys.exit()

# vary number of rels
rels = [13, 28, 33, 40, 48, 58, 62, 73, 81, 92, 101, 111, 123, 131, 141]
# rels = list(range(10, 150, 10))
rels = list(range(1, 15, 1))
# rels = list(range(10, 150, 10))
for n in range(4,10,1):
    step = 1
    rel_lis = rels[:n]
    distm = field_d(rel_lis,1000)
    print(rel_lis, np.sum(1.0 ,  (1 + distm), len(rels)))
sys.exit()


# fix number of rels
period = 1000
for i in [1, 10, 100]:
    rels = list(range(i, i*7, i))
    vmatrix(field_d(rels, period))
    distm = field_d(rels, period)
    print(rels, np.sum(1.0 ,  (1 + distm)))
sys.exit()
print(vmatrix(field_d([1,2,3,4,5,6], period)))
print(vmatrix(field_d([1, 2, 4, 8, 16, 32], period)))
print(vmatrix(field_d([5, 7, 11, 13, 17, 19], period)))
print(vmatrix(field_d([11, 13, 17, 41, 53, 59], period)))

print(vmatrix(field_d([1, 3, 5, 7, 9], period)))
print(vmatrix(field_d([2,4,6,8,10], period)))


print(vmatrix(field_d([10,20,30,40,50,60], period)))
print(vmatrix(field_d([100,200,300,400,500,600], period)))


sys.exit()


print(vmatrix(field_d([0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960], 1024)))
print(vmatrix(field_d([0, 121, 242, 363, 484, 605, 726, 847, 968], 1089)))


for i in range(5,20):
    lis = list(range(1,i))
    vmatrix(field_d(lis, 1000))



print(vmatrix(field_d([100,200,300,400,500,600], 997)))
print(vmatrix(field_d([10,20,30,40,50,60], 997)))

print(vmatrix(field_d([100,200,300,400,500], 550)))
print(vmatrix(field_d([100,200,300,400,500, 50,150,250,350,450], 550)))

print(vmatrix(field_d([50,150,250,350,450], 1000)))

import networkx as nx
lines = ["1 2 {'weight':3}",
          "2 3 {'weight':27}",
          "3 4 {'weight':3.0}"]
import numpy as np

for i in [10,20,30,40,50,60,70,80,90]:
    rel_choices = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    rel_choices.remove(i)
    print(vmatrix(field_d(rel_choices, 1000)))
    print('\n')

print(vmatrix(field_d([10, 20,30,40,50,60,70,80,90], 1000)))

print(vmatrix(field_d([1,2,3,4,5,6,7,8,9], 1000)))

vmatrix(field_d([4,8,12,16,20,24,28,32,36], 1000))

