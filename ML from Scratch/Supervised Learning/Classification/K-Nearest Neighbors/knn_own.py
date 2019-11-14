import numpy as np
from math import sqrt
import matplotlib.pyplot as plt 
from matplotlib import style 
from collections import Counter 

style.use('fivethirtyeight') 

dataset = {'blue': [[1,2],[2,3],[2,4],[3,1],[2,2],[3,3]] , 'red': [[6,5],[7,7],[8,6],[7,5],[6,6]] }
 
def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        print('K is set to a value less than the total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidian_distance = np.linalg.norm(np.array(features)- np.array(predict))
            distances.append([euclidian_distance, group])
            #print(distances)

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(votes)
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = ((Counter(votes).most_common(1)[0][1]) / k )

    return vote_result , confidence    


new_feature = [1,1]    # test data to predict the color grp it will belong (blue or red)
 
result , confidence = k_nearest_neighbors(dataset , new_feature )
print("\nPrediction = {}\nConfidence is {} ".format(result, confidence*100))


[[plt.scatter(ii[0],ii[1], s = 100 , color = i ) for ii in dataset[i]] for i in dataset]   
plt.scatter(new_feature[0],new_feature[1], s =200, color= result)
plt.show()
