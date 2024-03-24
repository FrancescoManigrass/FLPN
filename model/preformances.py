import numpy as np
#157,178,177 awa2
performance=np.array([[71.0,67.1,82.8,74.1],[69.1,65.4,81.9,72.8],[69.3, 65.0,82.4,72.7]])
perf = ["T1","U","S","ZSL"]
for i in range(4):
    print(perf[i], np.mean(performance[:, i]), "+-", np.std(performance[:, i]), "max:",
          np.mean(performance[:, i]) + np.std(performance[:, i]))






#479,480,481(metto 453 meglio) CUB
print("CUB")
performance=np.array([[71.2,62.0,83.6,71.2],[71.0,62.6 ,82.9,71.4],[71.5,63.4 ,82.9,71.9]])
perf = ["T1","U","S","ZSL"]
for i in range(4):
    print(perf[i],np.mean(performance[:,i]),"+-",np.std(performance[:,i]) , "max:", np.mean(performance[:,i]) +np.std(performance[:,i]))



print("SUN")
#393 , 549 ,548
performance=np.array([[62.0,49.2,32.4,39.1],[61.6,47.9 ,33.0,39.1],[61.6,48.1 ,32.8,39.0]])
perf = ["T1","U","S","ZSL"]
for i in range(4):
    print(perf[i],np.mean(performance[:,i]),"+-",np.std(performance[:,i]) , "max:", np.mean(performance[:,i]) +np.std(performance[:,i]))

