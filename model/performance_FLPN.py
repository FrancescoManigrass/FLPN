import numpy as np

def printperformance(performance):
    for i in range(4):
        print(perf[i], round(np.mean(performance[:, i]), 2), "+-", round(np.std(performance[:, i]), 2), "max:",
              round(np.mean(performance[:, i]), 2) + round(np.std(performance[:, i])), 2)


#vit
#1518,1975,177 awa2
performance=np.array([[71.0,67.1,82.8,74.1],[69.1,65.4,81.9,72.8],[69.3, 65.0,82.4,72.7]])
perf = ["T1","U","S","ZSL"]
for i in range(4):
    print(perf[i], np.mean(performance[:, i]), "+-", np.std(performance[:, i]), "max:",
          np.mean(performance[:, i]) + np.std(performance[:, i]))


print("--------VIT AWA2 2905,2906,2907 logltn---------")  # 2901 DA RIFARE
performance=np.array([[71.6,41.2,75.6,53.1],[74.3,43.7,83.5,57.4],[74.3,43.7,83.5,57.4]])
perf = ["T1","U","S","ZSL"]
printperformance(performance)



print("--------VIT CUB 2899,2900,2901 logltn---------")  # 2901 DA RIFARE
performance=np.array([[71.6,41.2,75.6,53.1],[74.3,43.7,83.5,57.4],[74.3,43.7,83.5,57.4]])
perf = ["T1","U","S","ZSL"]
printperformance(performance)


print("--------VIT CUB 2902,2903,2904\   ---------")
performance=np.array([[75.2,33.4,54.0,41.3],[70.0,27.0,84.9,41.0],[70.2, 26.9,84.6,40.8]])
perf = ["T1","U","S","ZSL"]
printperformance(performance)

#VIT SUN LOGLTN 2893,2894,2895
import numpy as np

print("--------VIT SUN 2893,2894,2895 logltn---------")
performance=np.array([[75.7,50.6,50.6,50.6],[74.5,46.3,56.0,50.7],[75.8, 49.6,54.7,52.0]])
perf = ["T1","U","S","ZSL"]
printperformance(performance)


print("--------VIT SUN 2896,2897,2898\   ---------")
performance=np.array([[72.4,33.4,54.0,41.3],[69.5,33.2,53.0,40.8],[71.6, 32.6,56.8,41.4]])
perf = ["T1","U","S","ZSL"]
printperformance(performance)









#RESNET101
import numpy as np
#1888,178,177 awa2
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

