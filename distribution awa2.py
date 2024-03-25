from collections import Counter
import matplotlib.pyplot as plt

# Dati di esempio
distribuzione_classi = Counter({39: 1084, 32: 1069, 22: 961, 30: 942, 24: 877, 0: 864, 16: 849, 34: 821, 6: 815, 37: 809, 17: 709, 10: 702, 29: 698, 31: 692, 1: 690, 36: 684, 21: 640, 5: 599, 28: 592, 18: 588, 12: 568, 20: 566, 15: 565, 23: 561, 33: 558, 13: 554, 11: 543, 19: 540, 25: 470, 26: 448, 4: 442, 38: 408, 7: 400, 14: 237, 2: 231, 27: 222, 8: 157, 3: 152, 35: 145, 9: 75})

# Estrai classi e frequenze
classi = list(distribuzione_classi.keys())
frequenze = list(distribuzione_classi.values())

# Crea il grafico a barre
plt.figure(figsize=(12, 6))
plt.bar(classi, frequenze, color='skyblue')

# Aggiungi titoli e label
plt.title('Distribuzione delle classi')
plt.xlabel('Classi')
plt.ylabel('Frequenza')

# Mostra il grafico
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("awa2_distribution.pdf")
plt.show()
