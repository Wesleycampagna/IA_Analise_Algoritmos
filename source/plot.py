import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

x = [[0.7814814814814814, 0.8259259259259257, 0.8185185185185185, 0.7851851851851851, 0.837037037037037], [0.7629629629629628, 0.7703703703703704, 0.7888888888888889, 0.7481481481481481, 0.8296296296296296], [0.8333333333333333, 0.8333333333333333, 0.8333333333333333, 0.8333333333333333, 0.8333333333333333], [0.5868498168498169, 0.5925641025641026, 0.587948717948718, 0.612014652014652, 0.6437362637362638], [0.5653479853479854, 0.5545421245421245, 0.6242857142857143, 0.5386813186813187, 0.5776556776556776], [0.5647985347985347, 0.5647985347985347, 0.5647985347985347, 0.5647985347985347, 0.5647985347985347]]
x = np.array(x)
print(x.shape)

tam_dataset = 2
tam_alg = 3

x.shape = (tam_alg, tam_dataset, len(x[0]))

print(x.shape)

print(x)


fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.set_xlim(-1, tam_alg)
ax1.set_ylim([0, 1])

#ax1.

data1 = x[0]
data2 = x[1]

ar = []


for i in x:
    x = []
    for k in i:
        x.append(k.mean())
        
    print('x: ', x)
    ar.append(x)
    

print('ar: ', ar)

ar = np.array(ar)

for i in ar:
    ax1.plot(np.arange(tam_alg), ar, 'o-')

data = 'mean of accuracy'
plt.title(data)
plt.xlabel('algoritmos')
plt.ylabel('y')
plt.legend()
plt.xticks(np.arange(0, tam_alg), ('knn', 'tree', 'nbayes', 'lregression', 'rneurais'))
plt.savefig('output-files/'+ data + '.png')
plt.show()






""" fig, ax1 = plt.subplots(figsize=(9, 7))

for means in all_means:
    ax1.set_xlim(-1, 5 + 1)
    ax1.set_ylim([0, 1])

    plt.xticks(np.arange(0, len(all_means)), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    plt.plot(means.mean(), , linewidth=0.4, color='#777777')
    plt.plot(x, y, linewidth=0.4, color='lightred')
    plt.ylabel('mean')
    plt.xlabel('generation')

    plt.title('Observação fitness médio e fitness melhor individuo x generation')        
    plt.title('fitness médio de todas as repetições e  maior fitness médio de todas as repetições x gerações')
    
    plt.savefig('output-files/media-media-var-e-acuracia.png')
    else: plt.savefig('geneticFiles/fitness.png')

plt.show() """