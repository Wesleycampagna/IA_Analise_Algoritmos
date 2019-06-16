import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

x = [[0.8348672512483253, 0.8508104981122886, 0.8492351723298015, 0.8449500669833151, 0.8593606138107417], [0.8017817561807332, 0.7928126902935086, 0.8319047619047619, 0.855180246011448, 0.8449488491048592], [0.8933145407564013, 0.911498708010336, 0.920589617101245, 0.8955320648343903, 0.920589617101245], [0.8684038054968287, 0.8659772140004698, 0.8840650692976274, 0.9136128729151984, 0.9182616866337796]]

x = np.array(x)
print(x.shape)

tam_dataset = 2
tam_alg = 2

x.shape = (tam_dataset, tam_alg, len(x[0]))

print(x.shape)

print(x)


fig, ax1 = plt.subplots(figsize=(9, 7))
ax1.set_xlim(-1, len(x[0]))
ax1.set_ylim([0, 1])

#ax1.

data1 = x[0]
data2 = x[1]

ar = []


for i in x:
    print(i) 
    for k in range(len(i)):
        print('****')
        print(i[k].mean()) 
        ar.append(i[k].mean())

        print('****')

    ar.append
    

print(ar)

# ax1.plot('date', 'adj_close', data=ar)



plt.xticks(np.arange(0, len(x[0])), ('knn', 'tree', 'nbayes', 'lregression', 'rneurais'))

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