import numpy as np
import matplotlib.pyplot as plt


N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.4       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [12674, 8629, 7656, 7037, 6800, 6775, 6800, 6774, 6794, 6798]

rects1 = ax.bar(ind, yvals, width, color='r')
plt.title('Features vs Mean Square Error')
ax.set_xlabel('Features')
ax.set_ylabel('Mean Square Error')
ax.set_xticks(ind+width)
ax.set_xticklabels( ("DTreeClassifier","DTreeRegressor","Lasso alpha=0.1","Lasso alpha=0.01+3000 iter","Unigrams from NER","Lasso alpha=0.01+1000 iter","Lasso with RTRegressor","Countries+Years+Numbers","Only Countries Feature","NER+Countries"), rotation=10)
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
plt.show()