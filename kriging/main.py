import csv
import exceptions
from kriging import KrigingModel

fileReader = csv.reader(open('2d-test.csv', 'rb'), delimiter = ',')
X=[]
Y=[]
for row in fileReader:
    #print ', '.join(row)
    try:
        x1 = map(float, row)
        y1 = x1.pop(-1)
        X.append(x1)
        Y.append(y1)
    except exceptions.ValueError:
        #print row
        pass
        
k = KrigingModel(X,Y)
#print k.condR
#print k.processVariance


        
