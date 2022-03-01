import csv
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as mp
import numpy as np
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("data.csv")

velocity = df["Velocity"].tolist()

escaped = df["Escaped"].tolist()

fig = px.scatter(x = velocity, y = escaped)
#fig.show()


x = np.reshape(velocity, (len(velocity) , 1))
y = np.reshape(escaped,(len(escaped),1))

lr = LogisticRegression()
lr.fit(x,y)

mp.figure()
mp.scatter(x.ravel() , y , color = 'blue' , zorder = 20)

def model(x):
    return 1/(1 + np.exp(-x))
 
x_test = np.linspace(0,100,200)
chances = model(x_test*lr.coef_ + lr.intercept_).ravel()

mp.plot(x_test , chances , color="red" , linewidth = 3)
mp.axhline(y=0 , color='k' ,linestyle = '-')
mp.axhline(y=1 , color='k' ,linestyle = '-')
mp.axhline(y=0.5 , color='b' ,linestyle = '--')


mp.axvline(x=x_test[23] , color='b' , linestyle='--')


mp.ylabel('y')
mp.xlabel('x')
mp.xlim(0,30)
#mp.show()


user_velocity = float(input("Please enter velocity: "))
chance = model(user_velocity*lr.coef_ + lr.intercept_).ravel()

if chance <= 0.01 :
    print("Too slow!!😁😁😁😁")
elif chance >= 1:
    print("Go'in great, but don't over speed")
elif chance < 0.5:
    print("Just a chance")
else:
    print("Now It's upon how you do")
  