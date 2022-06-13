#The motive is to derive the best fit line uisng m and b

import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0 #m and b
    iterations = 10000 #no. of steps going to take for reaching the minima
    n = len(x) #n = length of the datapoints

    #for lean_rate = 0.1 ,  the cost function rate will increase whihc is crossing the global minima and shooting in the other direction
    learn_rate = 0.08 #random value , trial and error testing case

    for i in range(iterations):
        y_pred = m_curr * x + b_curr # y = mx+b

        #Cost function calculation
        cost = (1/n) * sum([val ** 2 for val in (y-y_pred)])

        #calculating the m derivative
        md = -(2/n) * sum(x*(y-y_pred))

        #calculating the b derivative
        bd = -(2/n) * sum(y-y_pred)

        #calculting the value of m and b

        m_curr = m_curr - learn_rate * md
        b_curr = b_curr - learn_rate * bd

        print("The value of m : {}, the value of b : {} ,the cost is :{},  and the number of iterations : {}".format(m_curr,b_curr,cost,i))

#using numpy array for faster computations tthan python lists
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)