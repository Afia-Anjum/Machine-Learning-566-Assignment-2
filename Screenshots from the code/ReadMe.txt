
Report the errors as well:

2)a) If the number of feature is increased upto all the given features, then Singular matrix 
error occurs but if the number of feature is less than 69 then no such error occurs; singular 
matrix is a square matrix that does not have it's matrix inverse. That is if the value of it's 
determinant is equal to zero. The reason for this error is that the lower part of the matrix 
becomes full of zeros which makes the matrix not to be a full rank matrix. Also, when we try to
take inverse of a matrix whose inverse is not available, we will get such kind of error. 

The solution is that, we can think of taking the pseudo inverse of that matrix but this pseudo 
inverse will not have all the properties of an ordinary inverse. In python, we can use 
"np.linalg.pinv" to calculate pseudo inverse. The another solution is that, we can also add 
a regularizer which would deal with this sparse matrix and provide us a solution.

2)b) The code is modified to report the standard error in the line 108-109 of the 
script_regression.py with:

stand_error=np.std(errors[learnername][p, :]) / math.sqrt(numruns)
print('Standard error for '+ learnername + ' on parameters ' + str(learner.getparams())+' is '+str(stand_error))

2)c) By applying Ridge regression, we no longer have to use the pseudo inverse of the matrix, 
rather we would specify a regularizer parameter and use the ordinary inverse of that matrix 
to get the correct result. The reason is that due to the addition of regularizer parameter, 
our solution does not produce any singular matrix even with all the features. 
    For lambda=0.01,
      runs   error   standard_error  Avg_error


2)e) The algorithm is implemented and the error is saved on the lines 214-216 of the 
regressionalgorithms.py script.

2)f) The algorithm is implemented inside regressionalgorithms.py script. 
 
With the defined values of different parameters provided in the notes, as per my implementation,
the entire training set is processed 180 times inside the while loop of batch gradient descent 
and maximum iteration of the line search is 100. So in total, the entire training set is going 
to process for 180*100=18000 times. And for the stochastic gradient descent, the entire 
training set is processed is equal to the number of times of it's epochs. The error versus 
convergence plot is also provided for the batch gradient descent in the Screenshot folder with 
the caption "BatchGradientDescent_Convergence_Error_Q_2(f).png".  

The error versus epoch is reported with a graph and screenshot is provided in the Screenshot 
folder with the caption "StochasticGradientDescent_Epoch_Error_Q_2(e).png". 

The error versus runtime with a graph and screenshot is provided in the Screenshot folder
with the caption "StochasticGradientDescent_Runtime_Error_Q_2(f).png"

 