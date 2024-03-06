# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A neural network regression model is a type of deep learning model used for predicting continuous numerical values.It refers to neural networks with multiple hidden layers. These deeper architectures enable the network to learn complex patterns and relationships in the data.
### Architecture
This model comprises of a single input and output layer along with two hidden layers of 3 and 5 relu activation layer. So that it predicts the corresponding output for the given set of input. 
### Model
![Exp1](https://github.com/Archana2003-Jkumar/basic-nn-model/assets/93427594/4d4c9943-43d8-4c5d-b3db-6d2100391251)
## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:J.Archana Priya
### Register Number:212221230007
```python

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd



auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('Exp 1 DL').sheet1

data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'INPUT':'float'})


dataset1 = dataset1.astype({'OUTPUT':'float'})
dataset1.head()

import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train1 = scaler.transform(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([ Dense(units=5,activation='relu',input_shape=[1]),
                  Dense(units=3,activation='relu'),
                  Dense(units=1)])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=3300)

lossdf = pd.DataFrame(model.history.history)
lossdf.plot()

X_test1 = scaler.transform(X_test)

model.evaluate(X_test1,y_test)

X_n1 = [[10]]    

X_n1_1 = scaler.transform(X_n1)

model.predict(X_n1_1)
```
## Dataset Information
![image](https://github.com/Archana2003-Jkumar/basic-nn-model/assets/93427594/fb1ddbbe-b614-48cb-81e4-bf18ef81d524)

## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/Archana2003-Jkumar/basic-nn-model/assets/93427594/d0070257-9dcf-468f-ae31-d8d7377232fa)


### Test Data Root Mean Squared Error
![image](https://github.com/Archana2003-Jkumar/basic-nn-model/assets/93427594/90a04236-07d2-4019-9ddf-fff03e94bef5)

### New Sample Data Prediction
![image](https://github.com/Archana2003-Jkumar/basic-nn-model/assets/93427594/2b7cf193-7cee-450f-b9b2-ba38aa26f475)


## RESULT
Thus the neural network regression model for the given dataset has been successfully developed.
