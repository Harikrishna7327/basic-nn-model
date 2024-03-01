# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

In this model we will discuss with a neural network with 3 layers of neurons excluding input . First hidden layer with 5 neurons , Second hidden layer with 4 neurons and final Output layer with 1 neuron to predict the regression case scenario.

we use the relationship between input and output which is output = input * 2 + 1 and using epoch of about 50 to train and test the model and finnaly predicting the output for unseen test case.

## Neural Network Model

![Screenshot 2024-02-27 103619](https://github.com/Gchethankumar/basic-nn-model/assets/118348224/c975abd8-2403-4150-b715-192359fe52cc)


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
```
Name: M.Harikrishna.
Reg.no: 212221230059 .
```

### Dependenices:
``` python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```
### Data from sheets:
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```

### Data visulization:
```python
df = df.astype({'Input':'int'})
df = df.astype({'Output':'int'})
df.head()
X = df[['Input']].values
Y = df[['Output']].values
```

### Data split and preprocessing:
```python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)
```
### Regressive model:
```python

ai=Sequential([Dense(5,input_shape=[1]),
               Dense(4,activation='relu'),
Dense(1)])
ai.compile(optimizer="rmsprop",loss="mse")
ai.fit(X_train1,Y_train,epochs=50)
```
### Loss Calculation:
```python
loss_df=pd.DataFrame(ai.history.history)
loss_df.plot()
```
### Evaluate the model:
```python
ai.evaluate(X_test,Y_test)
```
### Prediction
```python
X_n1 = [[5]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```
## Dataset Information:

![Screenshot 2024-02-26 212810](https://github.com/Gchethankumar/basic-nn-model/assets/118348224/d0c33e13-7304-40c4-a61c-c769665fb3cc)

## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-02-26 212934](https://github.com/Gchethankumar/basic-nn-model/assets/118348224/fcc30a02-1362-462a-82f6-21c848c6d4be)

### Architecture and Training:

![Screenshot 2024-02-26 213134](https://github.com/Gchethankumar/basic-nn-model/assets/118348224/dc7b5f6a-34f2-490f-9436-12311d541ff5)

### Test Data Root Mean Squared Error

![Screenshot 2024-02-26 213420](https://github.com/Gchethankumar/basic-nn-model/assets/118348224/acebf048-f433-4c1e-873e-fda3002fb0b3)

![Screenshot 2024-02-26 213505](https://github.com/Gchethankumar/basic-nn-model/assets/118348224/d05c80e2-71c2-4aa6-9265-6ee70f28b78f)

### New Sample Data Prediction

![Screenshot 2024-02-26 213551](https://github.com/Gchethankumar/basic-nn-model/assets/118348224/5163cdb9-5d44-4f90-b8b7-decfa21a1705)

## RESULT

Summarize the overall performance of the model based on the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.
