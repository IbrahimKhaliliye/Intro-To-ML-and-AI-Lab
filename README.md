# Intro-To-ML-and-AI-Lab
 
# Machine Learning Lab: Predicting House Prices

In this lab, you will build a simple neural network using TensorFlow to predict house prices based on a set of house features.

## Prerequisites

- Familiarity with Python.
- Basic understanding of TensorFlow and the Keras API.

## Objective

Implement a neural network model to predict house prices and evaluate its performance using mean squared error. Additionally, you'll test your model with user input to predict custom house prices.

## Setup

Make sure you have the following libraries installed:

- TensorFlow
- NumPy
- scikit-learn

You can install them using pip:

```bash
pip install tensorflow numpy scikit-learn
```

## Instructions

1. **Setting Up Data**

   We've provided mock data for this lab. Copy the code below into your Python environment:

   ```python
   import tensorflow as tf
   import numpy as np
   from sklearn.model_selection import train_test_split
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense

   np.random.seed(42)

   # Generate 1000 houses.
   num_houses = 1000

   # Features: rooms, garden size (in sq meters), distance to nearest school (in km)
   rooms = np.random.randint(1, 6, num_houses)
   garden_size = np.random.random(num_houses) * 100
   distance_to_school = np.random.random(num_houses) * 5

   # Formula to generate house prices
   prices = (rooms * 50000) + (garden_size * 300) - (distance_to_school * 4000)

   # Stack features together
   data = np.column_stack((rooms, garden_size, distance_to_school))

   # Split data into training and testing
   X_train, X_test, y_train, y_test = train_test_split(data, prices, test_size=0.2, random_state=42)
   ```

2. **Building the Model**

   Implement a neural network with the following architecture:
   - An input layer with 32 neurons and ReLU activation.
   - A hidden layer with 16 neurons and ReLU activation.
   - An output layer to predict the house price.
   - Use `mean_squared_error` as the loss function and the `adam` optimizer.

3. **Training and Evaluation**

   Train your model with the provided training data. Additionally, evaluate your model on the testing data and print out the mean squared error.

4. **Predicting User's House Price**

   After building and training your model, prompt the user to enter details about their house: number of rooms, garden size, and distance to the nearest school. Use your model to predict the house price based on these inputs and display the predicted price.

## Challenge

Optimize your model to reduce the mean squared error further. Experiment with different architectures, loss functions, or training parameters.

---

**Once you've completed the lab, please show your work to an instructor or TA for review.**
