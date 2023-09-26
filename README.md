# Intro-To-ML-and-AI-Lab
 
## Predicting House Prices: A Hands-on Lab with TensorFlow

Welcome back, Developers! 🚀 Remember the TensorFlow magic from our last lecture? Let's put it to use!

### Objective:

By the end of this lab, you'll:
- Set up a neural network model.
- Train this model on sample house data.
- Use this model to predict house prices.

### Setting the Stage

1. **Environment Setup**

   Ensure you've got TensorFlow, NumPy, and scikit-learn ready. If not:

```bash
pip install tensorflow numpy scikit-learn
```

## Instructions

1. **Setting Up Data**

   We've prepared mock house data for you. Copy this into your Python environment:

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

  ## Constructing the Neural Network

Let's craft our own neural network!

1. **Input Layer**:
   - 32 neurons
   - Activation: `ReLU`
   
2. **Hidden Layer**:
   - 16 neurons
   - Activation: `ReLu`
   
3. **Output Layer**:
   - Just 1 neuron for predicting house price

Once you've constructed the model, don't forget to compile it. Use:
   - Loss function: `mean_squared_error`
   - Optimizer: `adam`

3. **Training and Evaluation**

   Train your model using the `X_train` and `y_train` datasets. Once done, evaluate its performance using the test datasets `(X_test and y_test)` and display the resulting `mean squared error`.

   `Hint`: Once you've evaluated your model on the test data, it will return a value. This value is the Mean Squared Error (MSE) between your model's predictions and the true values. Try capturing this return value and formatting it to print a clean  
   MSE result.

5. **Predicting User's House Price**

   Here is the fun part. After building and training your model, prompt the user to enter details about their house: number of rooms, garden size, and distance to the nearest school. Use your model to predict the house price based on these inputs and display the predicted price.

 ****Great job!****
  ****Call an Instructor/TA to check your completed tasks****
 
 
## Challenge 🌟
Can you refine your model to be more accurate? Play around with it! Maybe add more layers, and neurons, or change the activation functions.

Remember, the journey is more important than the destination. Don't hesitate to ask if you're stuck. Happy coding! 🎉
