import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Download the data
df = yf.download('AAPL', start='2020-01-01', end='2025-03-10')  # Use a larger date range

# Check if data is downloaded correctly
if df.empty:
    print("No data found for the specified date range.")
else:
    print("Data downloaded successfully.")

    # Dataframe 
    data = df.filter(['Close'])
    # convert the dataframe 
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * .8)

    # Print the training data length
    print(f"Training data length: {training_data_len}")

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]

    # Ensure there are enough data points
    if len(train_data) < 60:
        print("Not enough data to create training set.")
    else:
        # Split the data into x_train and y_train data sets
        x_train = [] # Independent training variables
        y_train = [] # Dependent training variables

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        
        # Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data to be 3-dimensional
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))  # Add dropout to prevent overfitting
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10)

        # Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2, callbacks=[early_stop])  # Increase epochs and add validation split

        # Create the testing data set
        test_data = scaled_data[training_data_len - 60:, :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data 
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get the model's predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Get the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean((predictions - y_test)**2))
        print(f"Root Mean Squared Error: {rmse}")

        # Plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()

        # Show the valid and predicted prices
        print(valid)

        # Get the quote
        apple_quote = yf.download('AAPL', start='2020-01-01', end='2025-01-20')

        # Create a new dataframe
        new_df = apple_quote.filter(['Close'])

        # Get the last 60 day closing price values and convert the dataframe to an array
        last_60_days = new_df[-60:].values

        # Scale the data to be values between 0 and 1
        last_60_days_scaled = scaler.transform(last_60_days)

        # Create an empty list
        X_test = []

        # Append the past 60 days
        X_test.append(last_60_days_scaled)
        
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)

        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Get the predicted scaled price
        pred_price = model.predict(X_test)

        # Undo the scaling
        pred_price = scaler.inverse_transform(pred_price)
        print(f"Predicted price: {pred_price}")

        # Get the actual price
        actual_price = yf.download('AAPL', start='2025-01-12', end='2025-01-12')
        print(f"Actual price: {actual_price['Close']}")

        # Get the quote
        apple_quote2 = yf.download('AAPL', start='2020-01-01', end='2025-03-10')
        new_df2 = apple_quote2.filter(['Close'])

