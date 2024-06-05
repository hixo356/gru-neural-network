import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('all_stocks_5yr.csv')

data['date'] = pd.to_datetime(data['date'])

data = data.sort_values('date')

stock_data = data[data['Name'] == 'AAPL']

prices = stock_data['close'].values

prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))


# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 15
X, y = create_sequences(prices, seq_length)


split1 = int(0.7 * len(X))
split2 = int(0.9 * len(X))
X_train, y_train = X[:split1], y[:split1]
X_val, y_val = X[split1:split2], y[split1:split2]
X_test, y_test = X[split2:], y[split2:]


class GRU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.00001, l2_lambda=0.0001, dropout_rate=0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W_z = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size))
        self.b_z = np.zeros((hidden_size, 1))

        self.W_r = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size))
        self.b_r = np.zeros((hidden_size, 1))

        self.W_h = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size))
        self.b_h = np.zeros((hidden_size, 1))

        limit = np.sqrt(6 / (hidden_size + output_size))
        self.W_o = np.random.uniform(-limit, limit, (output_size, hidden_size))
        self.b_o = np.zeros((output_size, 1))
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def tanh(self, x):
        return np.tanh(x)
    def relu(self, x):
        return np.maximum(0, x)
    def forward(self, x):
        T = x.shape[1]
        h = np.zeros((self.hidden_size, T))

        for t in range(T):
            xt = x[:, t].reshape(-1, 1)
            ht_1 = h[:, t - 1].reshape(-1, 1) if t > 0 else np.zeros((self.hidden_size, 1))

            zt = self.sigmoid(np.dot(self.W_z, np.vstack((xt, ht_1))) + self.b_z)
            rt = self.sigmoid(np.dot(self.W_r, np.vstack((xt, ht_1))) + self.b_r)
            h_hat_t = self.relu(np.dot(self.W_h, np.vstack((xt, rt * ht_1))) + self.b_h)
            h[:, t] = ((1 - zt) * h_hat_t + zt * ht_1).reshape(-1)

            # Apply dropout
            if self.dropout_rate > 0:
                h[:, t] *= np.random.binomial(1, 1 - self.dropout_rate, h[:, t].shape) / (1 - self.dropout_rate)

        output = np.dot(self.W_o, h[:, -1].reshape(-1, 1)) + self.b_o
        return output, h

    def compute_loss(self, y_pred, y_true):
        mse = np.mean((y_pred - y_true) ** 2)
        l2_reg = self.l2_lambda * (
                    np.sum(self.W_z ** 2) + np.sum(self.W_r ** 2) + np.sum(self.W_h ** 2) + np.sum(self.W_o ** 2))
        return mse + l2_reg

    def backpropagation(self, x, y_true):
        T = x.shape[1]
        y_pred, h = self.forward(x)

        loss = self.compute_loss(y_pred, y_true)
        dL_dy = 2 * (y_pred - y_true) / y_true.size

        dL_dWo = np.dot(dL_dy, h[:, -1].reshape(1, -1))
        dL_dbo = dL_dy

        self.W_o -= self.learning_rate * dL_dWo
        self.b_o -= self.learning_rate * dL_dbo

        return loss

    def train(self, X_train, y_train, X_val, y_val, epochs=300, early_stopping_rounds=50):
        best_val_loss = float('inf')
        early_stopping_counter = 0
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            loss = 0
            for x, y in zip(X_train, y_train):
                x = x.reshape(-1, self.input_size).T
                y = np.array([[y]])
                loss += self.backpropagation(x, y)
            loss /= len(X_train)
            val_loss, _ = self.evaluate(X_val, y_val, verbose=False)  # Pobierz tylko val_loss
            training_losses.append(loss)
            validation_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_rounds:
                    print(f"Early stopping on epoch {epoch + 1}")
                    break

        plot_loss(training_losses, validation_losses)

    def evaluate(self, X_test, y_test, verbose=True):
        total_loss = 0
        predictions = []
        for x, y in zip(X_test, y_test):
            x = x.reshape(-1, self.input_size).T
            y = np.array([[y]])
            y_pred, _ = self.forward(x)
            total_loss += self.compute_loss(y_pred, y)
            predictions.append(y_pred.item())
        average_loss = total_loss / len(X_test)
        if verbose:
            print(f"Test Loss: {average_loss:.4f}")
        return average_loss, predictions


def plot_loss(training_losses, validation_losses):
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def plot_predictions(y_test, y_pred):
    plt.plot(y_test, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.show()


# Initialize and train GRU model
gru = GRU(input_size=1, hidden_size=32, output_size=1, learning_rate=0.0005, l2_lambda=0.000005, dropout_rate=0.2)
gru.train(X_train, y_train, X_val, y_val, epochs=300, early_stopping_rounds=50)

# Evaluate GRU model and get predictions
test_loss, predictions = gru.evaluate(X_test, y_test)

# Plot actual vs predicted prices
plot_predictions(y_test, predictions)
