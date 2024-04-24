import numpy as np


class GRU:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicjalizacja wag
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size)
        self.U_z = np.random.randn(hidden_size, hidden_size)
        self.b_z = np.zeros((hidden_size, 1))

        self.W_r = np.random.randn(hidden_size, input_size + hidden_size)
        self.U_r = np.random.randn(hidden_size, hidden_size)
        self.b_r = np.zeros((hidden_size, 1))

        self.W_h = np.random.randn(hidden_size, input_size + hidden_size)
        self.U_h = np.random.randn(hidden_size, hidden_size)
        self.b_h = np.zeros((hidden_size, 1))

        # Warstwa wyjściowa
        self.W_o = np.random.randn(output_size, hidden_size)
        self.b_o = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def update_gate(self, xt, ht_1):
        zt = self.sigmoid(np.dot(self.W_z, np.vstack((xt, ht_1))) + self.b_z)
        return zt

    def reset_gate(self, xt, ht_1):
        rt = self.sigmoid(np.dot(self.W_r, np.vstack((xt, ht_1))) + self.b_r)
        return rt

    def forward(self, x):
        T = x.shape[1]  # Liczba kroków czasowych

        # Inicjalizacja stanu ukrytego
        h = np.zeros((self.hidden_size, T))

        # Pętla przekazywania przez czas
        for t in range(T):
            xt = x[:, t].reshape(-1, 1)
            ht_1 = h[:, t - 1] if t > 0 else np.zeros((self.hidden_size, 1))

            # Bramki
            zt = self.update_gate(xt, ht_1)
            rt = self.reset_gate(xt, ht_1)

            # Aktualizacja stanu komórki
            h_hat_t = self.tanh(np.dot(self.W_h, np.vstack((xt, rt * ht_1))) + self.b_h)
            h[:, t] = (1 - zt) * h_hat_t + zt * ht_1

        # Warstwa wyjściowa
        output = np.dot(self.W_o, h[:, -1]) + self.b_o

        return output, h
