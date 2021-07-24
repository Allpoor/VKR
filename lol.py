import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Импорт базы данных
database = pd.read_csv('diploma_5_stable.csv')

#Удаляем первый столбец с датой, он не потребуется
database = database.drop(['Date'], 1)

# Изменяем размерность
n = database.shape[0]
p = database.shape[1]

#Использовать для построения графика перед масштабированием данных
#plt.plot(data['FTSE 100'])

# Создаем из базы данных np.array
database = database.values

# выделяем тестовую и тренировочную базы данных
start_train = 0
end_train = int(np.floor(0.85*n))
start_test = end_train + 1
end_test = n
data_train = database[np.arange(start_train, end_train), :]
data_test = database[np.arange(start_test, end_test), :]

# Масштабирование базы данных
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Количество акций в базе данных
n_stocks = 100

# Количество нейронов
neurons_number_1 = 128
neurons_number_2 = 64
neurons_number_3 = 8
n_target = 1

net = tf.compat.v1.InteractiveSession()

tf.compat.v1.disable_eager_execution()

X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

sigma = 1
weight_init = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_init = tf.zeros_initializer()

#Скрытые веса
Weight_h1 = tf.Variable(weight_init([n_stocks, neurons_number_1]))
bias_h1 = tf.Variable(bias_init([neurons_number_1]))
Weight_h2 = tf.Variable(weight_init([neurons_number_1, neurons_number_2]))
bias_h2 = tf.Variable(bias_init([neurons_number_2]))
Weight_h3 = tf.Variable(weight_init([neurons_number_2, neurons_number_3]))
bias_h3 = tf.Variable(bias_init([neurons_number_3]))

W_out = tf.Variable(weight_init([neurons_number_3, n_target]))
bias_out = tf.Variable(bias_init([n_target]))

# Скрытые слои
h1 = tf.nn.relu(tf.add(tf.matmul(X, Weight_h1), bias_h1))
h2 = tf.nn.relu(tf.add(tf.matmul(h1, Weight_h2), bias_h2))
h3 = tf.nn.relu(tf.add(tf.matmul(h2, Weight_h3), bias_h3))

# Выходной слой
out = tf.transpose(tf.add(tf.matmul(h3, W_out), bias_out))

# Средняя квадратичная ошибка
error = tf.reduce_mean(tf.squared_difference(out, Y))

# Оптимизатор
opt = tf.train.AdamOptimizer().minimize(error)

net.run(tf.global_variables_initializer())

# Создание графика
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.2)
plt.xlabel('Time (days)')
plt.ylabel('Neuron index')
plt.show()

batch_size = 256
error_train = []
error_test = []

epochs = 20
for e in range(epochs):
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]

        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Отображение прогресса
        if np.mod(i, 5) == 0:

            error_train.append(net.run(error, feed_dict={X: X_train, Y: y_train}))
            error_test.append(net.run(error, feed_dict={X: X_test, Y: y_test}))
            print('Error Train: ', error_train[-1])
            print('Error Test: ', error_test[-1])

            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)

error_final = net.run(error, feed_dict={X: X_test, Y: y_test})
print(error_final)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100