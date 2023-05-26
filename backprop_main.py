import backprop_data

import backprop_network

import matplotlib.pyplot as plt

training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)

net = backprop_network.Network([784, 40, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# question 1.b
# learning_rates = [0.001, 0.01, 0.1, 1, 10, 100]

# fig, axs = plt.subplots(3, 1, constrained_layout=True)

# axs[0].set_title('Training accuracy as function of epoch number')
# axs[0].set_xlabel('epoch')
# axs[0].set_ylabel('accuracy')

# axs[1].set_title('Training loss as function of epoch number')
# axs[1].set_xlabel('epoch')
# axs[1].set_ylabel('loss')

# axs[2].set_title('Test accuracy as function of epoch number')
# axs[2].set_xlabel('epoch')
# axs[2].set_ylabel('accuracy')

# for rate in learning_rates:
#     net = backprop_network.Network([784, 40, 10])
#     epochs, test_acc, training_acc, training_loss = net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=rate, test_data=test_data)
#     axs[0].plot(epochs, training_acc, label="rate={0}".format(rate))
#     axs[1].plot(epochs, training_loss, label="rate={0}".format(rate))
#     axs[2].plot(epochs, test_acc, label="rate={0}".format(rate))

# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
# fig.set_figheight(10)
# fig.set_figwidth(10)
# plt.show()

# question 1.c
print("Question 1c")
training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
net = backprop_network.Network([784, 40, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data, statistics=False)