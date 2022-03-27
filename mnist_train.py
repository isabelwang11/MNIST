import csv
import numpy as np
import math
import pickle as pkl

def format_data():
    with open('mnist_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in csv_reader:
            label, pixels = int(row[0]), row[1:]
            input = []
            for item in pixels:
                sig_rep = float(item) / 255 # sigmoid representation of pixel
                input.append(sig_rep)
            input = np.array([input]) # 1x784 matrix)
            expected_output = [0] * 10
            expected_output[label] = 1
            expected_output = np.array([expected_output]) # 1x10 matrix
            data.append([input, expected_output])
    return data

def sigmoid(num): # sigmoid function
    z = 1/(1 + np.exp(-num))
    return z

def sig_deriv(vA, x): # derivative of sigmoid
    fx = vA(x)
    return fx * (1-fx)

def p_net(A, x, w_list, b_list): # perceptron network
    vA = np.vectorize(A)
    a0 = x
    a_prev = a0
    for layer in range(1, len(w_list)):
        aL = vA(a_prev@w_list[layer]+b_list[layer])
        a_prev = aL
    return aL

def calculate_error(A, y, x, w_list, b_list): # y is expected output, x is input
    a, dots = [x], [None]
    for layer in range(1, len(w_list)):
        dot_L = a[layer-1]@w_list[layer] + b_list[layer]
        dots.append(dot_L)
        a_L = A(dot_L)
        a.append(a_L)
    resultant_vector = (y - a[len(a)-1])[0]
    mag = 0
    for num in resultant_vector:
        mag += (num**2)
    error = 0.5 * mag
    return error

def load_last(infile):
    d = []
    with infile as f:
        while True:
            try:
                a = pkl.load(f)
            except EOFError:
                break
            else:
                d = a
    return d

def generate_wb(network): # generates random weight and bias values
    w_list, b_list = [None], [None]
    for i in range(1, len(network)):
        w_list.append(2 * np.random.rand(network[i-1], network[i]) - 1)
        b_list.append(2 * np.random.rand(1, network[i]) - 1)
    return w_list, b_list

def get_wb(infile): # retrieves w_list and b_list from file
    load = load_last(infile)
    epoch, wlist, blist = 0, [], []
    count = 0
    for pair in load:
        if count % 3 == 0:
            epoch = pair
        elif count % 3 == 1:
            wlist = pair
        else:
            blist = pair
        count += 1
    return epoch, wlist, blist

def train_network(vA, vADeriv, training_set, w_list, b_list, lr): # lr is the learning rate
    for i in range(60000):
        input, output = training_set[i]
        a, dots = [np.array(input)], [None]
        for l in range(1, len(w_list)):  # layer l
            dot_L = a[l - 1] @ w_list[l] + b_list[l]
            dots.append(dot_L)
            a_L = vA(dot_L)
            a.append(a_L)
        deltas, N = [None] * len(a), len(a) - 1
        deltas[N] = vADeriv(vA, dots[N]) * (output - a[N])
        for L in reversed(range(1, N)):
            deltas[L] = vADeriv(vA, dots[L]) * (deltas[L + 1] @ w_list[L + 1].transpose())
        for layer in range(1, len(w_list)):
            b_list[layer] = b_list[layer] + lr * deltas[layer]
            w_list[layer] = w_list[layer] + lr * a[layer - 1].transpose() @ deltas[layer]
    return w_list, b_list

# generate random weights and biases, training
# print("formatting data")
# data = format_data()
# filename = 'wb_matrices.txt'
# outfile = open(filename, 'wb') # use 'wb' to overwrite, 'ab' to append
# network = [784, 300, 100, 10]
# w_list, b_list = generate_wb(network)
# lr = 0.5
# vec_sigmoid, vec_sig_deriv = np.vectorize(sigmoid), np.vectorize(sig_deriv)
# print("starting training")
# for epoch in range(1, 6):
#     w_list, b_list = train_network(vec_sigmoid, vec_sig_deriv, data, w_list, b_list, lr)
#     print(epoch)
#     print("Last weight layer: %s" % str(w_list[-1]))
#     print("Last bias layer: %s" % str(b_list[-1]))
#     matrix_list = [epoch, w_list, b_list]
#     pkl.dump(matrix_list, outfile)
# outfile.close()
# print('wrote to file')

# file input, continue training
print("Retrieving matrices and epoch number")
filename = 'wb_matrices.txt'
infile = open(filename, 'rb')
last_epoch_num, wlist, blist = get_wb(infile)
infile.close()
print("Formatting data")
data = format_data()
outfile = open(filename, 'wb')
lr = 0.5
vec_sig_deriv = np.vectorize(sig_deriv)
print("Starting training")
for epoch in range(last_epoch_num + 1, 11):
    print(epoch)
    w_list, b_list = train_network(sigmoid, vec_sig_deriv, data, wlist, blist, lr)
    matrix_list = [epoch, wlist, blist]
    pkl.dump(matrix_list, outfile)
print('wrote to file')

# The weights and biases on the final layer should change the most.  Check there.  If you're not seeing any change there, up the learning rate.  But MNIST usually works with a really small learning rate, like .01 or .05.
# Look for changes after a whole epoch, not a single data point.