import csv
import numpy as np
import pickle as pkl

def sigmoid(num): # sigmoid function
    z = 1/(1 + np.exp(-num))
    return z

def p_net(A, x, w_list, b_list): # perceptron network
    vA = np.vectorize(A)
    a0, aL = x, []
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

def format_data():
    with open('mnist_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        line_count = 0
        for row in csv_reader:
            line_count += 1
            label, pixels = int(row[0]), row[1:]
            input = []
            for item in pixels:
                sig_rep = float(item) / 255 # sigmoid representation of pixel
                input.append(sig_rep)
            input = np.array([input]) # 1x784 matrix)
            expected_output = [0] * 10
            expected_output[label] = 1
            expected_output = np.array([expected_output]) # 1x10 matrix
            data.append([input, expected_output, label])
    return data

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

def round_result(result):
    arr, rounded = result[0], [0]*10
    max_val, max_index = 0.0, 0
    for i in range(10):
        if arr[i] > max_val:
            max_val = arr[i]
            max_index = i
    return max_index

data = format_data()
filename = 'wb_matrices.txt'
total_pts = 10000.0
while True:
    infile = open(filename, 'rb')
    epoch, w_list, b_list = get_wb(infile)
    infile.close()
    misclassified = 0
    for input, expected_output, label in data:
        result = p_net(sigmoid, input, w_list, b_list)
        rounded = round_result(result)
        if label != rounded:
            misclassified += 1
        # error = calculate_error(sigmoid, expected_output, input, w_list, b_list)
        # cum_error += error
    percent_misclassified = (misclassified / total_pts) * 100
    print("EPOCH %s: %s percent misclassified" % (str(epoch), str(percent_misclassified)))