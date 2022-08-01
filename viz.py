import csv
import numpy as np
import matplotlib.pyplot as plt

with open('logs/mnist-random/train_acc.csv', 'r') as f:
    rand_train_acc = np.array(list(csv.reader(f)))[:, -1].astype(float)

with open('logs/mnist-random/test_acc.csv', 'r') as f:
    rand_test_acc = np.array(list(csv.reader(f)))[:, -1].astype(float)

with open('logs/mnist-gfn/train_acc.csv', 'r') as f:
    gfn_train_acc = np.array(list(csv.reader(f)))[:, -1].astype(float)

with open('logs/mnist-gfn/test_acc.csv', 'r') as f:
    gfn_test_acc = np.array(list(csv.reader(f)))[:, -1].astype(float)

with open('logs/mnist-gfn-valid/train_acc.csv', 'r') as f:
    gfn_valid_train_acc = np.array(list(csv.reader(f)))[:, -1].astype(float)

with open('logs/mnist-gfn-valid/test_acc.csv', 'r') as f:
    gfn_valid_test_acc = np.array(list(csv.reader(f)))[:, -1].astype(float)

step = np.arange(len(rand_train_acc)) * 100
plt.plot(step, rand_train_acc, label='rand-train-acc')
plt.plot(step, rand_test_acc, label='rand-test-acc')
plt.plot(step, gfn_train_acc, label='gfn-train-acc')
plt.plot(step, gfn_test_acc, label='gfn-test-acc')
step = np.arange(len(rand_train_acc)) * 90
plt.plot(step, gfn_valid_train_acc, label='gfn(valid)-train-acc')
plt.plot(step, gfn_valid_test_acc, label='gfn(valid-test-acc')
plt.legend()
plt.tight_layout()
plt.savefig('viz.pdf')
plt.show()
