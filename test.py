import pandas as pd
import numpy as np
import sklearn.model_selection as ms

train_data = pd.read_csv("data/train.csv")[["id", "species"]]

print(train_data.head())
print("------")

test_idxs = [1,7,3,12]

print(train_data.loc[test_idxs])
print("------")

print(len(train_data))
print(train_data.shape)
print("------")

n = 7
k = 3

rng = np.random.default_rng()

valid = rng.choice(n, k, replace=False)
train = np.delete(np.arange(n), valid)


print(valid)
print(train)
print("------")

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

kf = ms.KFold(n_splits = 3)

for train_idxs, test_idxs in kf.split(X):
    print(train_idxs)
    print(test_idxs)


class Foo:
    def __init__(self) -> None:
        self.a = 1


foo = Foo()
print(foo.a)
print(foo.b)

