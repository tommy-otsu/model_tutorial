from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

heacet = load_iris()

print("与えられたデータ")
print(heacet.data)
print(heacet.data.shape)
print("-----------------")
print("予測するデータ")
print(heacet.target)
print(heacet.target.shape)
print(heacet.target_names)