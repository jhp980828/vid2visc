import torch
from torch.utils.data import DataLoader
import os
from os.path import dirname, basename, join
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils import *
from train_reg_aug import load_saved_FCN



### paths:str
modelpath = "../FINAL_AE/best_clf512/AE.ckpt"
train_data_path = "../processed_data/clf_dataset_aug_Train_data.pt"
train_label_path = "../processed_data/clf_dataset_aug_Train_label.pt"
test_data_path = "../processed_data/clf_dataset_aug_Test_data.pt"
test_label_path = "../processed_data/clf_dataset_aug_Test_label.pt"
model_config = modelpath.split ("/")[-2] # ex) "best_clf512"
print (model_config)

### Load Model

autoencoder, saved_loss = load_saved_AE (modelpath, device="cpu")
autoencoder.eval()
print (f"Saved loss on testset: {saved_loss:1.4f}")


### Load data
train_data = torch.load(train_data_path)
train_label = torch.load(train_label_path)
test_data = torch.load (test_data_path)
test_label = torch.load (test_label_path)
print (train_data.shape)
print (test_data.shape)

train_latent_vectors = encode_processed_data (autoencoder, train_data, batch_size=10, shuffle=False, num_workers=1)
test_latent_vectors = encode_processed_data (autoencoder, test_data, batch_size=10, shuffle=False, num_workers=1)
train_latent_vectors = arr(train_latent_vectors)
test_latent_vectors = arr(test_latent_vectors)
train_label = arr(train_label)
test_label = arr(test_label)
# print (train_label.shape, test_label.shape)
print ("Train: ", train_latent_vectors.shape)
print ("Test: ", test_latent_vectors.shape)


# fit LR
logreg = LogisticRegression(penalty='l2', tol=1e-3, C=1, random_state=None, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None)
classifier = logreg.fit(train_latent_vectors, train_label)

# test_latent_vectors, test_label_skip
test_prediction = classifier.predict (test_latent_vectors)
print (f"Test predictions: {test_prediction.shape}")

# 5 fluid classes used coffee, dish, mango, oil, water
class_names = ['Coffee', 'Dish', 'Mango', 'Oil', 'Water']

# confusion matrix
cm = confusion_matrix(test_label, test_prediction)
plt.figure(figsize=(8,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prediction',fontsize=14)
plt.ylabel('Label', fontsize=14)
plt.title ("Confusion Matrix", fontsize=16)
# plt.savefig ("./cm.jpg")
plt.show()

# classification accuracy on testset
print (f"Train score: {classifier.score(train_latent_vectors, train_label)}")
print (f"Test score: {classifier.score(test_latent_vectors, test_label)}")