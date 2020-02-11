import pandas as pd
import torch
import time
import math
import json

# edges: 610 * 9724 = 5,931,640， 100836
# 3000 epoches 420s
start = time.time()
# ratings = pd.read_csv('/Users/fiona/PycharmProjects/pmf/ml-latest-small/ratings.csv')
ratings = pd.read_csv('/Users/fiona/PycharmProjects/pmf/ml-latest-small/train_ratings.csv')
print(ratings.describe())
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
n_users, n_movies = rating_matrix.shape
# Scaling ratings to between 0 and 1, this helps our model by constraining predictions
min_rating, max_rating = ratings['rating'].min(), ratings['rating'].max()
rating_matrix = (rating_matrix - min_rating) / (max_rating - min_rating)

# Replacing missing ratings with -1 so we can filter them out later
rating_matrix[rating_matrix.isnull()] = -1
rating_matrix = torch.FloatTensor(rating_matrix.values)
n_ratings = (rating_matrix!=-1).sum().item()
latent_vectors = 5
user_features = torch.randn(n_users, latent_vectors, requires_grad=True)
user_features.data.mul_(0.01)
movie_features = torch.randn(n_movies, latent_vectors, requires_grad=True)
movie_features.data.mul_(0.01)

proc_time = time.time()

userIdMap = {}
itemIdMap = {}
folder = "ml-latest-small/"
def map_index():
    global userIdMap
    global itemIdMap
    with open(folder + "user.txt", 'r') as input:
        userIdMap = json.loads(input.read())['user']
    with open(folder + "item.txt", 'r') as input:
        itemIdMap = json.loads(input.read())['item']


def eval_accuracy():
    predictions = torch.sigmoid(torch.mm(user_features, movie_features.t()))
    predictions = predictions.mul(max_rating - min_rating).add(min_rating)
    pred_arr = predictions.detach().numpy()
    line_cnt = 0
    correct = 0
    with open(folder + "test_ratings.csv", 'r') as input:
        for line in input:
            if line_cnt > 0:
                lines = line.split(",")
                user_index, item_index, rating = int(lines[0]), int(lines[1]), float(lines[2])
                if rating == pred_arr[user_index, item_index].round():
                    correct += 1
            line_cnt += 1
    return correct/line_cnt

class PMFLoss(torch.nn.Module):
    def __init__(self, lam_u=0.3, lam_v=0.3):
        super().__init__()
        self.lam_u = lam_u
        self.lam_v = lam_v

    def forward(self, matrix, u_features, v_features):
        non_zero_mask = (matrix != -1).type(torch.FloatTensor)
        predicted = torch.sigmoid(torch.mm(u_features, v_features.t()))

        diff = (matrix - predicted) ** 2
        prediction_error = torch.sum(diff * non_zero_mask)

        u_regularization = self.lam_u * torch.sum(u_features.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(v_features.norm(dim=1))

        return prediction_error + u_regularization + v_regularization, prediction_error

pmferror = PMFLoss(lam_u=0.01, lam_v=0.01)
optimizer = torch.optim.Adam([user_features, movie_features], lr=0.005) # optimizer change

max_epoch = 1300
start = time.time()
eval_accuracy()
for step, epoch in enumerate(range(max_epoch)):
    optimizer.zero_grad()
    error = pmferror(rating_matrix, user_features, movie_features)
    loss = error[0]
    # computes dloss/dx for every parameter x which has requires_grad=True
    loss.backward()
    optimizer.step()
    rmse = math.sqrt(error[1] / n_ratings)
    if step % 50 == 0:
        print(f"Step {step}, loss: {loss:.3f}, rmse: {rmse: .4f}, accuracy: {eval_accuracy(): 3f}, time: {(time.time()-start): 2f}s")

train_time = time.time()
print("Accuracy: ", eval_accuracy())
print("Process Time: ", proc_time-start)
print("Training Time: ", train_time-proc_time)
