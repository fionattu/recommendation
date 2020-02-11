import pandas as pd
import torch
import time
import math

# edges: 610 * 9724 = 5,931,640， 100836
# 3000 epoches 420s
start = time.time()
folder = "ml-latest-small/"
train_file = "train_ratings.csv"
test_file = "test_ratings.csv"
latent_vectors_dim = 5
max_epoch = 1500
lr = 0.001
lamda = 0.01

# read ratings
ratings = pd.read_csv(folder + train_file)
print(ratings.describe())
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
n_users, n_movies = rating_matrix.shape
min_rating, max_rating = ratings['rating'].min(), ratings['rating'].max()
rating_matrix = (rating_matrix - min_rating) / (max_rating - min_rating)
rating_matrix[rating_matrix.isnull()] = -1
rating_matrix = torch.FloatTensor(rating_matrix.values)
n_ratings = (rating_matrix!=-1).sum().item()

# init latent factors
user_features = torch.randn(n_users, latent_vectors_dim, requires_grad=True)
user_features.data.mul_(0.01)
movie_features = torch.randn(n_movies, latent_vectors_dim, requires_grad=True)
movie_features.data.mul_(0.01)

proc_time = time.time()

def eval_accuracy():
    predictions = torch.sigmoid(torch.mm(user_features, movie_features.t()))
    predictions = predictions.mul(max_rating - min_rating).add(min_rating).detach().numpy().round()
    line_cnt = 0
    correct = 0
    with open(folder + test_file, 'r') as input:
        for line in input:
            if line_cnt > 0:
                lines = line.split(",")
                user_index, item_index, rating = int(lines[0]), int(lines[1]), float(round(float(lines[2])))
                if rating == predictions[user_index, item_index]:
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

pmferror = PMFLoss(lam_u=lamda, lam_v=lamda)
optimizer = torch.optim.Adam([user_features, movie_features], lr=lr) # optimizer change

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
