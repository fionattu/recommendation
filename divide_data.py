import random
import json
folder = "ml-latest-small/"

# 1. reindex user and item for ease of evaluation
userIdMap = {}
userIndex = 0
itemIdMap = {}
itemIndex = 0
line_cnt= 0
with open(folder + "ratings_index.csv", 'w') as output:
    with open(folder + "ratings.csv", 'r') as input:
        first_line = ""
        for line in input:
            if line_cnt > 0:
                lines = line.split(",")
                userId, itemId, ratings = lines[0], lines[1], lines[2]
                if not userIdMap or userId not in userIdMap:
                    userIdMap[userId] = userIndex
                    userIndex += 1
                if not itemIdMap or itemId not in itemIdMap:
                    itemIdMap[itemId] = itemIndex
                    itemIndex += 1
                output.write(str(userIdMap[userId]) + "," + str(itemIdMap[itemId]) + "," + str(ratings) + "\n")
            else:
                output.write("userId,movieId,rating\n")
            line_cnt+=1

user = {"user": userIdMap}
with open(folder + "user.txt", 'w') as output:
    output.write(json.dumps(user))

item = {"item": itemIdMap}
with open(folder + "item.txt", 'w') as output:
    output.write(json.dumps(item))

# 2. divide data, ensure userIds and itemIds of testing dataset appear in training dataset
users = []
items = []
line_cnt = 0
with open(folder + "train_ratings.csv", 'w') as output1:
    with open(folder + "test_ratings.csv", 'w') as output2:
        with open(folder + "ratings_index.csv", 'r') as input:
            for line in input:
                if line_cnt == 0:
                    output1.write(line)
                    output2.write(line)
                else:
                    lines = line.split(",")
                    user, item = lines[0], lines[1]
                    if random.random() < 0.2 and user in users and item in items:
                        output2.write(line)
                    else:
                        if user not in users:
                            users.append(user)
                        if item not in items:
                            items.append(item)
                        output1.write(line)
                line_cnt += 1

