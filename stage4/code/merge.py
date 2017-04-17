import csv
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

with open("data_features.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    s_features = []
    for row in reader:
        feature = []
        feature.append(float(row['name']))
        feature.append(float(row['addr']))
        feature.append(float(row['post']))
        feature.append(float(row['phone']))
        s_features.append(feature)

yelp = []
advisor = []
cols = ["shop name", "postal code", "phone number", "street address", "star", "price", "number of reviews"]
with open("advisor_clean.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        advisor.append(row)
with open("yelp_clean.csv", 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        yelp.append(row)

data = []
Matches = []
i = 0
for l_id, r_id in id_pairs:
    pair  = []
    pair.append(i, yelp[r_id][cols[0]], yelp[r_id][cols[1]], yelp[r_id][cols[2]], yelp[r_id][cols[3]], \
    yelp[r_id][cols[4]], yelp[r_id][cols[5]], yelp[r_id][cols[6]], advisor[l_id][cols[0]], \
    advisor[l_id][cols[1]], advisor[l_id][cols[2]], advisor[l_id][cols[3]], advisor[l_id][cols[4]],\
    advisor[l_id][cols[5]], advisor[l_id][cols[6]]);
    Matches.append(pair)
    data.append((i, yelp[r_id][cols[0]], yelp[r_id][cols[1]], \
                 yelp[r_id][cols[2]], yelp[r_id][cols[3]], \
                 yelp[r_id][cols[5]], yelp[r_id][cols[6]], advisor[l_id][cols[4]], \
                 advisor[l_id][cols[5]], advisor[l_id][cols[6]]))
    i += 1

csvfile = file("E.csv", 'wb')
writer = csv.writer(csvfile)
writer.writerow(["ID", "shop name", "postal code", "phone number", "street address",  \
        "yelp price", "yelp number of reviews", \
        "advisor star", "advisor price", "advisor number of reviews"])
writer.writerows(data)
csvfile.close()

csvfile = file("matches.csv", 'wb')
writer = csv.writer(csvfile)
writer.writerows(Matches)
csvfile.close()
