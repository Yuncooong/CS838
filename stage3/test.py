from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import numpy as np

with open("After_block.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    features = []
    labels = []
    i = 0
    cols = ["lID", "name", "postal code", "phone number"]

    for row in reader:
        features.append(row['ltable_ID'])
        
        train_set_shop = [row['ltable_shop name'], row['rtable_shop name']]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set_shop)#finds the tfidf score with normalization
        shop_sim = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)
  #      features.append(float(shop_sim))
        A = np.array(shop_sim)
        features.append(float(A[:,1]))
        print(float(A[:,1]))
        #print(row['ltable_postal code'])

        if(row['ltable_postal code'] == row['rtable_postal code']):
            features.append(1)
        else:
            features.append(-1)
        
        if(row['ltable_phone number'] == row['rtable_phone number']):
            features.append(1)
        else:
            features.append(-1)
        cols.append(features)
        
print (cols)
