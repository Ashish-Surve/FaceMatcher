from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/labelencoder.pickle", "wb")
f.write(pickle.dumps(le))
f.close()