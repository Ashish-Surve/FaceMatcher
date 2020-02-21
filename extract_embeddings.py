import os
import pickle

import cv2
import imutils
from imutils import paths

from Face_recognition_app import getFace


def extract_embeddings_datset():
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("dataset/"))
    print(imagePaths)
    knownEmbeddings = []
    knownNames = []
    total=0
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        print(name)
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        frame = image
        faces = getFace(frame)
        for face in faces:
            cv2.rectangle(frame, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
            person_embedding = face['embedding']
            knownNames.append(name)
            #print(name)
            knownEmbeddings.append(person_embedding.flatten())
            total += 1

        print("[INFO] serializing {} encodings...".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open("output/embeddings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()

if __name__=="__main__":
    extract_embeddings_datset()