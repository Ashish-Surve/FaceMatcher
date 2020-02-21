import os
import pickle
import time

import imutils
import tensorflow as tf
import numpy as np
from imutils import paths
from imutils.video import VideoStream

import facenet
from align import detect_face
import cv2

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

##########################
face_embedding_dict = {}


##########################
def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append(
                    {'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'embedding': getEmbedding(prewhitened)})
    return faces


def getEmbedding(resized):
    reshaped = resized.reshape(-1, input_image_size, input_image_size, 3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def compare2faces_decorator(func):
    def inner(img1, img2):
        # img1 = cv2.imread(img1)
        imgg1 = imutils.resize(img1, width=1000)
        # img2 = cv2.imread(img2)
        imgg2 = imutils.resize(img2, width=1000)
        return func(imgg1, imgg2)

    return inner


@compare2faces_decorator
def compare2face(img1, img2):
    face1 = getFace(img1)
    face2 = getFace(img2)

    ##########################################ash
    for face in face1:
        cv2.rectangle(img1, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
    cv2.imshow("faces", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for face in face2:
        cv2.rectangle(img2, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
    cv2.imshow("faces", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #############################################
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return dist
    return -1


def extract_embeddings_datset():
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("dataset/"))
    #print(imagePaths)
    knownEmbeddings = []
    knownNames = []
    total=0
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
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


        # cv2.imshow("faces", frame)


def addperson():
    print("[INFO] starting video stream, please wait...")
    vs = VideoStream(src=0).start()
    time.sleep(3)
    global frame
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 1000 pixels

        frame = vs.read()
        # frame = imutils.resize(frame, width=1000, height=10000)
        faces = getFace(frame)
        for face in faces:
            cv2.rectangle(frame, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
        cv2.imshow("faces", frame)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, break from the loop and use last frame to extract embeddings
        if key == ord("q"):
            break

    # cleanup and closing frame
    cv2.destroyAllWindows()
    vs.stop()
    person_embedding = getEmbedding(frame)
    name = input("Enter Name for the person")
    # face_embedding_dict[name]=person_embedding
    print(person_embedding)
    print(type(person_embedding))
    ##########################

    knownEmbeddings = []
    knownNames = []

    total = 0
    knownNames.append(name)
    knownEmbeddings.append(person_embedding.flatten())

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("output/embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

def findPersonImage(img):
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/labelencoder.pickle", "rb").read())
    frame=img
    faces = getFace(frame)
    for face in faces:
        cv2.rectangle(frame, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
        person_embedding = face['embedding']
        preds = recognizer.predict_proba(person_embedding)
        j = np.argmax(preds)
        proba = preds[0][j]
        name = le.classes_[j]
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = face['rect'][1] - 10 if face['rect'][1] - 10 > 10 else face['rect'][1] + 10
        cv2.putText(img, text, (face['rect'][0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cleanup and closing frame







def findPersonVideo():
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/labelencoder.pickle", "rb").read())
    print("[INFO] starting video stream, please wait...")
    vs = VideoStream(src=0).start()
    time.sleep(3)

    global frame
    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        #person_embedding = getEmbedding(frame)
        #frame = imutils.resize(frame, width=1000, height=10000)
        faces = getFace(frame)
        print(len(frame[0]))
        for face in faces:
            cv2.rectangle(frame, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
            person_embedding = face['embedding']
            preds = recognizer.predict_proba(person_embedding)
            j = np.argmax(preds[0])
            # if(j>=len(preds)):
            #     continue
            proba = preds[0][j]
            name = le.classes_[j]
            #print("NAME:"+str(name))
            text = "{}: {:.2f}%".format(name, proba * 100)
            y= face['rect'][1] - 10 if face['rect'][1] -10 >10 else face['rect'][1] +10
            cv2.putText(frame, text, (face['rect'][0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow("faces", frame)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, break from the loop and use last frame to extract embeddings
        if key == ord("q"):
            break

    # cleanup and closing frame
    cv2.destroyAllWindows()
    vs.stop()

    ####################################


if __name__ == "__main__":
    #choice = int(input("\n1.Add Person\n2.Extract_emb_dataset"))

    choice=1

    if (choice == 1):
        findPersonVideo()
    elif (choice == 2):
        extract_embeddings_datset()
    elif(choice == 3):
        img=cv2.imread("Brooke.jpeg")
        findPersonImage(img)

    # img1 = cv2.imread(args.img1)
    # img2 = cv2.imread(args.img2)
    # distance = compare2face(img1, img2)
    # threshold = 1.10    # set yourself to meet your requirement
    #
    # if distance==-1:
    #     print("Face not found in one of the image")
    # else:
    #     print("distance = "+str(distance))
    #     print("Result = " + ("same person" if distance <= threshold else "not same person"))
