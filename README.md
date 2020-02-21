# facematch
Face match in python using Facenet and their pretrained model


FaceNet by David Sandberg-- project looks mature, although at the time of writing does not provide a library-based installation nor clean API.

USAGE:
Detect Face
python face_detect_demo.py --img=images/faces.jpg

Find Embeddings
python face_embeddings_demo.py --img=images/daniel-radcliffe_5.jpg

compare 2 pics
python face_match_demo.py --img1=images/daniel-radcliffe_2.jpg --img2=images/daniel-radcliffe_4.jpg