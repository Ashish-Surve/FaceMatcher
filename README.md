#  How to use this model to train and recognise people.

1. Clone this repository.
2. Install these in a virtualenv created in the repository.
    a. tensorflow 1.7
    b. numpy 1.16.1
    c. opencv-python
    d. imutils
2. In /dataset create a folder for each person you wish to recognise and add pictures of that person in the folder.
3. execute extract_embeddings.py
4. execute train_model.py
5. execute Face_recognition_app.py

    `Note : Clone the branch with name Ashish_Surve`

Your webcam should start in a few seconds and classify the faces by bounding boxes. 

If you face any issue please contact me and also take a screenshot of the error.


# Face-recognition
What things I have tried?

**FACNET**

=> I tried to use facenet library directly but could not find proper documentation or API for it.

=> There was no code available that provided me basic building block for code.

=> FaceNet by David Sandberg-- project looks mature, although at the time of writing does not provide a library-based installation nor clean API.

**FACEMATCH**

**PROS**

It can detect similar faces even if the images compared have 10+ years of difference in ages. e.g is given below.


***I have intentionally chosen pictures which are a little difficult to recognize but the result is astonishing***

**IMAGE 1**
![Pic1](/uploads/e553a9d81496573ead922ffbe67fc6bc/Pic1.png)

**IMAGE 2**
![pic2](/uploads/2a75b2352922af89578030680c31807a/pic2.png)

**OUTPUT**

you can clearly see that in one image I am wearing an accessory and yet the model is able to guess that both are same person.

![pic3](/uploads/ba9f3559fd86aa2cbce3c87b26370dc7/pic3.png)


2. This model provides good implementation of MTCNN because of which the detection is accurate.
=> This code only provided basics for face detection and face matching.

**CONS**
1. the speed seems to be a little low, though on an average I am getting 20 fps + while the video implementation is in beta stage.

2. It uses old libraries. i.e. TensorFlow 1.7 and numpy 1.16.1

**What problems did I face?**
There were a lot of library issue with the code I discovered and there was no requirement.txt available.

**Solution**
=> using virtualenv I created a new environment
=>tensor-flow has no Session error.
This occurs in 16.04 not in 18.04.

This error solved by downgrading to "tf 1.7.1"
This resulted in another error related to numpy.
I again downgraded numpy to numpy-1.16.1 


**Steps:**
1. follow above steps and get the libraries and virtualenv.
2. Detect faces in an image or video using MTCNN and align them.
3. extract embedding from it.
4. match the embedding with image(euclidean distance is used as a parameter, if the threshold value generated is greater than 1.10 then the person are different and if less then the person is same.).
5. display whether the persons are same or not.

**TO DOs**
*  ~~Implement a SVM based classifier to classify people.~~
*  ~~Implement video based face recognition.~~
*  ~~add GitLab link for code.~~
*  Implement a DNN based classifier to classify.
*  Compare SVM with DNN. Check Speeds.
*  ~~Basic documentation for the code.~~

# EDIT 1

1. SVM based classifier implemented.
2. Video based recognition implemented.
3. Local repository pushed to [GitLab](https://gitlab.com/shunyaos/ai-batch/tree/Ashish_Surve)
4. Note: The repository is 700+ MBs.  

Some Working Sample.

** Since I usually get confused between these two, I tried to recognize them using this model.
  
![twvsrt](/uploads/83bfd86c6b4730b4788fdf50bb4dedfa/twvsrt.png)
