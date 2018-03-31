import cv2
import os
import glob
import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
#image_path='data-our/test/'

filename = dir_path + '/' + image_path + '/'
#filename='/home/thesis/Desktop/Signature-model/data-our/test'
path = os.path.join('test_own_data', filename, '*g')
files = glob.glob(path)
#print(files)

classes =["1","10","11","12","13","14","15",
"16","17","18","19","2","3","4","5","6","7","8"]#,"ch001","ch002","ch003","ch004","ch005","ch006","ch007",
#"ch008","ch009","ch010","dut001","dut002","dut003","dut004","dut006","dut009","dut012","dut014","dut015",
#"dut016"]

num_classes = len(classes)
image_size = 28
num_channels = 3

totalExperimented =0;
truePositive=0;
for i in files:

    filename = i

    print(i)
    p = 0
    className = ""
    chk2=0;
    i = i[::-1]
    for kk in i:
        if kk == '.':
            chk2=1;
        elif kk == '\\':
            break;
        elif kk == '/':
            break;
        elif chk2 == 1:
            className = className + kk;

   



    #className = className
    className = className[::-1]
    #print(className)

    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('F-Model-CharacterRecognition.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    # saver.save(sess, "G:\\study\\4-1\\300\\cv-tricks.com-master\\Tensorflow-tutorials\\tutorial-2-image-classifier\\ribo-object-classifier")
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, num_classes))

    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    #print('Result: ')

    j = 1;
    val =0;
    probableClass = 0;
    for ii in result:
        chk=0
        for jj in ii:
            chk=chk+1
            if jj>val:
                val =jj
                probableClass = chk
    totalExperimented = totalExperimented +1
    if val >= 0.90:

        print('Photo Name       : ', className)
        print('predicted person : ', classes[probableClass - 1])
        print('Probability value: ', val)
        #print('Match with       : ', probableClass)
        truePositive = truePositive + 1
    else:
        print('Photo Name       : ', className)
        print('predicted person : Unknown')
        print('Probability value: ', val)
        #print('Match with       : ', probableClass)

    print()
    print()
    # for i in m1:
    #     if i == m:
    #         break
    #     j = j + 1




    # print(m)


    # ans= m.argmax(axis=0)
    # print(m)

    # print(i)

# print(decisionMartrix)
# predans = 0
# totaltest = 0
# row = 0
# ii=0
# jj=0
# for i in decisionMartrix:
#     jj=0
#     for j in i:
#         if jj==ii:
#             predans = predans + decisionMartrix[ii][jj]
#             totaltest = totaltest + decisionMartrix[ii][jj]
#         else:
#             totaltest = totaltest + decisionMartrix[ii][jj]
#         jj = jj+1
#     ii= ii+1

# ans = float(truePositive * 100) / totalExperimented
# print ("Accuracy = ", ans)
print('Total Predicted    : ', truePositive)
print('Unknown predicted  : ', totalExperimented-truePositive)
