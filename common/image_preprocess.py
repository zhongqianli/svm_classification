import cv2
import glob
import os

imagepathlist = glob.glob('./resources/*.jpg')
for imagepath in imagepathlist:
    imagepath = imagepath.replace('\\', '/')
    print imagepath
    img = cv2.imread(imagepath, 1)
    img = cv2.resize(img, (500, 500))

    # dir = imagepath.split('.jpg')[0]
    # if not os.path.exists(dir):
    #     os.makedirs(dir)

    # n = 0
    # for i in range(5):
    #     for j in range(5):
    #         roi = img[100*i:100*(i+1), 100*j:100*(j+1)]
    #         n += 1
    #         filename = dir + '/{0}.jpg'.format(n)
    #         # print filename
    #         cv2.imwrite(filename, roi)

    # cv2.imshow('img', img)
    # cv2.waitKey()
    cv2.imwrite(imagepath, img)