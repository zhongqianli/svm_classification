import cv2
import numpy as np

SZ=20
bin_n = 16 # Number of bins

## [hog]
def hog(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
## [hog]

def get_hog_cv2(image):

    return hog

class HOG:
    def __init__(self, size):
        w, h = size
        winSize = (int(w), int(h))
        blockSize = (int(w / 2), int(h / 2))
        blockStride = (int(w / 4), int(h / 4))
        cellSize = (int(w / 4), int(h / 4))
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64

        if cv2.__version__[0] == '2':
            self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                         histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        else:
            signedGradient = True
            self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    def calc_hog_feature(self, image):
        hog_feature = self.hog.compute(image)
        hog_feature = np.squeeze(hog_feature)
        return hog_feature

if __name__ == '__main__':
    image = cv2.imread("../../resources/images/eye.jpg", 0)

    rows, cols = image.shape[:2]

    print(rows, cols)

    hog = HOG((32, 32))

    bt = cv2.getTickCount()
    # hist = hog(image)
    hog_feature = hog.calc_hog_feature(image)
    et = cv2.getTickCount()

    t = (et - bt) * 1000.0 / cv2.getTickFrequency()
    print('t = {0} ms'.format(t))

    print(type(hog_feature), len(hog_feature))
    # print hog_feature

    cv2.imshow("image", image)
    cv2.waitKey()