# import the necessary packages
#from skimage import feature
import numpy as np
import cv2

'''
    The LBP imitated from scikit-image.
    Compare to scikit-image's LBP, the imitated LBP has subtile difference.
    
    result:

scikit-image's LBP, maybe based on C    
    t = 18.5292077033 ms
[ 0.01032236  0.03460219  0.03182442  0.13655693  0.29533608  0.24399863
  0.08336763  0.05126886  0.06241427  0.05030864]

imitated LBP, based on python
t = 2072.75384571 ms
[ 0.00826475  0.02280521  0.0372428   0.10874486  0.28964335  0.24543896
  0.09588477  0.06323731  0.06004801  0.06868999]
  
'''

from skimage import feature
import numpy as np
import cv2

def calc_LBP_hist(image, P, R):
    lbp = feature.local_binary_pattern(image, P, R, method="uniform")
    hist = histogram(lbp, P, R)
    return hist

def histogram(lbp, P, R):
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))

    # normalize the histogram
    hist = np.float32(hist)
    eps = 1e-7
    hist /= (hist.sum() + eps)
    return hist

def original_lbp(image):
    """origianl local binary pattern"""
    rows = image.shape[0]
    cols = image.shape[1]

    lbp_image = np.zeros((rows - 2, cols - 2), np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            code = 0
            center_pix = image[i, j]
            if image[i - 1, j - 1] > center_pix:
                code = code | (1 << 7)
            if image[i - 1, j] > center_pix:
                code = code | (1 << 6)
            if image[i - 1, j + 1] > center_pix:
                code = code | (1 << 5)
            if image[i, j + 1] > center_pix:
                code = code | (1 << 4)
            if image[i + 1, j + 1] > center_pix:
                code = code | (1 << 3)
            if image[i + 1, j] > center_pix:
                code = code | (1 << 2)
            if image[i + 1, j - 1] > center_pix:
                code = code | (1 << 1)
            if image[i, j - 1] > center_pix:
                code = code | (1 << 0)
            lbp_image[i - 1, j - 1] = code
    return lbp_image


# scikit-image LBP
def local_binary_pattern(image, P, R, method='U'):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

    LBP is an invariant descriptor that can be used for texture classification.

    Parameters
    ----------
    image : (N, M) array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'default', 'ror', 'uniform', 'var'}
        Method to determine the pattern.

        * 'default': original local binary pattern which is gray scale but not
            rotation invariant.
        * 'ror': extension of default implementation which is gray scale and
            rotation invariant.
        * 'uniform': improved rotation invariance with uniform patterns and
            finer quantization of the angular space which is gray scale and
            rotation invariant.
        * 'nri_uniform': non rotation-invariant uniform patterns variant
            which is only gray scale invariant [2]_.
        * 'var': rotation invariant variance measures of the contrast of local
            image texture which is rotation but not gray scale invariant.

    Returns
    -------
    output : (N, M) array
        LBP image.

    References
    ----------
    .. [1] Multiresolution Gray-Scale and Rotation Invariant Texture
           Classification with Local Binary Patterns.
           Timo Ojala, Matti Pietikainen, Topi Maenpaa.
           http://www.rafbis.it/biplab15/images/stories/docenti/Danielriccio/Articoliriferimento/LBP.pdf, 2002.
    .. [2] Face recognition with local binary patterns.
           Timo Ahonen, Abdenour Hadid, Matti Pietikainen,
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851,
           2004.
    """
    # methods = {
    #     'default': ord('D'),
    #     'ror': ord('R'),
    #     'uniform': ord('U'),
    #     'nri_uniform': ord('N'),
    #     'var': ord('V')
    # }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _local_binary_pattern(image, P, R, method)
    return output


def _bit_rotate_right(value, length):
    """Cyclic bit shift to the right.

    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer

    """
    return (value >> 1) | ((value & 1) << (length - 1))


def get_texture(image, rows, cols, r, c, i):
    texture = 0
    if r == 0 and c == 0:   # 0
        if i == 0:
            pass
        elif i == 1:
            pass
        elif i == 2:
            pass
        elif i == 3:
            texture = image[r, c + 1]
        elif i == 4:
            texture = image[r + 1, c + 1]
        elif i == 5:
            texture = image[r + 1, c]
        elif i == 6:
            pass
        elif i == 7:
            pass
    elif r == 0 and c > 0 and c < cols - 1: # 1
        if i == 0:
            pass
        elif i == 1:
            pass
        elif i == 2:
            pass
        elif i == 3:
            texture = image[r, c + 1]
        elif i == 4:
            texture = image[r + 1, c + 1]
        elif i == 5:
            texture = image[r + 1, c]
        elif i == 6:
            texture = image[r + 1, c - 1]
        elif i == 7:
            texture = image[r, c - 1]
    elif r == 0 and c == cols - 1:  # 2
        if i == 0:
            pass
        elif i == 1:
            pass
        elif i == 2:
            pass
        elif i == 3:
            pass
        elif i == 4:
            pass
        elif i == 5:
            texture = image[r + 1, c]
        elif i == 6:
            texture = image[r + 1, c - 1]
        elif i == 7:
            texture = image[r, c - 1]
    elif r > 0 and r < rows -1 and c == cols - 1:   # 3
        if i == 0:
            texture = image[r - 1, c - 1]
        elif i == 1:
            texture = image[r - 1, c]
        elif i == 2:
            pass
        elif i == 3:
            pass
        elif i == 4:
            pass
        elif i == 5:
            texture = image[r + 1, c]
        elif i == 6:
            texture = image[r + 1, c - 1]
        elif i == 7:
            texture = image[r, c - 1]
    elif r == rows - 1 and c == cols - 1:   # 4
        if i == 0:
            texture = image[r - 1, c - 1]
        elif i == 1:
            texture = image[r - 1, c]
        elif i == 2:
            pass
        elif i == 3:
            pass
        elif i == 4:
            pass
        elif i == 5:
            pass
        elif i == 6:
            pass
        elif i == 7:
            texture = image[r, c - 1]
    elif r == rows - 1 and c > 0 and c < cols - 1:  # 5
        if i == 0:
            texture = image[r - 1, c - 1]
        elif i == 1:
            texture = image[r - 1, c]
        elif i == 2:
            texture = image[r - 1, c + 1]
        elif i == 3:
            texture = image[r, c + 1]
        elif i == 4:
            pass
        elif i == 5:
            pass
        elif i == 6:
            pass
        elif i == 7:
            texture = image[r, c - 1]
    elif r == rows - 1 and c == 0:  # 6
        if i == 0:
            pass
        elif i == 1:
            texture = image[r - 1, c]
        elif i == 2:
            texture = image[r - 1, c + 1]
        elif i == 3:
            texture = image[r, c + 1]
        elif i == 4:
            pass
        elif i == 5:
            pass
        elif i == 6:
            pass
        elif i == 7:
            pass
    elif r > 0 and r < rows - 1 and c == 0: # 7
        if i == 0:
            pass
        elif i == 1:
            texture = image[r - 1, c]
        elif i == 2:
            texture = image[r - 1, c + 1]
        elif i == 3:
            texture = image[r, c + 1]
        elif i == 4:
            texture = image[r + 1, c + 1]
        elif i == 5:
            texture = image[r + 1, c]
        elif i == 6:
            pass
        elif i == 7:
            pass
    else:   # 8
        if i == 0:
            texture = image[r - 1, c - 1]
        elif i == 1:
            texture = image[r - 1, c]
        elif i == 2:
            texture = image[r - 1, c + 1]
        elif i == 3:
            texture = image[r, c + 1]
        elif i == 4:
            texture = image[r + 1, c + 1]
        elif i == 5:
            texture = image[r + 1, c]
        elif i == 6:
            texture = image[r + 1, c - 1]
        elif i == 7:
            texture = image[r, c - 1]
    return texture


def _local_binary_pattern(image, P, R, method='U'):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

        LBP is an invariant descriptor that can be used for texture classification.

        Parameters
        ----------
        image : (N, M) double array
            Graylevel image.
        P : int
            Number of circularly symmetric neighbour set points (quantization of
            the angular space).
        R : float
            Radius of circle (spatial resolution of the operator).
        method : {'D', 'R', 'U', 'N', 'V'}
            Method to determine the pattern.

            * 'D': 'default'
            * 'R': 'ror'
            * 'U': 'uniform'
            * 'N': 'nri_uniform'
            * 'V': 'var'

        Returns
        -------
        output : (N, M) array
            LBP image.
    """
    # texture weights
    weights = 2 ** np.arange(P, dtype=np.int32)

    # pre-allocate arrays for computation
    texture = np.zeros(P, dtype=np.double)
    signed_texture = np.zeros(P, dtype=np.int8)
    rotation_chain = np.zeros(P, dtype=np.int32)

    output_shape = (image.shape[0], image.shape[1])
    output = np.zeros(output_shape, dtype=np.double)
    rows = image.shape[0]
    cols = image.shape[1]

    lbp = 0
    r = 0
    c = 0
    changes = 0
    i = 0
    rot_index = 0
    n_ones = 0
    first_zero = 0
    first_one = 0
    sum_ = 0
    var_ = 0
    texture_i = 0

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            lbp = 0
            for i in range(P):
                texture[i] = get_texture(image, rows, cols, r, c, i)
                # signed / thresholded texture
                if texture[i] - image[r, c] >= 0:
                    signed_texture[i] = 1
                else:
                    signed_texture[i] = 0

            # if method == 'uniform':
            if method == 'U':
                # determine number of 0 - 1 changes
                changes = 0
                for i in range(P - 1):
                    changes += (signed_texture[i]
                                - signed_texture[i + 1]) != 0

                if changes <= 2:
                    for i in range(P):
                        lbp += signed_texture[i]
                else:
                    lbp = P + 1
            else:
                # method == 'default'
                for i in range(P):
                    lbp += signed_texture[i] * weights[i]

                # method == 'ror'
                if method == 'R':
                    # shift LBP P times to the right and get minimum value
                    rotation_chain[0] = lbp
                    for i in range(1, P):
                        rotation_chain[i] = \
                            _bit_rotate_right(rotation_chain[i - 1], P)
                    lbp = rotation_chain[0]
                    for i in range(1, P):
                        lbp = min(lbp, rotation_chain[i])

            output[r, c] = lbp
    return np.asarray(output)


if __name__ == '__main__':
    image = cv2.imread("../../resources/images/face.jpg", 0)

    cv2.imshow("image", image)

    rows, cols = image.shape[:2]

    # org_lbp_image = original_lbp(image)
    # cv2.imshow("org_lbp_image", org_lbp_image)

    P = 8
    R = 1

    hist = calc_LBP_hist(image, P, R)

    print(hist)
    print(type(hist))

    # bt = cv2.getTickCount()
    # lbp1 = feature.local_binary_pattern(image, P, R, method="uniform")
    # hist = histogram(lbp1, P, R)
    # et = cv2.getTickCount()
    #
    # t = (et - bt) * 1000.0 / cv2.getTickFrequency()
    # print 't = {0} ms'.format(t)

    # print hist

    # cv2.imshow('skimage LBP U', lbp1 / 12)

    # bt = cv2.getTickCount()
    # lbp2 = local_binary_pattern(image, P, R, method='U')
    # hist = histogram(lbp2, P, R)
    # et = cv2.getTickCount()

    # t = (et - bt) * 1000.0 / cv2.getTickFrequency()
    # print 't = {0} ms'.format(t)
    #
    #
    # cv2.imshow('imitated LBP U', lbp2 / 12)

    cv2.waitKey()
