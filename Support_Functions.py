import numpy as np
import matplotlib.image as mpimg
import scipy.io
import scipy.misc
import random

def normalize_row(x):
    ss = np.multiply(x, x)
    ss = np.sum(ss, axis=1)
    ss = np.sqrt(ss)
    data = (x.T * (1 / ss)).T
    return data


def predict (logits):
    zero_count = np.count_nonzero(logits, axis=1)
    invalid_list = np.nonzero(zero_count == 0)
    predictions = np.argmax(logits, axis=1)
    predictions[invalid_list] = -1
    return predictions


def compute_acc (predictions, labels):
    # predictions = predict(logits)
    correct = float(np.count_nonzero(predictions == labels))
    return correct / len(predictions)


def compute_mAP(predictions, labels):
    # predictions = predict(logits)
    nc = np.max(labels)+1
    nc0 = np.min(labels)
    avg_sum = 0
    for i in range (nc0,nc):
        c_list = np.nonzero(labels == i)
        predictions_c = predictions[c_list]
        labels_c = labels[c_list]
        if len(predictions_c) ==0:
            continue
        avg_sum = avg_sum + float(np.count_nonzero(predictions_c == labels_c))/len(labels_c)
    return avg_sum/(nc-nc0)



def convert_label(file_name, num_class = 20):
    classes = 0
    class_map = {}
    labels = []
    with open(file_name) as f:
        for line in f:
            labels.append(int (line))
    return np.array(labels, dtype = np.int32)


def shuffle_data(x_train, y_train):
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train, :, :, :]
    y_train = y_train[idx_train]
    return x_train, y_train

#----------------------------------------------------------------------------------------------------------------------

def Array_de_zeros(A, epsilon=1e-5):
    A[np.nonzero((A >= 0) * (A < epsilon))] = epsilon
    A[np.nonzero((A < 0) * (A > -epsilon))] = -epsilon
    return A





def pad_batch (x, pad_size = 4, dtype = np.uint8):
    def pad_rgb(a, pad_size=4):
        d = (a[:, :, 0], a[:, :, 1], a[:, :, 2])
        im = np.asarray([np.pad(x, (pad_size, pad_size), 'reflect') for x in d])
        return im.transpose(1, 2, 0)
    x_pad = np.zeros((x.shape[0], x.shape[1]+2*pad_size, x.shape[2]+2*pad_size, x.shape[3]), dtype = dtype)
    for i in range(x.shape[0]):
        Img = x[i, :, :, :]
        Img_pad = pad_rgb(Img, pad_size = pad_size)
        x_pad[i,:,:,:] = Img_pad
    return x_pad

# def normalize_img (img):
#     Nmin = np.random.randint(0,10)
#     Nmax = np.random.randint(245,255)
#     img = Nmin + img*(Nmax - Nmin)/(img.max()-img.min)


def augment_batch(ImgArray, size, offset=127.5, resize = False, aug = False):

    if aug:
        ImgArray_out = np.zeros((ImgArray.shape[0], size, size, ImgArray.shape[3]), dtype=ImgArray.dtype)
        ImgArray = pad_batch (ImgArray, dtype = ImgArray.dtype)
        for i in range (ImgArray.shape[0]):
            Img = ImgArray[i,:,:,:]
            if random.uniform(0, 1) > 0.5:
                Img = np.flip(Img, 1)
            a = random.randint(0, Img.shape[0] - size - 1)
            b = random.randint(0, Img.shape[1] - size - 1)
            Img = Img[a:a + size, b:b + size, :]
            ImgArray_out[i, :, :, :] = Img
    else:
        ImgArray_out =ImgArray

    ImgArray_out = ImgArray_out.astype(np.float32)
    ImgArray_out = (ImgArray_out - offset)/offset
    return ImgArray_out

def zca_whitening_matrix(X):
    X = X.T
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    whitex = np.dot(ZCAMatrix, X)
    return whitex.T