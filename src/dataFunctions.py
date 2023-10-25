import cv2
import numpy as np
import glob
#Processes Images
import scipy.ndimage

def ProcessImage(ImageSize: tuple):
    for i in range(1,200):
        img = cv2.imread("data/OriginalImages/cat."+str(i)+".jpg")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.resize(img_gray,ImageSize)
        cv2.imwrite("data/ProcessedImages/"+str(i)+".jpg", processed_img)
    return

#Converts all images in Processed Image directory to numpy DF
def ImageToDataFrame() -> np.ndarray:
    ImageShape = cv2.imread(glob.glob('data/ProcessedImages/*.jpg')[0]).shape
    DataFrame = np.empty([len(glob.glob('data/ProcessedImages/*.jpg')),ImageShape[0],ImageShape[1]])
    i = 0
    for filename in glob.glob('data/ProcessedImages/*.jpg'):
        DataFrame[i]= cv2.imread(filename,cv2.COLOR_BGR2GRAY)
        i+=1
    return DataFrame

#1D Translation from I0
def Translation1D(I0: np.ndarray,T) -> np.ndarray:
    Ts = np.array(T)
    Ts -= (Ts>0)*((I0.shape)[1])

    Ix = np.empty(I0.shape)
    # THistory = np.empty(I0.shape[0])
    for i in range(I0.shape[0]):
        # T = np.random.randint(1,50)
        # THistory[i] = Ts

        for j in range(I0.shape[1]):
            Ix[i][j] = I0[i][j+Ts]

    return T,Ix

def Range(start,end,interval):
    i = start
    output = []
    while (i < end):
        output.append(i)
        i+=interval
    return output

def Image1D():
    Img = np.ones((1,20))
    Img[0][10] = 255.0
    Img[0][9] = 100.0
    Img[0][11] = 100.0
    return Img

def Translation1DSubPixel(I0: np.ndarray,T) -> np.ndarray:
    return T,scipy.ndimage.shift(I0, (0, T),mode ='wrap')
    

# def Rotation2D(I0: np.ndarray) -> np.ndarray:
#     G = np.array()







if __name__ == "__main__":
    # ProcessImage((20,20))
    # Translation1D(Data)
    Data = Image1D()
    cv2.imwrite('original.jpg',Data)
    i = -1.5
    Hist , Img = Translation1D(Data,i)
    cv2.imwrite('groundtruth'+str(i)+'.jpg',Img)
    # ProcessImage()
    # ImageToDataFrame()