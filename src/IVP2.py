import dataFunctions
import numpy as np
import scipy
import cv2
class InvariantVisualPercentron:
    def __init__(self,
                sigma: float,
                sigmaX: float,
                alpha: float,
                gamma: float,
                C: float):
        self.sigma = sigma
        self.sigmaX = sigmaX
        self.alpha = alpha
        self.gamma = gamma
        self.C = C
        return
    def DataExtraction(self):
        i= 10
        # self.I0 = dataFunctions.Image1D()
        self.I0 = dataFunctions.Image1D()
        self.T,self.Ix = dataFunctions.Translation1D(self.I0,i)
        self.x = 1
        self.savex = 1
        self.deltaI = self.calcDeltaI(self.Ix,self.I0)
        self.I0 = np.reshape(self.I0,(1,20))
        self.Ix = np.reshape(self.Ix,(1,20))
        self.G = 1*np.ones([20,1])
        self.saveG = 1*np.ones([20,1])
        return 
    
    def calcDeltaI(self,Ix,I0):
        deltaI = Ix-I0
        # self.deltaI = self.x * np.matmul(self.G,self.I0) + np.random.normal(0,self.sigma**2,size = self.I0.shape)
        # print(np.random.normal(0.0,self.sigma^2))#, size = (400)))# (self.I0.shape[0])))
        # print(self.sigma**2)
        return deltaI

    def Optimize(self):
        errComp = float('inf')
        for i in range(100):
            err = 0
            # print(np.sum(np.abs(self.deltaI -self.x[ * np.matmul(self.G,self.I0))))
            # print(np.matmul(self.G,self.I0))
        
            err = np.sum(np.abs(self.deltaI -self.x * np.matmul(self.G,self.I0)))

            print(err)
            # err = np.sum(np.abs(self.Ix - np.matmul(scipy.linalg.expm(self.savex * self.saveG),self.I0)))
            # print(i)
            # print(err)
            if err < errComp:
                self.saveG = self.G
                self.savex = self.x
            #     errComp = err

            # # print(self.G)
            self.OptimizeG()
            # self.x = 500
            for i in range(200):
            #     # print(scipy.linalg.expm(self.x * self.G)[0])
            #     # print(self.exp(11)[0])
                self.Optimizex()
            #     # print(scipy.linalg.expm(self.x * self.G))
            #     # print(np.linalg.expm(self.x*self.G))
            #     # print(self.exp(6))

        # print(self.x*self.G)
        # cv2.imwrite('outputs/G'+str(self.i)+'.jpg',self.x*self.G*10000)
        # print(self.x)
        return 
    
    def exp(self,i):
        if i ==0:
            return np.identity(self.G.shape[0])
        return ((self.x*self.G)**i)/np.math.factorial(i)+self.exp(i-1)

    def OptimizeG(self):
        self.G = self.alpha * np.matmul(
            (self.deltaI -self.x * np.matmul(self.G,self.I0)) , np.transpose(self.x*self.I0)
            ) - self.alpha * self.C * self.G
        return
    def Optimizex(self):
        filler = (self.gamma)*( ((1/self.sigma**2)*np.matmul( np.transpose((self.deltaI - np.matmul(self.x*self.G, self.I0))),
        np.matmul(self.G , self.I0))) -(self.x/self.sigmaX**2))[0][0]



        # filler = (self.gamma * np.matmul(
        # np.transpose(
        #     np.matmul(
        #         np.matmul(scipy.linalg.expm(self.x * self.G),self.G)
        #         ,self.I0)
        #         )
        #         , (self.Ix - np.matmul(scipy.linalg.expm(self.x * self.G),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x))[0][0]
        
        # filler = (self.gamma * np.matmul(
        # np.transpose(
        #     np.matmul(
        #         np.matmul(self.exp(10),self.G)
        #         ,self.I0)
        #         )
        #         , (self.Ix - np.matmul(self.exp(10),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x))[0][0]
        
        # print("sum")
        # print(np.sum(np.abs(filler-filler2)))

        # print("hello")
        # print(filler2)
        # print(filler)
        # print(filler)
        self.x += filler
        return    

    def ConstructImage(self,i):
        img3 = np.reshape(np.matmul(self.savex*self.saveG,self.I0),(20,20))
        img = np.reshape(np.matmul(scipy.linalg.expm(self.savex * self.saveG),self.I0),(20,20))
        cv2.imwrite('outputs/Catprediction'+str(i)+'.jpg', img)
        img2 = np.reshape(self.Ix,(20,20))
        cv2.imwrite('outputs/Catgroundtruth'+str(i)+'.jpg',img2)
        img2 = np.reshape(self.I0,(20,20))
        cv2.imwrite('outputs/Catoriginal.jpg',img2)
        cv2.imwrite('outputs/Catprediction2'+str(i)+'.jpg',img3)
        return


if __name__ == "__main__":
    # x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000000001,gamma=0.02,C=0.001)
    x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000000001,gamma=0.00000001,C=0.001)
    x.DataExtraction()
    x.Optimize()
    # x.ConstructImage(i)