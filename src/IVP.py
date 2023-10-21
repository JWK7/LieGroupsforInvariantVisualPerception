import dataFunctions
import numpy as np
import scipy
import cv2
import pandas

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
        # self.I0 = dataFunctions.Image1D()
        self.I0 = dataFunctions.Image1D()
        self.T = []
        self.Ix = []
        self.x = []
        self.savex = []
        self.deltaI = []
        for i in range(5,6,1):
            T, Ix = dataFunctions.Translation1D(self.I0,i)
            Ix = np.reshape(Ix,(20,1))
            self.Ix.append(Ix)
            self.deltaI.append(self.calcDeltaI(Ix,self.I0))
            self.savex.append(1)
            self.x.append([*range(-self.I0.shape[1]//2,self.I0.shape[1]//2)+np.random.normal(0,1,self.I0.shape[1])])
            self.T.append(T)

        # self.I0 = (self.I0 - np.min(self.I0)) / (np.max(self.I0)- np.min(self.I0))
        # self.Ix = (self.Ix - np.min(self.Ix)) / (np.max(self.Ix)- np.min(self.Ix))
        # self.calcDeltaI(Ix,I0)
        # self.T = self.T[0]
        self.I0 = np.reshape(self.I0,(20,1))
        self.G = 1*np.random.normal(0,1,(20,20))
        self.saveG = 1*np.random.normal((20,20))
        return 
    
    def calcDeltaI(self,Ix,I0):
        deltaI = Ix-I0
        # self.deltaI = self.x * np.matmul(self.G,self.I0) + np.random.normal(0,self.sigma**2,size = self.I0.shape)
        # print(np.random.normal(0.0,self.sigma^2))#, size = (400)))# (self.I0.shape[0])))
        # print(self.sigma**2)
        return deltaI

    def Optimize(self):
        errComp = float('inf')
        self.G = pandas.read_csv("LieOpOpt_20.csv",header=None).to_numpy()

        # img = np.reshape(np.matmul(scipy.linalg.expm(5 * self.G),self.I0),(1,20))
        # cv2.imwrite('UsingLieOpOPt.jpg', img)

        # return
        print(self.G.shape)
        for i in range(1000):
            err = 0
            for j in range(len(self.x)):
                err += np.sum(np.abs(self.deltaI[j] -self.x[j] * np.matmul(self.G,self.I0)))


            # err = np.sum(np.abs(self.Ix - np.matmul(scipy.linalg.expm(self.savex * self.saveG),self.I0)))
            print(i)
            print(err)
            if err < errComp:
                self.saveG = self.G
                self.savex = self.x
            #     errComp = err

            # # print(self.G)
            # self.OptimizeG()
            # self.x = 500

            # for i in range(10):
            #     # print(scipy.linalg.expm(self.x * self.G)[0])
            # #     # print(self.exp(11)[0])
            #     self.Optimizex()
            #     print(self.x)
                # print(self.x)
            # print('hi')
            #     # print(scipy.linalg.expm(self.x * self.G))
            #     # print(np.linalg.expm(self.x*self.G))
            #     # print(self.exp(6))
            self.Optimizex()
            # self.gamma *= 0.8
        print(self.gamma)

        # print(self.G)
        cv2.imwrite('G.jpg',self.G)
        print(self.x)
        print(self.savex)
        print(self.T)
        return 
    
    def exp(self,i,x):
        if i ==0:
            return np.identity(self.G.shape[0])
        return ((x*self.G)**i)/np.math.factorial(i)+self.exp(i-1,x)

    def OptimizeG(self):
        deltG = np.zeros((20,20))
        for i in range(len(self.x)):
            deltG += self.alpha * np.matmul(
                (self.deltaI[i] -self.x[i] * np.matmul(self.G,self.I0)) , (self.x[i]*self.I0)
                ) - self.alpha * (1/self.C) * self.G
        deltG /= len(self.x)
        self.G += deltG
        # print(G)
        return
    def Optimizex(self):
        # filler = [0,0,0,0,0,,0,0]
        # for i in range(len(self.x)):
        #     filler = (self.gamma * np.matmul(
        #     np.transpose(
        #         np.matmul(
        #             np.matmul(self.exp(10,self.x[i]),self.G)
        #             ,self.I0)
        #             )
        #             , (self.Ix - np.matmul(self.exp(10,self.x[i]),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i]))[0][0]
        
            # self.x[i] += (self.gamma)*( ((1/self.sigma**2)*np.matmul( np.transpose((self.deltaI[i] - np.matmul(self.x[i]*self.G, self.I0))),
            # np.matmul(self.G , self.I0))) -(self.x[i]/self.sigmaX**2))[0][0]


        for i in range(len(self.x)):
            self.x[i] += (self.gamma * np.matmul(
            np.transpose(
                np.matmul(
                    np.matmul(scipy.linalg.expm(self.x[i] * self.G),self.G)
                    ,self.I0)
                    )
                    , (self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i] * self.G),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i]))[0][0]
        
        # print("sum")
        # print(np.sum(np.abs(filler-filler2)))

        # print("hello")
        # print(filler2)
        # print(filler)
        # print(filler)
        # self.x += filler
        return    

    def ConstructImage(self):
        # img3 = np.reshape(np.matmul(self.savex*self.saveG,self.I0),(20,20))
        # img = np.reshape(np.matmul(scipy.linalg.expm(self.savex * self.saveG),self.I0),(20,20))
        # cv2.imwrite('outputs/Catprediction'+str(i)+'.jpg', img)
        # img2 = np.reshape(self.Ix,(20,20))
        # cv2.imwrite('outputs/Catgroundtruth'+str(i)+'.jpg',img2)
        # img2 = np.reshape(self.I0,(20,20))
        # cv2.imwrite('outputs/Catoriginal.jpg',img2)
        # cv2.imwrite('outputs/Catprediction2'+str(i)+'.jpg',img3)
        for i in range(len(self.x)):
            img = np.reshape(np.matmul(scipy.linalg.expm(self.savex[i] * self.saveG),self.I0),(1,20))
            cv2.imwrite('Catprediction'+str(i)+'.jpg', img)
            img2 = np.reshape(self.Ix[i],(1,20))
            cv2.imwrite('Catgroundtruth'+str(i)+'.jpg',img2)
            img2 = np.reshape(self.I0,(1,20))
            cv2.imwrite('Catoriginal.jpg',img2)
            img3 = np.reshape(np.matmul(self.savex[i]*self.saveG,self.I0),(1,20))
            cv2.imwrite('Catprediction2'+str(i)+'.jpg',img3)
        return


if __name__ == "__main__":
    # x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000000001,gamma=0.02,C=0.001)
    # x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000000001,gamma=0.000001,C=0.001)
    x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.0001,gamma=0.000001,C=0.0001)
    x.DataExtraction()
    print(x.x)
    # x.Optimize()
    # x.ConstructImage()
