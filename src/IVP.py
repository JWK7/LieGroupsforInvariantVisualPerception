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
        self.I0 = dataFunctions.Image1D()
        self.T = []
        self.Ix = []
        self.x = []
        self.savex = []
        self.deltaI = []
        for i in range(-10,10,1):
            T, Ix = dataFunctions.Translation1D(self.I0,i)
            # self.Ix.append(Ix)
            self.deltaI.append(np.reshape(self.calcDeltaI(Ix,self.I0),(20,1)))
            # print(deltaI)
            self.Ix.append(np.reshape(Ix,(20,1)))
            self.savex.append(1)
            self.x.append([*range(-self.I0.shape[1]//2,self.I0.shape[1]//2)])#+np.random.normal(0,1,self.I0.shape[1])])
            self.T.append(T)
        self.I0 = np.reshape(self.I0,(20,1))
        self.G = 1*np.random.normal(0,1,(20,20))
        self.saveG = 1*np.random.normal((20,20))
        return 
    
    def calcDeltaI(self,Ix,I0):
        deltaI = Ix-I0
        return deltaI

    def Optimize(self):
        errComp = float('inf')
        self.G = pandas.read_csv("LieOpOpt_20.csv",header=None).to_numpy()
        # print(expxGI0)
        # print(self.Ix)
        # print(np.matmul(-5*self.G,self.I0))
        # print(self.I0)
        # print((self.deltaI[0] - np.matmul(-5*self.G,self.I0)).astype(int))
        # print(self.deltaI[0])
        # print((np.matmul(scipy.linalg.expm(-5 * self.G),self.I0)))
        # print(self.Ix[0]-(np.matmul(scipy.linalg.expm(-5 * self.G),self.I0)))
        # print(np.sum(np.abs(self.Ix[0] - np.matmul(scipy.linalg.expm(-1 * self.G),self.I0))))
        # return
        # print(self.G.shape)
        
        for m in range(100):
            # errors = np.zeros((len(self.x),len(self.x[0])))
            # errors2 = np.zeros((len(self.x),len(self.x[0])))
            # for i in range(len(self.x)):
            #     for j in range(len(self.x[i])):
            #         print(self.x[i][j])
            #         errors[i][j] = np.sum(np.abs(self.deltaI[i] - np.matmul(self.x[i][j]*self.G,self.I0)))
            #         errors2[i][j] = np.sum(np.abs(self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i][j] * self.G),self.I0)))
            #         print(errors2[i][j])
            # err = (*range(len(self.x[0])))
            # for j in range(len(self.x)):
            #     for k in range(len(self.x[j])):
            #     err += np.sum(np.abs(self.deltaI[j] -self.x[j][k] * np.matmul(self.G,self.I0)))
            # print(i)
            # print(err)
            # if err < errComp:
            #     self.saveG = self.G
            #     self.savex = self.x
            self.Optimizex()
            # self.gamma *= 0.9
        # cv2.imwrite('G.jpg',self.G)
        # print(self.x)
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
        return
    def Optimizex(self):
        # filler = [0,0,0,0,0,,0,0]
        # for i in range(len(self.x)):
        #     for j in range(len(self.x[i])):
        #         self.x[i][j] += (self.gamma * np.matmul(
        #         np.transpose(
        #             np.matmul(
        #                 np.matmul(self.exp(10,self.x[i][j]),self.G)
        #                 ,self.I0)
        #                 )
        #                 , (self.Ix[i] - np.matmul(self.exp(10,self.x[i][j]),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0]
        #         print((self.gamma * np.matmul(
        #         np.transpose(
        #             np.matmul(
        #                 np.matmul(self.exp(10,self.x[i][j]),self.G)
        #                 ,self.I0)
        #                 )
        #                 , (self.Ix[i] - np.matmul(self.exp(10,self.x[i][j]),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0])
        
        # for i in range(len(self.x)):
        #     for j in range(len(self.x[i])):
        #         self.x[i][j] += (self.gamma)*( ((1/self.sigma**2)*np.matmul( np.transpose((self.deltaI[i] - np.matmul(self.x[i][j]*self.G, self.I0))),
        #         np.matmul(self.G , self.I0))) -(self.x[i][j]/self.sigmaX**2))[0][0]


        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                self.x[i][j] += (self.gamma * np.matmul(
                np.transpose(
                    np.matmul(
                        np.matmul(scipy.linalg.expm(self.x[i][j] * self.G),self.G)
                        ,self.I0)
                        )
                        , (self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i][j] * self.G),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0]
                # print((self.gamma * np.matmul(
                # np.transpose(
                #     np.matmul(
                #         np.matmul(scipy.linalg.expm(self.x[i][j] * self.G),self.G)
                #         ,self.I0)
                #         )
                #         , (self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i][j] * self.G),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0])
        
        # print("sum")
        # print(np.sum(np.abs(filler-filler2)))

        # print("hello")
        # print(filler2)
        # print(filler)
        # print(filler)
        # self.x += filler
        return    



    def ConstructImage(self):

        OriginalImage = np.reshape(self.I0,(1,20))
        cv2.imwrite('outputs/Original.jpg',OriginalImage)

        errors = np.zeros((len(self.x),len(self.x[0])))
        OptXs = np.zeros(len(self.x))

        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                # errors[i][j] = np.sum(np.abs(self.deltaI[i] - np.matmul(self.x[i][j]*self.G,self.I0)))
                # print(scipy.linalg.expm(x[i][j] * self.G))
                errors[i][j] = np.sum(np.abs(self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i][j] * self.G),self.I0)))
                # print(errors[i][j])

        for i in range(len(self.x)):
            print(self.T[i])
            # print(self.x[i][np.argmin(errors[i])])
            # print(np.argmin(errors[i]))
            OptXs[i] = self.x[i][np.argmin(errors[i])]
            GroundTruthImage = np.reshape(self.Ix[i],(1,20))
            cv2.imwrite('outputs/GroundTruth'+str(self.T[i])+'.jpg',GroundTruthImage)
            xGI0 = np.reshape(np.matmul(OptXs[i]*self.G,self.I0),(1,20))
            cv2.imwrite('outputs/xGI0'+str(self.T[i])+'.jpg',xGI0)

            expxGI0 = np.reshape(np.matmul(scipy.linalg.expm(OptXs[i] * self.G),self.I0),(1,20))
            cv2.imwrite('outputs/expxGI0'+str(self.T[i])+'.jpg',expxGI0)



        #     img = np.reshape(np.matmul(scipy.linalg.expm(min(self.x[i]) * self.saveG),self.I0),(1,20))
        #     cv2.imwrite('Catprediction'+str(i)+'.jpg', img)
        #     img2 = np.reshape(self.Ix[i],(1,20))
        #     cv2.imwrite('Catgroundtruth'+str(i)+'.jpg',img2)
        #     img2 = np.reshape(self.I0,(1,20))
        #     cv2.imwrite('Catoriginal.jpg',img2)
        #     img3 = np.reshape(np.matmul(min(self.x[i])*self.saveG,self.I0),(1,20))
        #     cv2.imwrite('Catprediction2'+str(i)+'.jpg',img3)
        return


if __name__ == "__main__":
    # x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000000001,gamma=0.02,C=0.001)
    # x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000000001,gamma=0.000001,C=0.001)
    x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000001,gamma=0.00001,C=0.0001)
    x.DataExtraction()
    # print(x.x)
    x.Optimize()
    x.ConstructImage()
