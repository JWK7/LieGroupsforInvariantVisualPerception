import dataFunctions
import numpy as np
import scipy
import cv2
import pandas as pd
# import matplotlib
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt 
import scipy.ndimage


class InvariantVisualPercentron:
    def __init__(self,
                sigma: float,
                sigmaX: float,
                alpha: float,
                gamma: float,
                CInv: float):
        self.sigma = sigma
        self.sigmaX = sigmaX
        self.alpha = alpha
        self.gamma = gamma
        self.CInv = CInv
        return
    def DataExtraction(self):
        self.I0 = dataFunctions.Image1D()
        self.T = []
        self.Ix = []
        self.x = []
        self.savex = []
        self.deltaI = []
        for i in range(1,2,1):
            # T, Ix = dataFunctions.Translation1D(self.I0,i)
            T,Ix = dataFunctions.Translation1DSubPixel(self.I0,-i*0.05)
            # print(Ix)
            # print(Ix2)
            # self.Ix.append(Ix)
            self.deltaI.append(np.reshape(self.calcDeltaI(Ix,self.I0),(20,1)))
            # print(deltaI)
            self.Ix.append(np.reshape(Ix,(20,1)))
            self.savex.append(1)
            self.x.append(dataFunctions.Range(-self.I0.shape[1]*0.05/4,self.I0.shape[1]*0.05/4,0.05))#+np.random.normal(0,1,self.I0.shape[1])])

            # print([*range(-self.I0.shape[1]//2,self.I0.shape[1]//2)])

            # self.x.append([*range(-self.I0.shape[1]//2,self.I0.shape[1]//2)])#+np.random.normal(0,1,self.I0.shape[1])])
            self.T.append(T)
        # self.x = [[5]]
        self.I0 = np.reshape(self.I0,(20,1))
        self.G = 1*np.random.normal(0,1,(len(self.x[0]),20,20))
        self.saveG = 1*np.random.normal((20,20))
        return 
    
    def calcDeltaI(self,Ix,I0):
        deltaI = Ix-I0
        return deltaI

    def Optimize(self):
        errComp = float('inf')
        trueG = pd.read_csv("LieOpOpt_20.csv",header=None).to_numpy()
        for i in range(self.G.shape[0]):
            self.G[i] = trueG
        # print(5*self.G[0]*self.I0.flatten())
        # print((1*np.matmul(trueG,(self.I0))).astype(int))


        # print((1*np.matmul(np.transpose(self.I0),(trueG)).astype(int)))


        # self.G *= 100
        # print(self.G)
        # print(expxGI0)
        # print(self.Ix)
        # print(np.matmul(-5*self.G,self.I0))
        # print(self.I0)
        # print((self.deltaI[0] - np.matmul(-5*self.G,self.I0)).astype(int))
        # print(self.deltaI[0])
        # print((np.matmul(scipy.linalg.expm(-5 * self.G),self.I0)))
        # print(self.Ix[0]-(np.matmul(scipy.linalg.expm(-5 * self.G),self.I0)))
        
        # print(self.Ix[0])
        # print(np.matmul(scipy.linalg.expm(5 * self.G),self.I0))
        # print(np.sum(np.abs(self.Ix[0] - np.matmul(scipy.linalg.expm(-1 * self.G),self.I0))))
        # return
        # print(self.G.shape)
        # print(self.G)
        # self.x = 
        print(self.x)
        print('\n\n\n\n')
        for m in range(1000):
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
            # print(self.x)
            self.OptimizeG()
            self.Optimizex()
            self.gamma *= 0.9
            self.alpha *= 0.9
            # self.alpha /=1.0001
            # for i in range(100):
            #     self.Optimizex()
            #     print(self.x)
            # self.OptimizeG()
            # print(self.x[0][0])
            # for i in range(len(self.x)):
            #     for j in range(len(self.x[i])):
            #         if self.x[i][j] > 100000 or self.x[i][j] < -100000:
            #             self.x[i][j] = 0
            # print(self.G[19])
            # self.alpha *= 0.9
            # print(np.sum(np.abs(self.G[15]-self.G[19])))
        # print(self.x)

        # print(self.G)
        # print(np.sum(np.abs(np.subtract(trueG,self.G[0]))))
        # print(np.sum(np.abs(np.subtract(trueG,np.zeros((20,20))))))
        #     print(trueG-self.G)
        #     for i in range(len(self.x)):
        #         print(np.sum(np.abs(np.subtract(trueG,self.G[i]))))
        #     # print(self.G)
        #     print(self.x)
        #     # return
        #     # self.gamma *= 0.9
        print(self.G.shape)
        for i in range(len(self.x[0])):
            print("i")
            print(np.sum(np.abs(self.Ix[0] - np.matmul(scipy.linalg.expm(self.x[0][i] * self.G[i]),self.I0))))
            print(np.sum(np.abs(self.Ix[0] - np.matmul(scipy.linalg.expm(self.x[0][i] * trueG),self.I0))))
        print(np.sum(np.abs(self.Ix[0] - np.matmul(scipy.linalg.expm(0.05 * trueG),self.I0))))
        print(self.x)
        print(np.sum(np.abs(np.subtract(trueG,self.G[0]))))

        cv2.imwrite('G.jpg',self.G[0]*1000)
        # print(self.T)
        # print(self.x)
        # for i in range(len(self.x[0])):
        #     print(self.x[0][i])
        #     print(np.sum(np.abs(trueG-self.G[i])))
        # print(self.G[0])
        # print(self.x[0])
        return 
    
    def exp(self,i,x):
        if i ==0:
            return np.identity(self.G.shape[0])
        return ((x*self.G)**i)/np.math.factorial(i)+self.exp(i-1,x)

    def OptimizeG(self):
        print("hi")
        deltG = np.zeros(self.G.shape)
        for i in range(len(self.x)):
            print("fs")
            for j in range(len(self.x[i])):
                print("\n\n")
                print(self.x[i][j])
                # print((self.deltaI[i] -self.x[i][j] * np.matmul(self.G,self.I0)))
                # print((self.x[i][j]*self.I0))
                # print(j)
                deltG[j] += self.alpha * np.matmul(
                    (self.deltaI[i] -self.x[i][j] * np.matmul(self.G[j],self.I0)) , np.transpose(self.x[i][j]*self.I0)
                    ) - self.alpha * (self.CInv) * self.G[j]
            deltG[i] /= (len(self.x)*len(self.x[0]))
        self.G += deltG
        # print("\n\n\n\n\n\n\n\n")
        print(deltG)

        return
    def Optimizex(self):

        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                print(self.x[i][j])
                self.x[i][j] += (self.gamma * np.matmul( np.transpose(np.matmul(self.G[j],self.I0)), (self.deltaI[i] - np.matmul(self.x[i][j] * self.G[j],self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0]
                # self.x[i][j] += (self.gamma * np.matmul(
                # np.transpose(
                #     np.matmul(
                #         np.matmul(scipy.linalg.expm(self.x[i][j] * self.G[j]),self.G[j])
                #         ,self.I0)
                #         )
                #         , (self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i][j] * self.G[j]),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0]
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


        # for i in range(len(self.x)):
        #     for j in range(len(self.x[i])):
        #         self.x[i][j] += (self.gamma * np.matmul(
        #         np.transpose(
        #             np.matmul(
        #                 np.matmul(scipy.linalg.expm(self.x[i][j] * self.G[j]),self.G[j])
        #                 ,self.I0)
        #                 )
        #                 , (self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i][j] * self.G[j]),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0]
                # print((self.gamma * np.matmul(
                # np.transpose(
                #     np.matmul(
                #         np.matmul(scipy.linalg.expm(self.x[i][j] * self.G),self.G)
                #         ,self.I0)
                #         )
                #         , (self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i][j] * self.G),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0])
        return    

    def OptimizexLarge(self):
        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                self.x[i][j] += (self.gamma * np.matmul(
                np.transpose(
                    np.matmul(
                        np.matmul(scipy.linalg.expm(self.x[i][j] * self.G[j]),self.G[j])
                        ,self.I0)
                        )
                        , (self.Ix[i] - np.matmul(scipy.linalg.expm(self.x[i][j] * self.G[j]),self.I0)))-((self.gamma)/(self.sigmaX**2) * self.x[i][j]))[0][0]

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

        print(errors)

        for i in range(len(self.x)):
            # print(self.T[i])
            # df = pd.DataFrame(self.T[i])
            # df.to_csv("path/to/file.csv")
            # print(self.x[i][np.argmin(errors[i])])
            # print(np.argmin(errors[i]))
            OptXs[i] = self.x[i][np.argmin(errors[i])]
            GroundTruthImage = np.reshape(self.Ix[i],(1,20))
            cv2.imwrite('outputs/GroundTruth'+str(self.T[i])+'.jpg',GroundTruthImage)
            xGI0 = np.reshape(np.matmul(OptXs[i]*self.G,self.I0),(1,20))
            cv2.imwrite('outputs/xGI0'+str(self.T[i])+'.jpg',xGI0)

            expxGI0 = np.reshape(np.matmul(scipy.linalg.expm(OptXs[i] * self.G),self.I0),(1,20))
            cv2.imwrite('outputs/expxGI0'+str(self.T[i])+'.jpg',expxGI0)


        print(self.x[0].shape)
        print(self.T)
        X_Y_Spline = make_interp_spline(self.T, errors[0])
        
        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(np.array(self.T).min(), np.array(self.T).max(), 500)
        Y_ = X_Y_Spline(X_)
        
        # Plotting the Graph
        plt.plot(X_, Y_)
        plt.title("Plot Smooth Curve Using the scipy.interpolate.make_interp_spline() Class")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("x.png")
        return


if __name__ == "__main__":
    # x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000000001,gamma=0.02,C=0.001)
    # x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.000000001,gamma=0.000001,C=0.001)
    # x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.00000001,gamma=0.00001,CInv=0.0001)
    x = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.00001,gamma=0.00001,CInv=1/0.0001)

    x.DataExtraction()
    # print(x.x)
    x.Optimize()
    # x.ConstructImage()
