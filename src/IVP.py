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
        self.trueG = pd.read_csv("LieOpOpt_20.csv",header=None).to_numpy()
        return
    
    def DataExtraction(self, DataSize: int = 1000,T: float = 0.5,mode = 'Small'):
        self.TParam = T
        self.I0 = dataFunctions.Image1D(DataSize)
        if mode == 'Small':
            self.T , self.Ix = dataFunctions.Translation1DImage(self.I0,T,self.trueG)
        if mode == 'Large':
            self.T , self.Ix = dataFunctions.Translation1DImageLarge(self.I0,T)
        self.deltaI = self.Ix-self.I0
        return 
    
    def calcDeltaI(self,Ix,I0):
        deltaI = Ix-I0
        return deltaI

    def Optimize(self, G: np.array = None, mode: str = 'Small', epoch: int = 100, timeEpoch: int = 1,numberOfInitialX=10):
        if mode == 'Small':
            self.SmallOptimization(epoch, timeEpoch,numberOfInitialX)
            return
        if mode == 'Large':
            self.LargeOptimization(G,epoch,numberOfInitialX)
            return
        print("Unavailable Mode")
        return 

    def SmallOptimization(self, epoch: int = 100, timeEpoch: int = 1,numberOfInitialX: int = 10):
        xs = (np.random.uniform(-self.TParam,self.TParam,(self.I0.shape[0],numberOfInitialX)))
        Gs = np.random.normal(0,0.0000,size = (numberOfInitialX,self.I0.shape[1],self.I0.shape[1]))
        for _ in range(epoch):
            Gs = self.OptimizeG(xs,Gs)
            self.alpha /= 1.0001
            for _ in range(timeEpoch):
                xs = self.OptimizexSmall(xs,Gs)
        self.SaveOptimal(xs,Gs)
        self.GenerateImages()
        return 
    
    def LargeOptimization(self, G,epoch: int = 100,numberOfInitialX: int = 10):
        # G = self.trueG
        xs = (np.random.uniform(-self.TParam,self.TParam,(self.I0.shape[0],numberOfInitialX)))
        # xs = np.asarray([*range(-int(self.TParam), int(self.TParam+1))])
        # xs = np.reshape(xs,(xs.shape,1))
        for _ in range(10):
            self.gamma *= 0.9
            xs = self.OptimizexLarge(xs,G)
            print(xs[0])
        self.SaveOptimalLarge(xs,G)
        self.GenerateImagesLarge()
        return 
    
    def GenerateImages(self,samples: int = 5):
        cv2.imwrite('outputs/Ghat.jpg',self.Ghat*1000)
        cv2.imwrite('outputs/G.jpg',self.trueG*1000)

        for i in range(samples):
            cv2.imwrite('outputs/Small'+str(i)+'Ixhat.jpg',np.reshape(self.Ixhat[i],(1,20)))
            cv2.imwrite('outputs/Small'+str(i)+'Ix.jpg',np.reshape(self.Ix[i],(1,20)))
            cv2.imwrite('outputs/Small'+str(i)+'I0.jpg',np.reshape(self.I0[i],(1,20)))
        return
    
    def GenerateImagesLarge(self,samples: int = 5):

        for i in range(samples):
            cv2.imwrite('outputs/Large'+str(i)+'Ixhat.jpg',np.reshape(self.Ixhat[i],(1,20)))
            cv2.imwrite('outputs/Large'+str(i)+'Ix.jpg',np.reshape(self.Ix[i],(1,20)))
            cv2.imwrite('outputs/Large'+str(i)+'I0.jpg',np.reshape(self.I0[i],(1,20)))
        return

    def SaveOptimal(self,xs,Gs):
        xsReshape = np.reshape(xs,(xs.shape[0],xs.shape[1],1,1))
        IxReshape = np.reshape(self.Ix,(self.Ix.shape[0],1,self.Ix.shape[1],self.Ix.shape[2]))
        I0Reshape = np.reshape(self.I0,(self.I0.shape[0],1,self.I0.shape[1],self.I0.shape[2]))
        GsReshape = np.reshape(Gs,(1,Gs.shape[0],Gs.shape[1],Gs.shape[2]))

        predictions = (np.matmul(scipy.linalg.expm(xsReshape*GsReshape),I0Reshape))
        Errors = np.sum(np.abs(IxReshape - predictions),axis=(0,2,3))

        self.xhat = xs[:,np.argmin(Errors)]
        self.Ghat = Gs[np.argmin(Errors),:,:]
        self.Ixhat = predictions[:,np.argmin(Errors),:,:]
        print(self.xhat[0])
        print(self.Ixhat[0])

        return
    
    def SaveOptimalLarge(self,xs,Gs):
        xsReshape = np.reshape(xs,(xs.shape[0],xs.shape[1],1,1))
        IxReshape = np.reshape(self.Ix,(self.Ix.shape[0],1,self.Ix.shape[1],self.Ix.shape[2]))
        I0Reshape = np.reshape(self.I0,(self.I0.shape[0],1,self.I0.shape[1],self.I0.shape[2]))
        GsReshape = np.reshape(Gs,(1,1,Gs.shape[0],Gs.shape[1]))

        predictions = (np.matmul(scipy.linalg.expm(xsReshape*GsReshape),I0Reshape))
        Errors = np.sum(np.abs(IxReshape - predictions),axis=(0,2,3))

        self.xhat = xs[:,np.argmin(Errors)]
        self.Ixhat = predictions[:,np.argmin(Errors),:,:]

        print(self.xhat[0])
        print(self.Ixhat[0])

        return
    
    def exp(self,i,x):
        if i ==0:
            return np.identity(self.G.shape[0])
        return ((x*self.G)**i)/np.math.factorial(i)+self.exp(i-1,x)

    def OptimizeG(self,xs,Gs):
        xsReshape = np.reshape(xs,(xs.shape[0],xs.shape[1],1,1))
        deltaIReshape = np.reshape(self.deltaI,(self.deltaI.shape[0],1,self.deltaI.shape[1],self.deltaI.shape[2]))
        I0Reshape = np.reshape(self.I0,(self.I0.shape[0],1,self.I0.shape[1],self.I0.shape[2]))
        GsReshape = np.reshape(Gs,(1,Gs.shape[0],Gs.shape[1],Gs.shape[2]))

        I0Transpose =  (np.transpose(self.I0,axes=(0,2,1)))
        I0Transpose = np.reshape(I0Transpose,(I0Transpose.shape[0],1,I0Transpose.shape[1],I0Transpose.shape[2]))
        xI0Transpose = (np.multiply(xsReshape,I0Transpose))
        deltaIsubxGI0 =  deltaIReshape- (xsReshape*np.matmul(GsReshape,I0Reshape))
        deltaGs =  self.alpha * np.matmul(deltaIsubxGI0,xI0Transpose) - self.alpha * self.CInv * GsReshape

        return (Gs + np.mean(deltaGs,axis= 0))

    
    def OptimizexSmall(self,xs,Gs):
        xsReshape = np.reshape(xs,(xs.shape[0],xs.shape[1],1,1))
        deltaIReshape = np.reshape(self.deltaI,(self.deltaI.shape[0],1,self.deltaI.shape[1],self.deltaI.shape[2]))
        I0Reshape = np.reshape(self.I0,(self.I0.shape[0],1,self.I0.shape[1],self.I0.shape[2]))
        GsReshape = np.reshape(Gs,(1,Gs.shape[0],Gs.shape[1],Gs.shape[2]))

        GI0Transpose = np.transpose(np.matmul(GsReshape,I0Reshape),axes = (0,1,3,2))

        deltaIsubxGI0 = deltaIReshape - xsReshape * np.matmul(GsReshape,I0Reshape)
        deltaXs = self.gamma * np.matmul(GI0Transpose,deltaIsubxGI0) - self.gamma/(self.sigmaX**2) * xsReshape
        return xs + np.reshape(deltaXs,(deltaXs.shape[0],deltaXs.shape[1]))

    def OptimizexLarge(self,xs,G):
        xsReshape = np.reshape(xs,(xs.shape[0],xs.shape[1],1,1))
        IxReshape = np.reshape(self.Ix,(self.Ix.shape[0],1,self.Ix.shape[1],self.Ix.shape[2]))
        I0Reshape = np.reshape(self.I0,(self.I0.shape[0],1,self.I0.shape[1],self.I0.shape[2]))
        GsReshape = np.reshape(G,(1,1,G.shape[0],G.shape[1]))

        expxGGI0Transpose = np.transpose(np.matmul(np.matmul(scipy.linalg.expm(xsReshape*GsReshape),GsReshape),I0Reshape),axes = (0,1,3,2))
        IxsubexpxGI0 = IxReshape - np.matmul(scipy.linalg.expm(xsReshape*GsReshape),I0Reshape)

        deltaXs = self.gamma * np.matmul(expxGGI0Transpose,IxsubexpxGI0) - self.gamma/(self.sigmaX**2) * xsReshape
        return xs + np.reshape(deltaXs,(deltaXs.shape[0],deltaXs.shape[1]))


if __name__ == "__main__":
    Small = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.00001,gamma=0.000001,CInv=0.0001)
    Small.DataExtraction(DataSize=25000)
    Small.Optimize(epoch=1000,numberOfInitialX=2)

    a,b= (np.linalg.eig(Small.Ghat))

    plt.plot(a.real)
    plt.savefig('real.png')
    plt.close()
    plt.plot(a.imag)
    plt.savefig('imag.png')
    plt.close()
    plt.plot(Small.Ghat[10])
    plt.savefig('G10thRow.png')
    plt.close()

    Large = InvariantVisualPercentron(sigma=1,sigmaX=1,alpha=0.00001,gamma=0.00001,CInv=0.0001)
    Large.DataExtraction(DataSize=250,T = 1.0,mode = 'Small')
    Large.Optimize(G= Small.Ghat,epoch=100,numberOfInitialX=1,mode = 'Large')
    # x.ConstructImage()
