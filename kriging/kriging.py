import numpy as np
from math import log10, log, pi, sqrt
from scipy import linalg
from utility import cho_invert, cond_num
from time import time
#import matplotlib.pyplot as plt
#from enthought.mayavi.mlab import *

class KrigingModel:
    
    MAX_CORR = 10.0
    MIN_CORR = 0.01
    MAX_COND = 13
    GOLDEN_SECT = 1.618
    ITER_LIMIT = 100
    
    def __init__(self, XIn, yIn, KSIn = None,TMIn = None, SMIn= None, dIn = None):
        self.X=np.mat(XIn)
        self.y=yIn[:]
        # the number of input dimensions
        # print self.X
        self.p=np.size(self.X,1)
        # print 'number of dimensions: ',self.p
        # the number of observations
        self.n=len(self.y)
        self.setScale()
        self.scaleData()
        self.setIPD()
        self.solutions = []
        optModel = False
        if KSIn is None:
            self.ks=0.0
        else:
            self.ks=KSIn
            
        if TMIn is None:
            self.createTrendModelForm()
            # flag that we need to determine the optimal trend mode
            optModel= True
        else:
            # otherwise, we leave the flag as false and just use the given
            # form of the trend model
            self.trendModel=np.mat(TMIn)
        if SMIn is None:
        # the number of dimensions used in the model - start with all
            self.s=self.p
            # select all dimensions to be in the spatial model
            self.spatialModel=np.ones(self.s)
        else:
            self.spatialModel=np.mat(SMIn)
            self.s=self.spatialModel.sum()
        # number of estimated model parameters    
        
        if dIn is None:
            self.resetCorrelationModel()
            self.updateTrendModel()
            # self.q = self.s + self.numTrendParams + 1    
            if optModel:
                self.createOptModel()
            else:
                self.estimateRandomMLE()
        else:
            self.d=np.mat(dIn)
            self.createRMatrix()
            self.updateTrendModel()
            # self.q = self.s + self.numTrendParams + 1    
            self.getCorrGradient()
            test = np.sqrt(np.vdot(self.grad, self.grad))
            if test <0.01:
                self.MLEOptimal = True
            else:
                self.MLEOptimal = False
        
        
    def setScale(self):
        self.max=np.array(self.X.max(axis=0))
        self.min=np.array(self.X.min(axis=0))
        
    def scaleData(self):
        temp=(self.X[0]-self.min)/(self.max-self.min)
        for row in self.X[1:]:
            temp=np.append(temp,(row-self.min)/(self.max-self.min), axis=0)
        self.scaledX=np.array(temp)
        
    def setIPD(self):
        self.ipd=np.zeros((self.p, self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.p):
                    self.ipd[k,i,j]=np.abs(self.scaledX[i,k]-self.scaledX[j,k])
                    
    def createRMatrix(self):
        self.R = np.zeros((self.n, self.n))
        self.Rp = np.zeros((self.n, self.n))
        self.Rinv = np.zeros((self.n, self.n))
        # set the flag off for the MSE calculations
        self.flagMSE = 0
        # calculate the R and Rp matrices
        # the Rinv matrix is actually Rp^-1
        for i in range(self.n):
            for j in range(i+1):
                t=0.0
                l =0
                for k in range(self.p):
                    if self.spatialModel[k]==1:
                        t += self.ipd[k,i,j]*self.ipd[k,i,j]/(self.d[l]*self.d[l])
                        l += 1
                self.R[i,j]=np.exp(-1.0*t)
                if i!=j:
                    self.Rp[i,j]=(1.0-self.ks)*self.R[i,j]
                    self.Rinv[i,j]=self.Rp[i,j]
                    # now reflect
                    self.R[j,i]=self.R[i,j]
                    self.Rp[j,i]=self.Rp[i,j]
                    self.Rinv[j,i]=self.Rinv[i,j]
                else:
                    self.Rp[i,j]=self.R[i,j]
                    self.Rinv[i,j]=self.R[i,j]
        # return the cholesky factors in lower triangular 
        try:
            (self.Rinv, flag) =linalg.cho_factor(self.Rinv, False, True)
            # the determinant of a positive symmetric matrix is the square 
            # of the product of the diagonal elements of the Cholesky decomposition
            # this method is faster than linalg.det()
            self.detR=1.0
            for i in range(self.n):
                temp = self.Rinv[i,i]
                if temp>0.0:
                    self.detR=self.detR*temp
                else:
                    # something bad has happened
                    self.detR=self.detR
                    
            self.detR=self.detR*self.detR
            
            if self.detR<1e-200:
                # this is not good either
                self.detR=1e-200

            # self.detR=linalg.det(self.R)
            
            # Using the Cholesky factorization to invert the matrix is slightly faster than
            # using the .I property from numpy, though it is not as accurate
            # self.Rinv = cho_invert(self.Rinv, flag, True)
            # self.Rinv= np.matrix(self.R).I
            #print time()-start
            #start = time()
            # the best method in terms of speed and accuracy is to use linalg.inv
            self.Rinv = linalg.inv(self.Rp)
            # the last piece of information needed here is the condition number
            # it appears to be faster to use the built-in routine that calculates the inverse in the process
            # than it is to use the interpreted version that takes the inverse as an input
            #
            self.condR = cond_num(self.R, self.Rinv)
            # the 1 and the np.inf norm are the same for a symmetric matrix
            #start = time()
            #self.condR = log10(np.linalg.cond(self.R, 1))
            #print time()-start
            # print 'condR: ', self.condR
        except np.linalg.linalg.LinAlgError:
            self.detR = 1e-200
            self.condR = 100          
     
    def resetCorrelationModel(self):
        self.d = np.zeros((self.s))
        for i in range(self.s):
            self.d[i]=self.p/float(self.n)
        self.createRMatrix()
        
    """
    This method is responsible for creating an initial trend model form
    based entirely on the number of input dimensions to the problem and 
    the number of observations that are currently available.
    """
    def createTrendModelForm(self):
        # allocate space to specify a full second-order model
        # with cross terms
        self.trendModel=np.zeros((self.p+1,self.p+1))
        # the initial form of the trend model is a function of the number
        # of input dimensions and the number of observations
        
        # create a full second-order model
        if self.n>(((self.p+1)*(self.p+2)/2)+2*self.p) :
            for i in range(self.p+1):
                for j in range(self.p+1):
                    if j>=i :
                        self.trendModel[i,j]=1
        # if not enough observations then
        # create a first-order model
        elif self.n>3*self.p+1:
            for i in range(self.p+1):
                self.trendModel[0,i]=1
        # the default is to create a constant model
        else:
            self.trendModel[0,0]=1
    """
    This method for making sure the value of the number of trend
    parameters is up to date and the create the F matrix for the
    current trend model form.
    """    
    def updateTrendModel(self):
        self.numTrendParams = int(self.trendModel.sum())
        self.q = self.s + self.numTrendParams + 1
        self.createFMatrix()
        
    """
    This method sets the values for the F matrix given the current
    trend model form and the observations. It will then set the
    Q matrix (the 'Hat' matrix for the trend model.
    """
    def createFMatrix(self):
        self.F=np.zeros((self.n, self.numTrendParams))
        l=0
        for i in range(self.p+1):
            for j in range(self.p+1):
                if self.trendModel[i,j]==1:
                    for k in range(self.n):
                        if i==0:
                            row=1.0
                        else:
                            row=self.scaledX[k,i-1]
                        if j==0:
                            col = 1.0
                        else:
                            col=self.scaledX[k,j-1]
                        self.F[k,l]=row*col
                    l+=1
        self.createQMatrix()
        
    """ 
    The method handles the rest of the calculations for the trend 
    model given the model form and the observations. It calculates
    the trend model coefficients, the cross validation error terms
    for each of the observations, the cross validation mean square 
    error, and the estimated process variance for the model.
    """
    def createQMatrix(self):
        # this matrix gets used multiple times so keep it around
        c = np.dot(self.F.T, self.Rinv)
        self.Q=np.dot(c, self.F)
        self.Qinv=linalg.inv(self.Q)
        # what are the error terms
        err = np.dot(c,self.y)
        # calculate the trend coefficients
        self.beta = np.dot(self.Qinv,err)
        # the vector W holds the distance between the trend model and 
        # the observation
        self.W=self.y-np.dot(self.F,self.beta)
        self.z = np.dot(self.Rinv,self.W)
        # determine the cross validation errors 
        # this calculation works because of the convenient form of
        # the kriging equations
        sse=0.0
        self.cvErr=np.zeros(self.n)
        for i in range(self.n):
            self.cvErr[i]=self.z[i]/self.Rinv[i,i]
            sse+=self.cvErr[i]*self.cvErr[i]
        self.cvMSE=sse/self.n
        self.processVariance=np.dot(self.W.T,self.z)/self.n
        # need to test for the degenerative case where the process variance is near zero
        if self.processVariance<1e-100:
            self.processVariance=1e-100
        
            
    def getdVdd(self, dV):
        l=0
        # need to range through p instead of s since ipd is for all p 
        # dimensions
        for i in range(self.p):
            if self.spatialModel[i]==1:
                # the correlation range, d, is indexed on l not i
                scale=2.0*self.processVariance/(self.d[l]*self.d[l]*self.d[l])
                for j in range(self.n):
                    for k in range(self.n):
                        # ipd is indexed on i not l
                        dV[l,j,k]=scale*self.Rp[j,k]*self.ipd[i,j,k]*self.ipd[i,j,k]
                l+=1
            
    def getScore(self, H):
        # dV is a 3D tensor, s is the number of active spatial correlation
        # dimensions and n is the number of observations
        dV=np.zeros((self.s, self.n, self.n))
        self.getdVdd(dV)
        # inverse of covariance matrix
        Vinv = self.Rinv/self.processVariance
        # fill in the Hessian matrix with zeros
        # print 'before score cond R: ', self.condR
        # print 'd: ', self.d
        for i in range(self.s):
            temp1=np.dot(Vinv,dV[i])
            H[i,i]=np.trace(np.dot(temp1,temp1))/(-2.0)
            
    """
    This method calculates the analytical Hessian of the log-likelihood
    function
    """
    def getHessian(self):
        # allocate space for the information matrix and fill it with zeros
        # the information matrix is a square matrix consisting of all
        # of the trend model elements, the process variance, and the
        # correlation range parameters.
        self.info = np.zeros((self.q, self.q))
        Vinv = self.Rinv/self.processVariance
        
        # lbb is the information content of the trend model parameters
        # lbb = -1.0*F^T.Vinv.F
        lbb = -1.0*np.dot(self.F.T, np.dot(Vinv, self.F)) 
        # assign lbb into the top left of info
        for i in range(self.numTrendParams):
            for j in range(self.numTrendParams):
                self.info[i,j] = lbb[i,j]
        
        # the next needed value is the derivative of the covariance 
        # matrix V. There are two parts to this, the process variance
        # and the correlation range parameters.
        # dV for the process variance is the correlation matrix
        dVp=np.array(self.Rp)
        dVinvp=-1.0*np.dot(Vinv, np.dot(dVp, Vinv))
        lbs = np.dot(self.F.T,np.dot(dVinvp,self.W))

        # copy this row and column into info
        for i in range(self.numTrendParams):
            self.info[self.numTrendParams,i]=lbs[i]
            self.info[i,self.numTrendParams]=lbs[i]
        
        # calculate dVinv for each of the correlation parameters
        # start by allocating space     
        dVinv = np.zeros((self.s,self.n,self.n))
        # grab the rest of the dV matrix with respect to each correlation
        # parameter
        # dV is a 3D tensor, s is the number of active spatial correlation
        # dimensions and n is the number of observations
        dV=np.zeros((self.s, self.n, self.n))
        self.getdVdd(dV)
        # for each correlation parameter dVinv(i) = -1.0*Vinv.dV(i).Vinv
        # the vector of info content between trend parameters and each
        # correlation parameter is given as lbs=F^T.dVinv(i).W
        for i in range(self.s):
            dVinv[i]=-1.0*np.dot(Vinv, np.dot(dV[i],Vinv))
            lbs = np.dot(self.F.T, np.dot(dVinv[i],self.W))
            for j in range(self.numTrendParams):
                # this will copy these values in the upper right and
                # lower left parts of the information matrix (symmetry)
                self.info[self.numTrendParams+1+i, j]=lbs[j]
                self.info[j, self.numTrendParams+1+i]=lbs[j]
            
        # This section calculates the second derivative of Vinv (d2Vinv)
        # it is split into two parts, the process variance (postfix p)
        # and the s different correlation parameters.
        # calculate the Hessian of the process variance to itself and the
        # correlation parameters
        # the second derivative with respect to the process variance is zero
        d2Vp = np.zeros((self.n,self.n))
        d2Vp = np.dot(dVp,np.dot(Vinv,dVp))-d2Vp
        d2Vp = d2Vp + np.dot(dVp, np.dot(Vinv, dVp))
        d2Vinvp = np.dot(Vinv,np.dot(d2Vp,Vinv))
        ll = np.trace(np.dot(dVinvp,dVp)+np.dot(Vinv,d2Vp))
        ll += np.dot(self.W.T,np.dot(d2Vinvp,self.W))    
        self.info[self.numTrendParams,self.numTrendParams]=-0.5*ll
        # this section calculates the Hessian of the process variance with
        # respect to the correlation parameters
        for i in range(self.s):
            d2V = np.array(dV[i]/self.processVariance)
            temp1 = np.subtract(np.dot(dVp,np.dot(Vinv,dV[i])), d2V)  
            temp2 = np.add(temp1, np.dot(dV[i],np.dot(Vinv,dVp)))    
            d2Vinv = np.dot(Vinv,np.dot(temp2, Vinv))
            ll = np.trace(np.add(np.dot(Vinv,d2V),np.dot(dVinv[i], dVp)))
            ll = ll + np.dot(self.W.T,np.dot(d2Vinv, self.W))
            ll = -0.5*ll
            # copy result into row and column of process variance/correl params
            self.info[self.numTrendParams, self.numTrendParams+1+i]=ll
            self.info[self.numTrendParams+1+i,self.numTrendParams]=ll
            
        lss = np.zeros((self.s, self.s))
        self.getCorrHessian(lss)
        # print 'lss=', lss
        for i in range(self.s):
            for j in range(self.s):
                self.info[self.numTrendParams+1+i, self.numTrendParams+1+j]=lss[i,j]

        # print self.numTrendParams, self.s, self.q
        #print self.info
            
    """
    This routine will calculate the portion of the Hessian matrix for the
    correlation parameters.
    
    """
    def getCorrHessian(self,h):
        # calculate Vinv
        Vinv = self.Rinv/self.processVariance
        # dV is a 3D tensor, s is the number of active spatial correlation
        # dimensions and n is the number of observations
        dV=np.zeros((self.s, self.n, self.n))
        self.getdVdd(dV)
        # for each correlation parameter dVinv(i) = -1.0*Vinv.dV(i).Vinv
        # the vector of info content between trend parameters and each
        # correlation parameter is given as lbs=F^T.dVinv(i).W
        dVinv=np.zeros((self.s, self.n, self.n))
        for i in range(self.s):
            dVinv[i]=-1.0*np.dot(Vinv, np.dot(dV[i],Vinv))
        # h = np.zeros((self.s, self.s))
        for i in range(self.s):
            for j in range(self.s):
                if i==j:
                    scale = 2.0/(self.d[i]*self.d[i]*self.d[i])   
                    d2V = np.multiply(dV[i], (scale*np.multiply(self.ipd[i],self.ipd[i])-3.0/self.d[i])) 
                elif i>j:
                    scale = 2.0/(self.d[j]*self.d[j]*self.d[j])
                    d2V = np.multiply(dV[i], (scale*np.multiply(self.ipd[j],self.ipd[j]))) 
                else:
                    scale = 2.0/(self.d[i]*self.d[i]*self.d[i])
                    d2V = np.multiply(dV[j], (scale*np.multiply(self.ipd[i], self.ipd[i])))
                temp1 = np.subtract(np.dot(dV[j],np.dot(Vinv, dV[i])), d2V)
                temp1 = np.add(temp1, np.dot(dV[i], np.dot(Vinv, dV[j])))
                d2Vinv = np.dot(Vinv, np.dot(temp1, Vinv))
                h[i,j] = np.trace(np.add(np.dot(Vinv, d2V), np.dot(dVinv[i], dV[j])))
                h[i,j] = h[i,j] + np.dot(self.W, np.dot(d2Vinv, self.W))
                h[i,j] = -0.5*h[i,j]
                
        
    """
    This method calculates the gradient of the correlation parameters
    
    - not done yet
    """
    def getCorrGradient(self):
        # need to calculate dV[i].Vinv
        self.grad=np.zeros(self.s)
        dV=np.zeros((self.s, self.n, self.n))
        self.getdVdd(dV)
        Vinv = self.Rinv/self.processVariance
        for i in range(self.s):
            dVinv = -1.0*np.dot(Vinv, np.dot(dV[i],Vinv))
            self.grad[i]=np.trace(np.dot(Vinv, dV[i]))
            self.grad[i]+= np.dot(self.W,np.dot(dVinv, self.W))
            self.grad[i]=-0.5*self.grad[i]
            
    def getProfileLogLikelihood(self):
        return -0.5*(log(self.detR)+self.n+self.n*log(2.0*pi*self.processVariance))
        
    def getRndCorr(self):
        return np.power(10*np.ones(self.s),np.random.random(self.s,)*log10(self.MAX_CORR*100.0)-2)    
    
    def getAICc(self):
        ll = self.getProfileLogLikelihood()
        return -2.0*ll+2.0*self.q+(2.0*self.q*(self.q+1))/(self.n-self.q-1)
    
    
    def estimateLocalMLE(self):
        # q=self.numTrendParams
        start = time()
        test = 100.0
        term = 0
        oldCorr = np.array(self.d)
        #print self.d
        #print oldCorr
        #self.getdVdd()
        oldLL = self.getProfileLogLikelihood()
        #print 'start: ', self.d, oldLL
        done = False
        iters=0
        stuckH=0
        stuckS=0
        # Levenburg-Marquardt parameter
        lm = 0.001
        # initialize Hessian of correlation parameters with zeros
        H=np.zeros((self.s,self.s))

        while not done:
            # print 'oldLL : ', oldLL
            iters +=1
            # get the Score matrix
            self.getScore(H)
            # print self.H
            # subtract the lm from the diagonal elements of H
            for i in range(self.s):
                H[i,i]-=lm
            #    
            # print self.H
            Hinv = linalg.inv(H)
            condH = cond_num(H, Hinv)
            # the Hessian is nearly singular, no sense continuing
            # with these values since we are most likely at a 
            # saddlepoint and we will be stuck here for quite 
            # some time.
            # Just make the Hessian non-singular and jump out
            if condH>8.0:
                H=H+np.eye(self.s)
                Hinv=linalg.inv(H)
                condH = cond_num(H,Hinv)
                stuckH +=1
            else:
                stuckH=0
            # get the gradient
            self.getCorrGradient()
            step = -1.0*np.dot(Hinv, self.grad)
            # print 'step is: ',step
            # print 'gradient is: ', self.grad
            # test each value if it is within valid ranges
            scale = 1.0
            # Check to see if the current step moves us outside of the
            # current constraints. If so, then scale the step to put the 
            # next step on the constraint.
            
            # Need to put in a test to see if we are already on a constraint.
            # If we are, then the step in the constrained dimension will be
            # set to zero.
            constraint = 0;
            for i in range(self.s):
                if abs(self.MAX_CORR-oldCorr[i])<0.00000001:
                    # already on the max constraint
                    step[i]=0.0
                    constraint = 1
                    #scale = 0.0
                if abs(self.MIN_CORR-oldCorr[i])<0.00000001:
                    # already on the min constraint
                    step[i]=0.0
                    constraint = 2
                    #scale = 0.0
                tempCorr = oldCorr[i]+step[i]
                if tempCorr>self.MAX_CORR and abs(step[i])>0:
                    testScale = abs((self.MAX_CORR-oldCorr[i])/step[i])
                    if testScale < scale:
                        scale = testScale
                        constraint = 1
                if tempCorr<self.MIN_CORR  and abs(step[i])>0:
                    testScale = abs((oldCorr[i]-self.MIN_CORR)/step[i])
                    if testScale<scale:
                        scale = testScale
                        constraint = 2
            
            step = scale* step
            
            # check the current step size
            testStep = np.linalg.norm(step)
            if testStep < 0.00000001:
                # we're not going anywhere
                stuckS += 1
            else:
                stuckS = 0
                
            # check to see if the new values result in an ill-conditioned
            # correlation matrix
            newCorr = np.array(oldCorr+step)
            # print 'before d: ', self.d
            self.d = np.array(newCorr)
            # print 'after d: ', self.d
            #print self.d
            
            self.createRMatrix()
            # print 'cond R: ',self.condR
            
            while self.condR>self.MAX_COND:
                step = step/self.GOLDEN_SECT
                newCorr = np.array(oldCorr+step)                
                self.d = np.array(newCorr)             
                self.createRMatrix()
                constraint = 3
            self.createQMatrix()
            newLL = self.getProfileLogLikelihood()
            #print 'loc, step, gradient :', lm, self.d, step, self.grad, oldLL, newLL

            # print 'newLL: ', newLL
            # termination test
            test = np.linalg.norm(self.grad)
            # print 'test :', test, np.sqrt(np.vdot(self.grad,self.grad))
            if test>0.01:
                term=0
            else:
                term += 1
                if term == 4:
                    
                    done = True
            
            if newLL<=oldLL:
                # new values are worse, change l(a)m(bda)
                lm = lm *10
                self.d=np.array(oldCorr)        
                self.createRMatrix()
                self.createQMatrix()
            else:
                # new location is better
                lm = lm*0.1
                oldLL=newLL
                oldCorr = np.array(newCorr)
            if iters>self.ITER_LIMIT:
                # didn't converge
                done = True
                constraint = 4
            if stuckS>3:
                # got stuck
                done = True
                constraint = 5
        # print iters, constraint, self.d, oldLL, stuckS, stuckH, time()-start
            
            
        
        
    def estimateRandomMLE(self): 
        self.solutions = []
        bestCorr = np.array(self.d)
        
        bestLL = self.getProfileLogLikelihood()
        
        done = False
        iters = 0
        replicates = 0
        while not done:
            # get a random location to start
            self.d = self.getRndCorr()
            # create the correlation matrix
            self.createRMatrix()
            
            if self.condR>self.MAX_COND:
                # starting location is bad
                # go back to best solution
                self.d = np.array(bestCorr)
                self.createRMatrix()
                self.createQMatrix()
            else:
                # things look good
                iters += 1
                self.createQMatrix()
                # now let's estimate local MLE
                # print iters
                self.estimateLocalMLE()
                newLL = self.getProfileLogLikelihood()
                
                if abs(newLL-bestLL)<0.0001:
                    # got the same value as the best
                    # check to see if the correlation parameters are the same
                    diff = bestCorr - self.d
                    dsum = np.sqrt(np.vdot(diff, diff))
                    if dsum<0.001:
                        replicates+=1
                    else:
                        replicates=0
                elif newLL >bestLL:
                    replicates = 0
                
                if replicates==3 or iters>self.ITER_LIMIT:
                    self.d=np.array(bestCorr)
                    self.createRMatrix()
                    self.createQMatrix()
                    done = True
                    
                if newLL > bestLL:
                    bestCorr= np.array(self.d)
                    bestLL = newLL
                else:
                    pass

        # test to see if this is a real MLE or just on a
        # constraint boundary - is the first derivative zero?
        self.getCorrGradient()    
        test = np.sqrt(np.vdot(self.grad,self.grad))
        # print 'best answer : ', self.d, bestLL, self.getAICc(), test
        if test <0.01:
            self.MLEOptimal = True
        else:
            self.MLEOptimal = False
    
    """
    This method determines the best form of the trend model for the given
    dimensions of the correlation model included
    """                     
    def createOptModel(self):
        # number of model parameters
        self.q = self.s + self.numTrendParams + 1
        # set initial values for the correlation model
        self.resetCorrelationModel()
        # set up the initial trend model
        self.createTrendModelForm()
        # Update the trend model information
        self.updateTrendModel()
        # determine the best correlation parameters for a model that
        # includes the current trend model components
        self.estimateRandomMLE()
        # the corrected AIC for this model
        oldAIC = self.getAICc()
        newAIC = oldAIC
        # the number of model parameters
        if self.numTrendParams==1:
            test = True
        else:
            test = False
        # print self.trendModel
        # print self.beta, self.d
        while test is False:
            # print self.trendModel
            # print self.getAICc()
            # go calculate the full Hessian for the model
            self.getHessian()
            test_stat = linalg.inv(self.info)
            # determine the parameter with the smallest test statistic
            # start with the 1st element (not the 0th) since the constant
            # coefficient should never be removed
            bestValue = abs(self.beta[1]/sqrt(abs(test_stat[1,1])))
            # print "1", bestValue
            loc = 1
            for i in range(2,self.numTrendParams):
                currentValue = abs(self.beta[i]/sqrt(abs(test_stat[i,i])))
                # print i, currentValue
                if currentValue<bestValue:
                    loc = i
                    bestValue=currentValue
            # now remove the trend coefficient with the smallest test statistic
            # print loc, bestValue
            num = 0
            for i in range(self.p+1):
                for j in range(self.p+1):
                    num += self.trendModel[i,j]
                    if num ==(loc+1) and self.trendModel[i,j]==1:
                        self.trendModel[i,j] = 0
                        row = i
                        col = j
            # print row, col, self.trendModel
            # update the new trend model (F, Qinv, beta, Z, and W)
            self.updateTrendModel()
            # finally, estimate the correlation parameters for the 
            # new model form. One option would be to start with the
            # expensive random start MLE method, but, since we already
            # know something about the data, it is more efficient to
            # just the the local search method
            self.estimateLocalMLE()
            # grab the new AIC
            # print self.beta, self.d
            newAIC = self.getAICc()
            # if the new AIC is less than the old AIC then continue
            # also need to test if there is only the contant term left
            # print 'newAIC', newAIC
            if newAIC<oldAIC and self.numTrendParams>1:
                test = False
                oldAIC = newAIC
            else:
                test = True
        # check to see if we went one too far, return the last trend parameter to
        # the model
        if newAIC>oldAIC:
            # print 'replacing TM', row, col
            self.trendModel[row,col]=1
            self.updateTrendModel()  
            # self.q = self.s + self.numTrendParams + 1      
            self.estimateLocalMLE()
        # print 'trend model:', self.trendModel
        # print 'beta:', self.beta
        # print 'corr range:', self.d
        # print 'corrected AIC:', self.getAICc()
        
    """
    This method returns an estimate to the response surface at any location within
    the domain of the metamodel created previously.
    """
    def estimate(self,XIn):
        # how many dimensions are there to the input location
        dim = np.size(XIn)
        if self.p != dim:
            # the number of dimensions does not match the created model
            y=0.0
        else:
            # scale the input data
            x_scale = (XIn - self.min[0])/(self.max[0] - self.min[0])
            # calculate the correlation of the current point to the 
            # observation, start by calculating the scaled distance from
            # all of the current observations
            # this calculation is made slightly more complex because not all 
            # of the input dimensions may be included in the spatial correlation
            # model
            r = np.zeros((self.n))
            for i in range(self.n):
                k=0
                for j in range(self.p):
                    if self.spatialModel[j]==1:
                        dis = abs(x_scale[j]-self.scaledX[i,j])/self.d[k]
                        r[i] += dis*dis
                        k += 1
                r[i] = (1.0 - self.ks)*np.exp(-1.0*r[i])
            # calculate the f vector for the current point
            k=0
            f = np.zeros((self.numTrendParams))
            for i in range(self.p+1):
                for j in range(self.p+1):
                    if self.trendModel[i,j]==1:
                        if i==0:
                            row = 1.0
                        else:
                            row = x_scale[i-1]
                        if j==0:
                            col = 1.0
                        else:
                            col = x_scale[j-1]
                        f[k] = row*col
                        k += 1
            # the output is the product of all vectors and results in a scalar
            y = 0.0
            y += np.dot(f, self.beta) + np.dot(r,self.z)
            
        return y
    """
    This is the standard deviation of the estimate, at the given location.
    There are two sources of uncertainty, that which is inherit in the Gaussian
    Process model that is being used, and that which is inherit in using 
    estimates of the model parameters.
    
    This version is only calculating the model portion.
    """    
    def stdev(self,XIn):
        # how many dimensions are there to the input location
        dim = np.size(XIn)
        sd = 0.0
        if self.p != dim:
            # the number of dimensions does not match the created model
            sd=0.0
        else:
            # scale the input data
            x_scale = (XIn - self.min[0])/(self.max[0] - self.min[0])
            # calculate the correlation of the current point to the 
            # observation, start by calculating the scaled distance from
            # all of the current observations
            # this calculation is made slightly more complex because not all 
            # of the input dimensions may be included in the spatial correlation
            # model
            r = np.zeros((self.n))
            for i in range(self.n):
                k=0
                for j in range(self.p):
                    if self.spatialModel[j]==1:
                        dis = abs(x_scale[j]-self.scaledX[i,j])/self.d[k]
                        r[i] += dis*dis
                        k += 1
                r[i] = (1.0 - self.ks)*np.exp(-1.0*r[i])
            sd = self.processVariance*(1-np.dot(r,np.dot(self.Rinv,r)))
            sd = sqrt(abs(sd))
            
        return sd
               
        

    
