import numpy as np
import glob
import math
from sklearn.model_selection import train_test_split

class dataUtils:
    
    def __init__( self, inputDir ):
        
        self.inputDir = inputDir
        self.files = glob.glob( inputDir )
        self.numberOfLabels = len( self.files ) #each file has a unique label
        self.sources = list()
        
        #outputs
        self.labelLevelMap = {} #map {string, int}
        self.labels = [] #will be converted to an ndarray during process
        self.images = [] #will be converted to an ndarray during process
        
        pass
        
    def process( self ):
        
        self.populateSourcesAndLevels()
        self.populateDataAndLabels()
        
        pass
    
    def populateSourcesAndLevels( self ):
        
        i = 0
        for path in self.files:

            label = path.split( "\\" )[-1].replace( ".npy", "" )
            self.sources.append( ( label, path ) )
            self.labelLevelMap[label] = i
            
            i = i + 1
            
        pass
    
    def populateDataAndLabels( self ):
        
        i = 0
        for ( label, path ) in self.sources:

            print( "Processing", label, path )

            labelData = np.load( path ).astype( np.float32 ) / 255
            print( "Observation #:", len( labelData ) )
            self.images.append( labelData )

            hotVector = np.array( [0] * self.numberOfLabels, np.float32 ) #must convert to 32 bit. by default it's 64
            hotVector[i] = 1.0
            i = i + 1

            hotVectors = [ hotVector ] * labelData.shape[0] #ndarray for single observations, list for all
            print( "label hot vectors #:", len( hotVectors ) )
            self.labels.append( hotVectors )

        self.images = np.vstack( self.images )
        self.labels = np.vstack( self.labels )
        
        pass
    
    def getData( self ):
        print( self.images.shape, self.labels.shape, self.labelLevelMap )
        return ( self.images, self.labels, self.labelLevelMap )
    
    def getRandomizedData( self, seed = 0 ):
        
        np.random.seed = seed
        seq = np.arange( self.labels.shape[0] )
        np.random.shuffle( seq )
        return ( self.images[seq], self.labels[seq], self.labelLevelMap )
        
        pass
    
    def reshapeDataForKeras( self, X, channel = 1 ):
        
        #TODO update it for multiple channels
        side = int( math.sqrt( X.shape[1] ) )
        return X.reshape( ( X.shape[0], side , side, channel ) )
        
        pass
    
    def sliceInverse( self, arr, selectedSeq ):
        
        mask = np.ones( len( data ), np.bool )
        mask[ selectedSeq ] = False
        return arr[mask]
        
        pass
    