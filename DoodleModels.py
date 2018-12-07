import glob
import logging, sys, math
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
from keras import metrics
from sklearn.model_selection import train_test_split
class DoodleModels:
    def __init__( self, X, Y, test_size = 0.2, random_state = 39 ):
        self.X = X
        self.Y = Y
        self.test_size = test_size
        self.random_state = random_state
        self.trX, self.tsX, self.trY, self.tsY = train_test_split( 
                                                    self.X, 
                                                    self.Y, 
                                                    test_size = self.test_size, 
                                                    random_state = self.random_state 
                                                 )
        
        print( "train X:", self.trX.shape )
        print( "test X:", self.tsX.shape )
        print( "train Y:", self.trY.shape )
        print( "test Y:", self.tsY.shape )
        pass
    
    def runModel( self, model, epochs = 5, batch_size = 64 ):
        history = model.fit( self.trX, self.trY, epochs = epochs, batch_size = batch_size )
        tsLoss, tsAcc = model.evaluate( self.tsX, self.tsY )
        return ( history, tsLoss, tsAcc )
    
    def evaludateModel( self, model ):
        return model.evaluate( self.tsX, self.tsY )
    
    def getModel(self, modelNo ):
        
        if modelNo == 1:
            #model 1
            model = models.Sequential()
            model.add( layers.Conv2D( 16, (3,3), activation = activations.relu, input_shape = ( 28, 28, 1 ) ) )
            model.add( layers.MaxPooling2D( (2,2) ) )
            model.add( layers.Conv2D( 32, (3,3), activation = activations.relu ))
            model.add( layers.MaxPooling2D( (2,2) ) )
            model.add( layers.Conv2D( 32, (3,3), activation = activations.relu ))
            model.add( layers.Flatten() )
            model.add( layers.Dense( 32, activation = activations.relu ) )
            model.add( layers.Dense( 10, activation = activations.softmax ) )
            model.summary()
            model.compile( 
                optimizer = optimizers.rmsprop( lr = .001 ), 
                loss = losses.categorical_crossentropy,
                metrics = [ metrics.categorical_accuracy ] 
            )
        
            return model
    
        if modelNo == 3:
            return self.getConvPoolCNNCModel()
        if modelNo == 4:
            return self.getAllCNNC()
        if modelNo == 5:
            return self.NINCNN()
    
    def getConvPoolCNNCModel( self ):
        
        model_input = layers.Input( shape = ( 28, 28, 1 ) )
        x = layers.Conv2D(96, kernel_size=(3, 3), activation=activations.relu, padding = 'same')(model_input)
        x = layers.Conv2D(96, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.Conv2D(96, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
        
        x = layers.Conv2D(192, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.Conv2D(192, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.Conv2D(192, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
        
        x = layers.Conv2D(192, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.Conv2D(192, (1, 1), activation=activations.relu)(x)
        x = layers.Conv2D(10, (1, 1))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Activation(activation='softmax')(x)

        model = models.Model(model_input, x, name='conv_pool_cnn')

        model.summary()
        model.compile( 
            optimizer = optimizers.rmsprop( lr = .001 ), 
            loss = losses.categorical_crossentropy,
            metrics = [ metrics.categorical_accuracy ] 
        )

        return model
    
    def getAllCNNC( self ):
        pass
    
    def NINCNN( self ):
        pass
    
    
    def ensembleModelsByAverageOutput( self, modelList ):
        
        model_input = layers.Input(shape=modelList[0].input_shape[1:]) # c*h*w
        
        yModels = [model( model_input ) for model in modelList ]
        yAvg = layers.average( yModels )

        model = models.Model(inputs = model_input, outputs = yAvg, name='ensemble')
        return model
    
    def ensembleModelDirectoryByAverageOutput( self, modelDirectory ):
        
        modelList = self.getModelsFromDirectory( modelDirectory )
        return self.ensembleModelsByAverageOutput( modelList )
    
    def getModelsFromDirectory( self, modelDirectory ):
        
        modelPaths = glob.glob( modelDirectory )
        logging.warning( msg= "# of models found: " + str( len( modelPaths ) ) )
        
        unitModels = []
        for path in modelPaths:
            modelName = self.getFileNameWithoutExtensionWindows( path )
            print( modelName )
            model = models.load_model( path )
            model.name = modelName
            unitModels.append( model )   
            
        return unitModels
                                              
    
    def getFileNameWithoutExtensionWindows( self, path ):
        
        fileNameWithExtension = path.split( "\\" )[-1]
        return fileNameWithExtension.split( "." )[0]                                        