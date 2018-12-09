import glob
import logging, sys, math
import numpy as np
import matplotlib.pyplot as plt

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
        
        self.tsYLabels = np.argmax( self.tsY, axis = 1 )
        
        print( "train X:", self.trX.shape )
        print( "test X:", self.tsX.shape )
        print( "train Y:", self.trY.shape )
        print( "test Y:", self.tsY.shape )
        pass
    
    def runModel( self, model, epochs = 5, batch_size = 64 ):
        history = model.fit( self.trX, self.trY, epochs = epochs, batch_size = batch_size )
        tsLoss, tsAcc = model.evaluate( self.tsX, self.tsY )
        return ( history, tsLoss, tsAcc )
    
    def evaluateModel( self, model ):
        return model.evaluate( self.tsX, self.tsY )
    
    def evaluateEnsemble( self, ensemble, batch_size = 32 ):
        
        pred = ensemble.predict( self.tsX, batch_size = batch_size)
        predLabels = np.argmax(pred, axis=1)
        error = np.sum(np.not_equal(predLabels, self.tsYLabels ) ) / len( self.tsYLabels )
        
        return error;
    
    def getModel(self, modelNo ):
        
        model_input = layers.Input( shape = ( 28, 28, 1 ) )
        
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
        
            model.name = "basic CNN, 32"
            return model
    
        if modelNo == 2:
            #model 1
            model = models.Sequential()
            model.add( layers.Conv2D( 32, (3,3), activation = activations.relu, input_shape = ( 28, 28, 1 ) ) )
            model.add( layers.MaxPooling2D( (2,2) ) )
            model.add( layers.Conv2D( 64, (3,3), activation = activations.relu ))
            model.add( layers.MaxPooling2D( (2,2) ) )
            model.add( layers.Conv2D( 64, (3,3), activation = activations.relu ))
            model.add( layers.Flatten() )
            model.add( layers.Dense( 64, activation = activations.relu ) )
            model.add( layers.Dense( 10, activation = activations.softmax ) )
            model.summary()
            model.compile( 
                optimizer = optimizers.rmsprop( lr = .001 ), 
                loss = losses.categorical_crossentropy,
                metrics = [ metrics.categorical_accuracy ] 
            )
        
            model.name = "basic CNN, 64"
            return model
        
        if modelNo == 3:
            return self.getConvPoolCNNCModel( model_input )
        if modelNo == 4:
            return self.getAllCNNC( model_input )
        if modelNo == 5:
            return self.NINCNN( model_input )
    
    def getConvPoolCNNCModel( self, model_input ):
        
        return self.getConvPoolCNNCModelReduced( model_input )
    
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
    
    def getConvPoolCNNCModelReduced( self, model_input ):
        
        x = layers.Conv2D(96, kernel_size=(3, 3), activation=activations.relu, padding = 'same')(model_input)
        x = layers.Conv2D(96, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.Conv2D(96, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
        
        x = layers.Conv2D(192, (3, 3), activation=activations.relu, padding = 'same')(x)
        x = layers.Conv2D(192, (1, 1), activation=activations.relu)(x)
        x = layers.Conv2D(10, (1, 1))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Activation(activation='softmax')(x)

        model = models.Model(model_input, x, name='conv_pool_cnn reduced')

        model.summary()
        model.compile( 
            optimizer = optimizers.rmsprop( lr = .001 ), 
            loss = losses.categorical_crossentropy,
            metrics = [ metrics.categorical_accuracy ] 
        )
        
        return model
    def getAllCNNC( self, model_input ):
           
        x = layers.Conv2D(96, kernel_size=(3, 3), activation= activations.relu, padding = 'same')(model_input)
        x = layers.Conv2D(96, (3, 3), activation= activations.relu, padding = 'same')(x)
        x = layers.Conv2D(96, (3, 3), activation= activations.relu, padding = 'same', strides = 2)(x)
        x = layers.Conv2D(192, (3, 3), activation= activations.relu, padding = 'same')(x)
        x = layers.Conv2D(192, (3, 3), activation= activations.relu, padding = 'same')(x)
        x = layers.Conv2D(192, (3, 3), activation= activations.relu, padding = 'same', strides = 2)(x)
        x = layers.Conv2D(192, (3, 3), activation= activations.relu, padding = 'same')(x)
        x = layers.Conv2D(192, (1, 1), activation= activations.relu)(x)
        x = layers.Conv2D(10, (1, 1))(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Activation(activation='softmax')(x)

        model = models.Model(model_input, x, name='all_cnn')

        model.summary()
        model.compile( 
            optimizer = optimizers.rmsprop( lr = .001 ), 
            loss = losses.categorical_crossentropy,
            metrics = [ metrics.categorical_accuracy ] 
        )

        return model
    
    def NINCNN( self, model_input ):
        
        #mlpconv block 1
        x = layers.Conv2D(32, (5, 5), activation= activations.relu,padding='valid')(model_input)
        x = layers.Conv2D(32, (1, 1), activation= activations.relu)(x)
        x = layers.Conv2D(32, (1, 1), activation= activations.relu)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.5)(x)

        #mlpconv block2
        x = layers.Conv2D(64, (3, 3), activation= activations.relu,padding='valid')(x)
        x = layers.Conv2D(64, (1, 1), activation= activations.relu)(x)
        x = layers.Conv2D(64, (1, 1), activation= activations.relu)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.5)(x)

        #mlpconv block3
        x = layers.Conv2D(128, (3, 3), activation= activations.relu,padding='valid')(x)
        x = layers.Conv2D(32, (1, 1), activation= activations.relu)(x)
        x = layers.Conv2D(10, (1, 1))(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Activation(activation='softmax')(x)

        model = models.Model(model_input, x, name='nin_cnn')

        model.summary()
        model.compile( 
            optimizer = optimizers.rmsprop( lr = .001 ), 
            loss = losses.categorical_crossentropy,
            metrics = [ metrics.categorical_accuracy ] 
        )
        
        return model
    
    
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
    
    def plotModelTrainPerformance( self, model, savePath = "" ):
        
        history = model.history
        
        print( history.history )
        
        epochX = np.arange( len( history.epoch ) ) + 1
        plt.close()
        plt.plot( epochX, history.history['categorical_accuracy'], color = "blue", label = "accuracy" )
        plt.plot( epochX, history.history['loss'], color = "red", label = "loss" )

        plt.title( 'Epoch vs Test ' + model.name )
        plt.xlabel( "Epoch" )
        plt.ylabel( "amount" )
        plt.xticks( epochX )
        plt.legend()
        
        if savePath != "":
            plt.savefig( savePath )
            print( "saved model performance plot at:", savePath )
        
        