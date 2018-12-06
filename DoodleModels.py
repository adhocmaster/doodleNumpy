
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
    
    def getModel(self, modelNo ):
        
        if modelNo == 1:
            #model 1
            model = models.Sequential()
            model.add( layers.Conv2D( 16, (3,3), activation = activations.relu, input_shape = (28, 28, 1 ) ) )
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
    
    def runModel( self, model, epochs = 5, batch_size = 64 ):
        history = model.fit( self.trX, self.trY, epochs = epochs, batch_size = batch_size )
        tsError, tsAcc = model.evaluate( self.tsX, self.tsY )
        return ( history, tsError, tsAcc )