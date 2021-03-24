from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.regularizers import l1



def train_model3(my_shape):
    print("----- 3 Layers network------")
    model = Sequential()
    model.add(Dense(my_shape, input_dim=my_shape, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/2, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['acc'])
    return model



def train_model4(my_shape):
    print("----- 4 Layers network------")
    model = Sequential()
    model.add(Dense(my_shape, input_dim=my_shape, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/2, activation='relu',kernel_constraint=maxnorm(3),activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/4, activation='relu',kernel_constraint=maxnorm(3),activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001)))
    model.add(Dense(my_shape/8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['acc'])
    return model


def train_model5(my_shape):
    print("----- 5 Layers network------")
    model = Sequential()
    model.add(Dense(my_shape, input_dim=my_shape, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/2, activation='relu',kernel_constraint=maxnorm(3),activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001)))
    model.add(Dense(my_shape/4, activation='relu',activity_regularizer=l1(0.001)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/8, activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/16, activation='relu',kernel_constraint=maxnorm(3),activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['acc'])
    return model

def train_model6(my_shape):
    print("----- 6 Layers network------")
    model = Sequential()
    model.add(Dense(my_shape, input_dim=my_shape, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/2, activation='relu',kernel_constraint=maxnorm(3),activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/4, activation='relu',activity_regularizer=l1(0.001)))
    model.add(Dense(my_shape/8, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/16, activation='relu',kernel_constraint=maxnorm(3),activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001)))
    model.add(Dense(my_shape/32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['acc'])
    return model

