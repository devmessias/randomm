#! /usr/bin/env python
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
import tensorflow as tf
import numpy as np
from io import  BytesIO
from sqlalchemy import create_engine, Column, Integer, String, Table, MetaData, Text, UnicodeText
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import argparse
import os
import warnings 
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


Base = declarative_base()
engine = create_engine('postgresql://postgres:nano@localhost:5432/randomautoencoder', pool_size=30)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)


input_unit_size = 28*28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], input_unit_size)
x_train = x_train.astype('float32')
x_train /= 255
show_size = 10

inputs = Input(shape=(input_unit_size,))
x = Dense(144, activation='relu')(inputs)
outputs = Dense(input_unit_size)(x)



class HiddensTable(Base):
    __tablename__ = 'hiddens'
    id = Column(Integer, primary_key=True)

    step = Column(Integer)
    hidden_id = Column(Integer)
    epoch = Column(Integer)
    ensemble = Column(Integer)
    data = Column(UnicodeText)

class VisiblesTable(Base):
    __tablename__ = 'visibles'
    id = Column(Integer, primary_key=True)

    step = Column(Integer)
    visible_id = Column(Integer)
    epoch = Column(Integer)
    ensemble = Column(Integer)
    data = Column(UnicodeText)
    
class WeightsTable(Base):
    __tablename__ = 'weights'
    id = Column(Integer, primary_key=True)

    step = Column(Integer)
    epoch = Column(Integer)
    ensemble = Column(Integer)
    data = Column(UnicodeText)

    
class WeightsTable2(Base):
    __tablename__ = 'weights2'
    id = Column(Integer, primary_key=True)

    step = Column(Integer)
    epoch = Column(Integer)
    ensemble = Column(Integer)
    data = Column(UnicodeText)
    
def create_tables():


    weights = Table('weights', MetaData(),
        Column('id', Integer, primary_key=True),
        Column('step', Integer),
        Column('epoch', Integer),
        Column('ensemble', Integer),
        Column('data', UnicodeText),


    )
    weights2 = Table('weights2', MetaData(),
        Column('id', Integer, primary_key=True),
        Column('step', Integer),
        Column('epoch', Integer),
        Column('ensemble', Integer),
        Column('data', UnicodeText),


    )
    hiddens = Table('hiddens', MetaData(),
        Column('id', Integer, primary_key=True),
        Column('hidden_id', Integer),
        Column('step', Integer),
        Column('epoch', Integer),
        Column('ensemble', Integer),
        Column('data', UnicodeText),


    )
   
    hiddens.create(engine)

    weights.create(engine)
    weights2.create(engine)

    
    
class History(callbacks.Callback):
    def  __init__(self, ensemble):
        self.ensemble = ensemble
        self.epoch = 1
        self.step = 0
        self.step_weight = 1
        self.step_hidden = 1

    def on_epoch_end(self, epoch, logs={}): 
        self.step = 0
        self.epoch += 1

    
    def on_batch_end(self, batch, logs={}):
        session =  DBSession()
        if self.step %  self.step_weight == 0 and False:
            list_connections = self.model.layers[1].get_weights()[0].flatten()
            m = np.reshape(list_connections, (336, 336))
            s = BytesIO()
            np.savetxt(s, m)
            new_weight = WeightsTable(
                step = self.step, 
                data = s.getvalue().decode(), 
                epoch = self.epoch, 
                ensemble = self.ensemble)

            session.add(new_weight)
            list_connections = self.model.layers[2].get_weights()[0].flatten()
            m = np.reshape(list_connections, (336, 336))
            s = BytesIO()
            np.savetxt(s, m)
            new_weight = WeightsTable2(
                step = self.step, 
                data = s.getvalue().decode(), 
                epoch = self.epoch, 
                ensemble = self.ensemble)

            session.add(new_weight)
            del s
            del list_connections
            del m
            session.commit()

        if self.step %  self.step_hidden == 0:
            get_layer_hidden_output = K.function([self.model.layers[0].input],
                                          [self.model.layers[1].output])
        
            get_layer_output = K.function([self.model.layers[0].input],
                              [self.model.layers[2].output])

            visible_outputs = get_layer_output([x_train[0:show_size**2]])[0]
            
            hidden_outputs = get_layer_hidden_output([x_train[0:show_size**2]])[0]
            
            size = int(np.sqrt(hidden_outputs[0].shape[0]))
            for item_id in range(len(hidden_outputs)):
                #for i in range(show_size):
                    #for j in range(show_size):
                try:
                    h = hidden_outputs[item_id]
                    s = BytesIO()
                    np.savetxt(s, h.reshape(size, size))

                    new_hidden = HiddensTable(
                                step = self.step, 
                                epoch = self.epoch, 
                                ensemble = self.ensemble,
                                hidden_id = item_id, 
                                data = s.getvalue().decode()
                    )

                    session.add(new_hidden)
                    session.commit()
                except:
                    pass
        self.step +=1
        

        session.close()


def create_exp(i):
    model = Model(input=inputs, output=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adadelta', )
    model.fit(x_train, x_train, epochs=3,  batch_size=256, callbacks=[History(i)], verbose=0  )
    
    
def main():
    parser = argparse.ArgumentParser(
        prog="",
    )
    parser.add_argument(
        "--number", "-n",
        required=True,
        help="number of ensemble"
    )
    args = parser.parse_args()

    number = int(args.number)
    create_exp(number)
    return None
    
if __name__ == "__main__":
    main()