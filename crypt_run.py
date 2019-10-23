import pandas as pd

import sklearn

import random

import time

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.callbacks import ModelCheckpoint

#data=pd.read_csv("LTC-USD.csv",names=["time","low","high","open","close","volume"])

BITCOIN_TYPE_PREDICT = 'BCH-USD'

FUTURE_TIME_TO_PREDICT = 5

PREDICT_LAST_MINUTES = 60

def run_program():
    
    empty_df = pd.DataFrame()


    all_files= ['BCH-USD','BTC-USD','ETH-USD','LTC-USD']

    for file in all_files:

        file_name='{}.csv'.format(file)

        data=pd.read_csv(file_name,names=["time","low","high","open","close","volume"])

        data.rename(columns={"close":file+" close","volume":file+" volume"},inplace=True)

        #Merge the data

        data.set_index("time", inplace=True)

        data = data[[file+" close",file+" volume"]]   #only closing price and volume

        if len(empty_df)==0:

            empty_df = data
            
        else:

            empty_df = empty_df.join(data)

        empty_df.fillna(method="ffill", inplace=True)  

        empty_df.dropna(inplace=True)   # fills in place by dropping the unwanted values


        empty_df['future'] = empty_df[BITCOIN_TYPE_PREDICT+' close'].shift(-FUTURE_TIME_TO_PREDICT)

        empty_df['target'] = list(map(predict, empty_df[BITCOIN_TYPE_PREDICT+' close'], empty_df['future']))

        print(empty_df.head())

        times = sorted ( empty_df.index.values )

        last_10_percent = times [-int(0.1*len(times))] #Avoids overfitting

        print( last_10_percent )

        #seperate out validation data with training data
        
        validation_empty_df = empty_df[(empty_df.index >= last_10_percent)]  


        empty_df = empty_df[(empty_df.index < last_10_percent)]


        training_set_x , training_set_y = process(empty_df)

        validation_set_x , validation_set_y = process( validation_empty_df )

        print(f"train data: {len(training_set_x)} validation: {len(validation_set_x)}")
        
        print(f"Dont buys: {training_set_y.count(0)}, buys: {training_set_y.count(1)}")

        print(f"VALIDATION Dont buys: {validation_set_y.count(0)}, buys: {validation_set_y.count(1)}")

        model = Sequential()

        model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))

        model.add(Dropout(0.2))

        model.add(BatchNormalization())

        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='softmax'))

        
def process (df):

    #Normalizing the data

    df = df.drop('future',1)

    for col in df.columns:

        df[col]= df[col].pct_change()

        df = df.dropna( inplace = True )

        df[col] = sklearn.preprocessing(df[col].values) #normalizes the data

    df = df.dropna(inplace= True) #drops the null values of the data

    data=[]

    current_set_days = deque(maxlen=PREDICT_LAST_MINUTES)   #pops out the last element

    for columns in df.values:   #makes a list of lists including the target column

        current_set_days.append([l for l in columns[:-1]])

        if len(current_set_days) == PREDICT_LAST_MINUTES:

            data.append([np.array(current_set_days)],i[-1])

    random.shuffle(data)

    buy=[]

    sell = []

    for info, target in data:

        if target == "DON'T BUY":

            sell.append([info,target])

        elif target == 'BUY':

            buy.append([info,target])


    random.shuffle(buy)  

    random.shuffle(sell)

    minimum_length = min (len(buy),len(sell))

    buy = minimum_length[:1]

    sell = minimum_length[:1]

    data = buy + sell

    random.shuffle(data) 
    
    list1= []

    list2 = []
        
    for info,target in data:

        list1.append(info)

        list2.append(target)
    
    
    return np.array(list1),list2
    
        
        

def predict(current,future):

    if future > current :

        return "BUY"

    else:

        return "DON'T BUY"




    
if __name__=='__main__':

    run_program()
 
