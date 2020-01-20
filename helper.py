import numpy as np
import pandas as pd
from scipy import stats
import collections
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


east_low_mape_defect = ['20180116','20180925','20181130','20181202','20181216','20181226','20190118','20190124',
                        '20190129','20190228','20190220','20171114','20171223','20180115','20180122','20180131',
                        '20180201','20180206','20180210','20180219','20180223','20180225','20180326','20180406',
                        '20180417','20180626','20180707','20180709','20180710','20180712','20180718','20180720',
                        '20180823','20180905','20181211','20181010','20181018','20181021','20181111','20181118',
                        '20181213','20181121','20181124','20181129','20181201','20181204','20181217','20181224',
                        '20190110','20190219','20190308','20190912','20191114','20191213','20190104','20190126',
                        '20190206','20190305','20190313','20190924','20191118','20190108','20190204','20190217',
                        '20190307','20190901','20191107','20191210','20170914','20170930','20171005','20171019',
                        '20171020','20171021','20171022','20171104','20171108','20171109','20171202','20171205',
                        '20171207','20171208','20171211','20171220','20180124','20180127','20180202','20180203',
                        '20180205','20180213','20180228','20180324','20180325','20180614','20180617','20180620',
                        '20180622','20200104','20200106','20180709','20180719','20180814','20181026','20181107',
                        '20181119','20181122','20181128','20181208','20181209','20181210','20181218','20181219',
                        '20181227','20190102','20190117','20190108','20190130','20200105','20200110','20190209',
                        '20190215','20190216','20190228','20190301','20190306','20190309','20190310','20190323',
                        '20190426','20190720','20191019','20191104','20191124','20191128','20191212','20191226',
                        '20191231']


## Data Processing

def create_dataset(data):
    dataX, dataY = [],[]
    for i in range(len(data)):
        dataX.append(data[i][:-1])
        dataY.append([data[i][-1]])
        
    return np.array(dataX), np.array(dataY)

    # Mean Absolute Percentage Error
def mean_absolute_percentage_error(data_true, data_predict):
    error = 0
    count = 0
    data_true_de = sum(data_true)/len(data_true)
    
    for i in range(len(data_true)):
        error += np.abs((data_true[i]-data_predict[i])/data_true_de)

    return((error/len(data_true))*100)

def add_noise(trainX, trainY):

    trainX = [trainX[i]+np.random.normal(0,0.0001) for i in range(len(trainX))]
    trainY = [trainY[j]+np.random.normal(0,0.0001) for j in range(len(trainY))]
    
    return np.array(trainX), np.array(trainY) 

# get weather condition from darksky_weather_data.csv
def get_weather():
    weather_file = pd.read_csv('darksky_weather_data.csv')
    weather_data = {}
    snow_count = 100
    for i in range(len(weather_file)):
        day = str(weather_file.loc[i]['date_time'])[:8]
        time = str(weather_file.loc[i]['date_time'])[8:10]
        condition = weather_file.loc[i]['icon']
                    
        if condition == 'clear-night':
            condition = 'sunny'
        elif condition == 'partly-cloudy-night' or condition == 'partly-cloudy-day':
            condition = 'partly-cloudy'
        elif condition == 'clear-day':
            condition = 'sunny'
        
        if day not in weather_data:
            weather_data[day] = [condition]
        else:
            weather_data[day].append(condition)            
        
    for day in weather_data:
        condition = weather_data[day]
        
        if 'snow' in condition:
            weather_data[day] = 'snow'
            snow_count = 0
        elif 'sleet' in condition:
            weather_data[day] = 'snow'
            snow_count = 0
        elif 'fog' in condition:
            weather_data[day] = 'fog'
        elif 'rain' in condition:
            weather_data[day] = 'rain'
        elif 'wind' in condition:
            weather_data[day] = 'wind' 
        elif snow_count < 3:
            weather_data[day] = 'snow' 
        else:
            weather_data[day] = collections.Counter(condition).most_common()[0][0]
        snow_count += 1
            
    return weather_data


def mape_distribution(dataset):
    # drop weather column
    dataset = dataset.drop(['weather'],axis=1)
    n = len(dataset)
    daily_data = {}
    
    date_col = 'date_time'

    for i in dataset[date_col]:
        if str(i)[:8] not in daily_data:
            daily_data[str(i)[:8]] = []
            
        
    for i in dataset.index:
        day = str(dataset[date_col].loc[i])[:8]
        daily_data[day].append(dataset.loc[i])
        
        
    day_dict = {'0-5':[],'5-10':[],'10-15':[],'15-20':[],'20-30':[],'30-40':[],'>40':[]}
    data_dict = {'0-5':[],'5-10':[],'10-15':[],'15-20':[],'20-30':[],'30-40':[],'>40':[]}
    mape_dict = {'0-5':[],'5-10':[],'10-15':[],'15-20':[],'20-30':[],'30-40':[],'>40':[]}
    
    for day, data in daily_data.items():
        one_day = pd.DataFrame(data).drop([date_col],axis=1)
        
        if len(one_day) < 10:
            continue
            
        mape,_ = model_predict(one_day)
        mape = np.mean(mape)
        
        if (mape <= 5):
            day_dict['0-5'].append(day)
            data_dict['0-5'].append(one_day)
            mape_dict['0-5'].append(mape)
        elif (mape > 5) and (mape <= 10):
            day_dict['5-10'].append(day)
            data_dict['5-10'].append(one_day)
            mape_dict['5-10'].append(mape)
        elif (mape > 10) and (mape <= 15):
            day_dict['10-15'].append(day)
            data_dict['10-15'].append(one_day)
            mape_dict['10-15'].append(mape)
        elif (mape > 15) and (mape <= 20):
            day_dict['15-20'].append(day)
            data_dict['15-20'].append(one_day)
            mape_dict['15-20'].append(mape)
        elif (mape > 20) and (mape <= 30):
            day_dict['20-30'].append(day)
            data_dict['20-30'].append(one_day)
            mape_dict['20-30'].append(mape)
        elif (mape > 30) and (mape <= 40):
            day_dict['30-40'].append(day)
            data_dict['30-40'].append(one_day)
            mape_dict['30-40'].append(mape)
        else:
            day_dict['>40'].append(day)
            data_dict['>40'].append(one_day)
            mape_dict['>40'].append(mape)
            
    return [day_dict,data_dict,mape_dict]


def generate_condition_label(dataset,threshold,direction):
        
    west_defect = []
    
    lower_defect = []
    
    if direction == "east":
        defect = east_low_mape_defect
    elif direction == 'west':
        defect = west_low_mape_defect
    else:
        direct = lower_low_mape_defect
    
    # Calculate the daily MAPE and return its corresponding daily dataset
    day_dict,data_dict,mape_dict = mape_distribution(dataset)
    
    if threshold == 1:
        low_mape_day = day_dict['0-5']
    elif threshold == 2:
        low_mape_day = [*day_dict['0-5'],*day_dict['5-10']]
    elif threshold == 3:
        low_mape_day = [*day_dict['0-5'],*day_dict['5-10'],*day_dict['10-15']]
    else:
        print("No valid threshold provided. Please pick among [1,2,3] to define the level of classification")
        return 
    
    normal_day = []

    for ith, data in enumerate(low_mape_day):
        day = low_mape_day[ith]
        if day not in defect:
            normal_day.append(day)
            
    condition = []
    
    n = len(dataset)

    for i in range(n):
        day = str(dataset.loc[i]['date_time'])[:8]
        if day in normal_day:
            condition.append('normal')
        else:
            condition.append('anomaly')
        
    dataset['condition'] = condition
    
    return dataset


## Machine Learning Models

def random_forest(X,Y):
    
    cvscores = []
    k_fold = KFold(n_splits=2, shuffle=True, random_state=0)
        
    for train_index, test_index in k_fold.split(X, Y):
        
        X_train, y_train = add_noise(X[train_index], Y[train_index])
        
        X_test = X[test_index]
        y_test = Y[test_index]        
        
        rf_model = RandomForestRegressor(n_estimators=10,max_depth=10, random_state=0).fit(X_train, y_train)
        rf_prediction = rf_model.predict(X_test)
        score = mean_absolute_percentage_error(y_test, rf_prediction)
        cvscores.append(score)
        
    return np.mean(cvscores),rf_model

def k_nearest_neighbor(X,Y):
    
    cvscores = []
    k_fold = KFold(n_splits=4, shuffle=True, random_state=0)
        
    for train_index, test_index in k_fold.split(X, Y):
        
        X_train, y_train = add_noise(X[train_index], Y[train_index])
        
        X_test = X[test_index]
        y_test = Y[test_index]        
        
        knn_model = KNeighborsRegressor().fit(X_train, y_train)
        knn_prediction = knn_model.predict(X_test)
        score = mean_absolute_percentage_error(y_test, knn_prediction)
        cvscores.append(score)
        
    return np.mean(cvscores),knn_model


def rf_return_pred(trainX, trainY, testX, testY):
    
    model = RandomForestRegressor(n_estimators=10,max_depth=10, random_state=0)
    model.fit(trainX, trainY)
    
    pred_train = model.predict(trainX)
    pred_test = model.predict(testX)
        
    predictions = np.array([*pred_train,*pred_test])
    test = np.array([*trainY,*testY])
    
    score = mean_absolute_percentage_error(test, predictions)
    
    return [score,predictions]


def model_predict(dataset, return_pred=False):
    
    result = []
    prediction = []
    n = len(dataset)
    
    for col in dataset.columns:
            
        # prepare data
        X = dataset.loc[:, dataset.columns!=col].values
        Y = dataset.loc[:, dataset.columns==col].values
        
#         # normalize the data
#         scaler = MinMaxScaler(feature_range=(0,1))
#         X = np.array(scaler.fit_transform(X))
#         Y = np.array(scaler.fit_transform(Y))
        
        res, model = random_forest(X, Y)
#         res, model = k_nearest_neighbor(X,Y)
        
        if return_pred == False:
        
#             # normalize the data
#             scaler = MinMaxScaler(feature_range=(0,1))
#             X = np.array(scaler.fit_transform(X))
#             Y = np.array(scaler.fit_transform(Y))
        
#             res, model = random_forest(X, Y)
            result.append(res)
            
        else:
            
            pred = model.predict(X)
            result.append(res)
            prediction.append(pred)
          
            # prepare the data
#             train_test_ratio = 0.4
#             train_size = int(train_test_ratio*n)
#             trainX, trainY = X[:train_size], Y[:train_size]
#             testX, testY = X[train_size:], Y[train_size:]
            
#             res, pred = rf_return_pred(trainX, trainY, testX, testY)
#             result.append(res)
#             prediction.append(pred)
        
    return [result,prediction]



## Visualization

def raw_data_plot(dataset,title,save=False):
    # make the plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1,1,1)
    
    box = plt.plot(dataset)

    indices = [i for i in dataset.index]
    
    ax.set_title(title,fontsize=16)
    ax.set_xticks(np.linspace(indices[0],indices[-1],12))
    ax.xaxis.set_ticklabels(['7am','8am','9am','10am','11am','12pm','1pm','2pm','3pm','4pm','5pm','6pm'])
    ax.set_xlabel('Time',fontsize=14)
    ax.set_ylabel('Power(W)',fontsize=14)
    plt.legend(['Panel '+str(i+1) for i in range(len(dataset)-1)])
    
    if save == True:
        plt.savefig(title + '.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        
        
def histogram(data_dict,file_name,save=False):
    
    data = []
    for i in data_dict:
        data.extend(data_dict[i])
        
    for j, mape in enumerate(data):
        if mape > 50:
            data[j] = 50

    res = stats.cumfreq(data, numbins=15, defaultreallimits=(0, 50))
    x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
    cum_y = [i/(max(res.cumcount)-min(res.cumcount))*100 for i in res.cumcount]
    
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(data, bins=15,histtype='bar', ec='black')
    ax1.set_title('Histogram')
    ax1.set_xlabel('MAPE(%)',fontsize=12)
    ax1.set_ylabel('Frequency (Days)',fontsize=12)
    # ax2.bar(x, res.cumcount, width=res.binsize)
    ax2.plot(x,cum_y,'-o')
    ax2.set_title('Cumulative Histogram')
    ax2.set_xlim([x.min(), x.max()])
    ax2.set_xlabel('MAPE(%)',fontsize=12)
    ax2.set_ylabel('Dataset Percentage (%)',fontsize=12)
    
    if save is True:
        plt.savefig(file_name+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')