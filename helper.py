import collections
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr  



from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings('ignore')

west_defect = []
    
lower_defect = []

class data_processing:
    
    def __init__(self):
        
        self.east_defect = []
        self.west_defect = []
        self.lower_defect = []
        
    ## Data Processing

    def create_dataset(self,data):
        dataX, dataY = [],[]
        for i in range(len(data)):
            dataX.append(data[i][:-1])
            dataY.append([data[i][-1]])
        
        return np.array(dataX), np.array(dataY)
    
    # get weather condition from darksky_weather_data.csv
    def get_weather(self):
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
    
    def mape_distribution(self,dataset,return_mape=False):
        
        Model = modeling()
       
        n = len(dataset)
        daily_data = {}
    
        date_col = 'date_time'

        for i in dataset[date_col]:
            if str(i)[:8] not in daily_data:
                daily_data[str(i)[:8]] = []
        
        for i in dataset.index:
            day = str(dataset[date_col].loc[i])[:8]
            daily_data[day].append(dataset.loc[i])
            
        if return_mape == False:
            for day in daily_data:
                daily_data[day] = pd.DataFrame(daily_data[day]).drop(['date_time'],axis=1)
            return daily_data
        
        day_dict = {'0-5':[],'5-10':[],'10-15':[],'15-20':[],'20-30':[],'30-40':[],'>40':[]}
        data_dict = {'0-5':[],'5-10':[],'10-15':[],'15-20':[],'20-30':[],'30-40':[],'>40':[]}
        mape_dict = {'0-5':[],'5-10':[],'10-15':[],'15-20':[],'20-30':[],'30-40':[],'>40':[]}
    
        for day, data in daily_data.items():
            one_day = pd.DataFrame(data).drop(['date_time'],axis=1)
        
            if len(one_day) < 10:
                continue
                           
            mape,_ = Model.model_predict(one_day)
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
    
    
    def daily_dict(self,dataset,data_col=None):
        n = len(dataset)
        day_data = {}

        for i in range(n):
            day = str(dataset.loc[i]['date_time'])[:8]
            if day not in day_data:
                day_data[day] = []
                if data_col != None:
                    for col in data_col:
                        day_data[day].append(dataset.loc[i][col])
                    
        return day_data
    
    def daily_feature(self,dataset):
        day_list = list(set(dataset['date']))
        pass_day = []
        n = len(dataset)
        df = pd.DataFrame()
    
        temp_col = []
        weather_col = []
        cloud_col = []
        power_col = []
        corr_col = []
        label_col = []
    
        for i in range(n):
            day = dataset.loc[i]['date']
            if day not in pass_day:
                pass_day.append(day)
                temp_col.append(dataset.loc[i]['temperature'])
                weather_col.append(dataset.loc[i]['weather_summary'])
                cloud_col.append(dataset.loc[i]['cloudCover'])
                power_col.append(dataset.loc[i]['power_level'])
                corr_col.append(dataset.loc[i]['correlation'])
                label_col.append(dataset.loc[i]['label'])
    
        df['date'] = pass_day
        df['temperature'] = temp_col
        df['cloudCover'] = cloud_col
        df['power_level'] = power_col
        df['correlation'] = corr_col
        df['label'] = label_col
    
        return df 
  

    def generate_label(self,dataset,threshold,direction):
        
        Model = modeling()
    
        if direction == "east":
            defect = self.east_defect
        elif direction == 'west':
            defect = self.west_defect
        else:
            direct = self.lower_defect
    
        # Calculate the daily MAPE and return its corresponding daily dataset
        day_dict,data_dict,mape_dict = self.mape_distribution(dataset)
    
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
    
    
    def feature_generate(self,dataset):
        
        model = modeling()
        
        # Create daily raw data
        raw_data = self.mape_distribution(dataset)
        
        weather_collect = collections.Counter(dataset['weather']).most_common()
        weather_dict = {}
        for ith, weather_info in enumerate(weather_collect):
            weather_dict[weather_info[0]] = ith
    
        feature_dict = {'date':[],'mse':[],'std':[],'corr':[],'max_power':[],'weather':[],'cond':[]}

        for day in raw_data:
            
            if len(raw_data[day]) < 20:
                continue
    
            feature_dict['date'].append(day)
            feature_dict['weather'].append(weather_dict[raw_data[day]['weather'].values[0]])
            data = raw_data[day].drop(['weather','condition'],axis=1)
    
            # MSE and STD features
            res,_ = model.model_predict(data)
            feature_dict['mse'].append(np.mean(res))
            feature_dict['std'].append(np.std(res))
    
            # Correlation feature
            res = []
            first = 'panel_1_power'
            for col in data.columns:
                if col != first:
                    res.append(pearsonr(data[col].values,data[first].values)[0])
    
            feature_dict['corr'].append(min(res))
    
            # Max power feature
            power = []
            for col in data.columns:
                power.append(max(data[col]))
    
            feature_dict['max_power'].append(np.mean(power))
    
            # Condition
            condition = dataset[dataset['date_time']==float(day)]['condition'].values[0]
            feature_dict['cond'].append(condition)
            
        df = pd.DataFrame()

        for key,value in feature_dict.items():
            df[key] = value
        
        return df
    
    
class modeling:
    def __init__(self):
        self.regression_model_list = ['LR','SVR','DT','RF','MLP']
        self.classification_model_list = ['LR','SVM','DT','RF','MLP']
       
    ## Machine Learning Models

    def random_forest(self,X,Y):
    
        cvscores = []
        k_fold = KFold(n_splits=3, shuffle=True, random_state=0)
        
        for train_index, test_index in k_fold.split(X, Y):
        
            X_train, y_train = self.add_noise(X[train_index], Y[train_index])
        
            X_test = X[test_index]
            y_test = Y[test_index]        
        
            rf_model = RandomForestRegressor(n_estimators=10,max_depth=10, random_state=0).fit(X_train, y_train)
            rf_prediction = rf_model.predict(X_test)
            score = self.mean_absolute_percentage_error(y_test, rf_prediction)
#             score = mean_squared_error(y_test, rf_prediction)
            cvscores.append(score)
        
        return np.mean(cvscores),rf_model
    
    def linear_regression(self,X,Y):
    
        cvscores = []
        k_fold = KFold(n_splits=3, shuffle=True, random_state=0)
        
        for train_index, test_index in k_fold.split(X, Y):
        
            X_train, y_train = self.add_noise(X[train_index], Y[train_index])
        
            X_test = X[test_index]
            y_test = Y[test_index]        
        
            lr_model = LinearRegression().fit(X_train, y_train)
            lr_prediction = lr_model.predict(X_test)
            score = self.mean_absolute_percentage_error(y_test, lr_prediction)
            cvscores.append(score)
        
        return np.mean(cvscores),lr_model
    
    def naive_approach(self,X,Y):
    
        cvscores = []
        k_fold = KFold(n_splits=3, shuffle=True, random_state=0)
        
        for train_index, test_index in k_fold.split(X, Y):
            
            X_test = X[test_index]
            y_test = Y[test_index]
                
            naive_prediction = np.array([np.mean(i) for i in X_test])
            score = self.mean_absolute_percentage_error(y_test, naive_prediction)
#             score = mean_squared_error(y_test, naive_prediction)
            cvscores.append(score)
        
        return np.mean(cvscores)
    
    
    def random_forest_class(self,dataset):
        
        process = data_processing()
    
        date = dataset['date'].values
        dataset = dataset.drop(['date'],axis=1).values
     
        X, Y = process.create_dataset(dataset)
        cvscores = []
        cm = []
        
        kfold = StratifiedKFold(n_splits=3, shuffle=True)
        
        for train_index, test_index in kfold.split(X, Y):
            
            X_train = X[train_index]
            y_train = Y[train_index]
            X_test = X[test_index]
            y_test = Y[test_index]
            test_date = date[test_index]
        
            rf_model = RandomForestClassifier(max_depth=15, random_state=1).fit(X_train, y_train)
            rf_prediction = rf_model.predict(X_test)
            cvscores.append(accuracy_score(y_test, rf_prediction)*100)
            cm.append(confusion_matrix(y_test, rf_prediction))
        
#         return [cvscores, cm]
        return [cvscores, cm, rf_model,X_test,y_test,test_date]

    def k_nearest_neighbor(self,X,Y):
    
        cvscores = []
        k_fold = KFold(n_splits=4, shuffle=True, random_state=0)
        
        for train_index, test_index in k_fold.split(X, Y):
        
            X_train, y_train = add_noise(X[train_index], Y[train_index])
        
            X_test = X[test_index]
            y_test = Y[test_index]        
        
            knn_model = KNeighborsRegressor().fit(X_train, y_train)
            knn_prediction = knn_model.predict(X_test)
            score = self.mean_absolute_percentage_error(y_test, knn_prediction)
            cvscores.append(score)
        
        return np.mean(cvscores),knn_model


    def rf_return_pred(self,trainX, trainY, testX, testY):
    
        model = RandomForestRegressor(n_estimators=10,max_depth=10, random_state=0)
        model.fit(trainX, trainY)
    
        pred_train = model.predict(trainX)
        pred_test = model.predict(testX)
        
        predictions = np.array([*pred_train,*pred_test])
        test = np.array([*trainY,*testY])
    
        score = self.mean_absolute_percentage_error(test, predictions)
    
        return [score,predictions]
    
    def add_noise(self,trainX, trainY):

        trainX = [trainX[i]+np.random.normal(0,0.0001) for i in range(len(trainX))]
        trainY = [trainY[j]+np.random.normal(0,0.0001) for j in range(len(trainY))]
    
        return np.array(trainX), np.array(trainY)


    def model_predict(self,dataset, model_option, return_pred=False):
           
        result = []
        prediction = []
        n = len(dataset)
    
        for col in dataset.columns:
            
            # prepare data
            X = dataset.loc[:, dataset.columns!=col].values
            Y = dataset.loc[:, dataset.columns==col].values
        
            if return_pred == False:
        
                # normalize the data
                scaler = MinMaxScaler(feature_range=(0,1))
                X = np.array(scaler.fit_transform(X))
                Y = np.array(scaler.fit_transform(Y))
        
                if model_option == 'RF':
                    res, model = self.random_forest(X, Y)
                elif model_option == 'LR':
                    res, model = self.linear_regression(X,Y)
                elif model_option == 'Naive':
                    res = self.naive_approach(X,Y)
                else:
                    print("Model is not defined")
                    return 
                result.append(res)
            
            else:
                
                res, model = self.random_forest(X, Y)
                pred = model.predict(X)
                result.append(res)
                prediction.append(pred)
        
        return [result,prediction]
    
    
    # Mean Absolute Percentage Error
    def mean_absolute_percentage_error(self,data_true, data_predict):
        
        error = 0
 
        data_true_de = sum(data_true)/len(data_true)
    
        for i in range(len(data_true)):
            error += np.abs((data_true[i]-data_predict[i])/data_true_de)

        return((error/len(data_true))*100) 
    

class visualizer:
    def __init__(self):
        self.visualizer_tool = ['raw','hist']
    ## Visualization

    def daily_plot(self, data, dpi, save=False):
    
        time_weather = {}
        weathers = data['weather_summary'].values
        hours = list(data['time'].values)
    
        for t in hours:
            hour = t.split(':')[0]
            if hour not in time_weather:
                time_weather[hour] = weathers[hours.index(t)]
            
        date = data['date'].values[0]
        drop_col = [d_col for d_col in data.columns if d_col[:5] != 'panel']
        data = data.drop(drop_col,axis=1)
    
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(1,1,1)
    
        box = plt.plot(data)
        plt.legend(['Panel '+str(i+1) for i in range(len(data)-1)])

        indices = [i for i in data.index]

        title = 'Power Output on '+'-'.join(date.split('/'))
    
        ax.set_title(title,fontsize=30,y=1.05)
    
        # Time x-axis(bottom)
        ax.set_xticks(np.linspace(indices[0],indices[-1],len(time_weather)))
        ax.xaxis.set_ticklabels([t+':00' for t in time_weather], fontsize=13)
        ax.set_xlabel('Time',fontsize=30)
    
        # Power y-axis(left)
        power_y = np.linspace(0,int(max(data.max()))*1.1,5)
        power_y = [int(i) for i in power_y]
        ax.set_yticks(power_y)
        ax.yaxis.set_ticklabels(power_y, fontsize=20)
        ax.set_ylabel('Power(W)',fontsize=30)
    
#         # Weather x-axis(top)
#         ax2 = ax.twiny()    
#         ax2.set_xlim(indices[0],indices[-1])
#         ax2.set_xticks(np.linspace(indices[0],indices[-1],len(time_weather)))
#         ax2.set_xticklabels([time_weather[t] for t in time_weather],rotation=45)
    
        if save == True:
    
            plt.savefig(title + '.jpg', format='jpg', dpi=dpi, bbox_inches='tight')
            
            
    def pie_chart(self,dataset,title,save=False,resolution=None):
        
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.1f}%'.format(p=pct,v=val)
            return my_autopct

        counts = collections.Counter(dataset)
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(1,1,1)
        value = [v for v in counts.values()]
        plt.pie(value, labels=[k for k in counts], autopct=make_autopct(value),
                textprops={'fontsize': 15},pctdistance=0.7,labeldistance=1.2)
    
        ax.set_title(title,fontsize=30)
        
        if save == True:
            plt.savefig(title + '.jpg', format='jpg', dpi=resolution, bbox_inches='tight')
        
        
    def histogram(self,data_dict,file_name=False,save=False,resolution=None):
    
        if type(data_dict) == dict:
            data = []
            for i in data_dict:
                data.extend(data_dict[i])
        elif type(data_dict) == list:
            data = data_dict
        else:
            print("Input must be dictionary or list")
            return 
       
        
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
            plt.savefig(file_name+'.jpg', format='jpg', dpi=resolution, bbox_inches='tight')
            
            
    def confusion_matrix_plot(self,cm,acc,save=False):
    
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1) 
        sns.set(font_scale=1)
        sns.heatmap(cm, annot=True,fmt='g',annot_kws={"size": 15},cmap="Blues", square=True, cbar=False)

        # labels, title and ticks
        ax.set_xlabel('Prediction',fontsize=18)
        ax.set_ylabel('Ground Truth',fontsize=18)
        title_name = 'Confusion Matrix - Accuracy ' + str(round(acc,2)) + "%"
        ax.set_title(title_name,fontsize=20) 
        ax.xaxis.set_ticklabels(['Anomaly', 'Normal'],rotation=45)
        ax.yaxis.set_ticklabels(['Anomaly', 'Normal'],rotation=45)

        plt.tight_layout()
        if save is True:
            plt.savefig(title_name+'.jpg', format='jpg', dpi=1000)
        plt.show()