class Data:
    def __init__(self,df,handle_na=False,train=False):
        self.df = df
        self.handle_na = handle_na
        self.train = train
        
        #filling null_values:
        if handle_na:
            for col in self.df.columns:  #As values are only numeric we only have to handle that. 
                if self.df[col].isnull:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    
        #Applying Min_maxScalerr             
        
    def scale(self):
        if self.train:
            scaled_features = nor_scaler.fit_transform(self.df)
            scaled_features_df = pd.DataFrame(scaled_features ,columns=self.df.columns)
        else:
            scaled_features =nor_scaler.transform(self.df)
            scaled_features_df = pd.DataFrame(scaled_features ,columns=self.df.columns)
        return scaled_features_df

        #we need to create a Machine Learning Algorithm for model prediction .....
class Model(Data):
    def __init__(self,df,target,train=False,handle_na=False):
        super().__init__(df,handle_na=False,train=False)
        self.target = target
    def split(self):
        skf.get_n_splits(self.df,self.target)
        for train_index , test_index in skf.split(self.df,self.target):
            self.X_train = np.array(self.df.iloc[train_index]) 
            self.X_val = np.array(self.df.iloc[test_index])
            self.y_train  = np.array(self.target.iloc[train_index]) 
            self.y_val = np.array(self.target.iloc[test_index])
            return self.X_train,self.X_val,self.y_val,self.y_train
    ##Model creation, trianing and Prediction.
    def class_model(self):
        Ran_for.fit( self.X_train, self.y_train)
        score = Ran_for.score(self.X_val,self.y_val)
        pred = np.where(Ran_for.predict_proba(self.X_val)[:,1]> 0.36 ,1,0).astype(float)
        finite_scores = confusion_matrix(pred,self.y_val)
        return score,pred,finite_scores

    def test_set_pred(self): ###Here Df is X_test:
        final_prediction = Ran_for.predict_proba(self.df)[:,1]
        final_prediction = np.where(final_prediction > 0.36 ,1,0).astype(float)
        submission_prod = pd.DataFrame({'target_class':final_prediction})
        return submission_prod

# ***Main*** #
if __name__ == "__main__":
    #Calling required dependencies & libraries.
    import pandas as pd
    import pickle as pck
    import warnings
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix,roc_auc_score
    
    #filtering out warniings
    warnings.filterwarnings("ignore")  
    #scaler object.For Normalising the feautures.
    nor_scaler = MinMaxScaler()
    #StratifiedKfold object:For splitting the Dataframe.
    skf = StratifiedKFold(n_splits=10 ,random_state=0)
    #Classifier Random Forest:
    Ran_for = RandomForestClassifier(n_estimators=52,criterion='entropy',max_leaf_nodes=200,n_jobs=-1)
    
    #importing datasets for the workflow.
    train_file_path = "E:\All Python\Kaggle_challenges\pulsar_star_dataset\pulsar_data_train.csv"
    test_file_path = "E:\All Python\Kaggle_challenges\pulsar_star_dataset\pulsar_data_test.csv"
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    #selectiong feature dataframe and label dataframe out of train_dataframe
    df_train = train_df.iloc[:,:-1]
    train_target = train_df.target_class
    #selecting only feature dataframe out of test_data_set for final
    #prediction
    df_test = test_df.iloc[:,:-1]
    #Now we can convert this dataset into specific objects of our Class
    
    train_set = Data(df=df_train ,handle_na=True,train=True)
    test_set = Data(df=df_test,handle_na=True)
    scaled_train_data = train_set.scale() #always perform this operation
    scaled_test_data = test_set.scale()   #Then this
    print(scaled_train_data)
    print(scaled_test_data)
    ###
    #StratifiedKFoldsplitting of training data
    
            #Testing set as X_test
    X_test = np.array(scaled_test_data)       
    ###Model Class###
    Train = Model(df=scaled_train_data,target=train_target,train=True,handle_na=False)
    print(Train.split())
    print(Train.class_model()) ###Mandatory:.. Run this training before moving on to the final prediction
   
    Test = Model(df=X_test,target=None,train=False,handle_na=False)
    print(Test.test_set_pred())