class Data:
    def __init__(self,df,handle_na=False,train=False):
        self.df = df
        self.handle_na = handle_na
        self.train = train
        
        #filling null_values:
        if handle_na:
            for col in self.df.columns:  #As values are only numeric we can only take consider of that. 
                if self.df[col].isnull:
                    self.df[col] = self.df[col].fillna(df[col].median())
                    
        #Applying Min_maxScalerr             
        
    def scale(self):
        if self.train:
            scaled_features = nor_scaler.fit_transform(self.df)
        else:
            scaled_features =nor_scaler.transform(self.df)
      
        return scaled_features

        #we need to create a Machine Learning Algorithm for model prediction .....
class Model:
    pass
      




if __name__ == "__main__":
    
    #calling required libraries
    import pandas as pd
    import pickle as pck
    import warnings
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    #filtering out warniings
    warnings.filterwarnings("ignore")
    
    #scaler object.
    nor_scaler = MinMaxScaler()
    
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
    #Now we can convert this datset into specific objects of our Class
    
    train_set = Data(df=df_train ,handle_na=True,train=True)
    test_set = Data(df=df_test,handle_na=True)
    #test_set = Data(df=df_test ,handle_na=True)
    scaled_train_data = train_set.scale() #always perform thios operation
    scaled_test_data = test_set.scale()   #Then this
    print(scaled_train_data,'\n\n')
    print(scaled_test_data)