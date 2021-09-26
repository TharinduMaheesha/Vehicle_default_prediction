import streamlit as st
import numpy as np
import pandas as pd
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV,cross_val_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier


temp = {"About" : "Something"}
st.set_page_config(page_title="Vehicle default", page_icon="data:image/png;base64,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", layout='wide', initial_sidebar_state='auto', menu_items=temp)

st.image("https://www.sterlingbankasia.com/sites/default/files/2019-11/ultima%20inne-01.png" , width = 500 )


def getData():
    data = pd.read_csv('Data/Train_Dataset.csv')
    return data

def encode_df(df, todummy_list):
    for x in todummy_list:
        df[x] = LabelEncoder().fit_transform(df[x])
    return df

def ModelTraining(data):
    initdf=data.drop(columns=['ID','Score_Source_1','Score_Source_2','Score_Source_3','Credit_Bureau'])
    toconvert_type_list=['Client_Income','Credit_Amount','Loan_Annuity','Population_Region_Relative','Age_Days','Employed_Days','Registration_Days','ID_Days']
    categorical_list = ['Accompany_Client','Client_Income_Type','Client_Education','Client_Marital_Status','Client_Gender','Loan_Contract_Type','Client_Housing_Type','Client_Occupation','Client_Permanent_Match_Tag','Client_Contact_Work_Tag','Type_Organization']
    numeric_list=['Bike_Owned','Active_Loan','House_Own','Child_Count','Own_House_Age','Mobile_Tag','Homephone_Tag','Workphone_Working','Client_Family_Members','Cleint_City_Rating','Application_Process_Day','Application_Process_Hour','Social_Circle_Default','Phone_Change','Default']

    for x in initdf:
        if x in toconvert_type_list:
            initdf[x] = pd.to_numeric(initdf[x],errors = 'coerce')
            numeric_list.append(x)

    categ_dummy_list=[]
    for x in initdf:
        if x in categorical_list:      
            categ_dummy_list.append(x)

    totcount =0
    count=0
    t=0
    for x in initdf:
        for xx in initdf[x]:
            if xx == 'XNA':
                if t==0:
                    t =t+1
                count =count +1   
                totcount =totcount +1

    initdf=initdf.drop(['Type_Organization'],axis=1)
    categ_dummy_list.remove('Type_Organization')
        
    for x in initdf['Client_Gender']:
        if x == 'XNA':
            initdf.drop(initdf[initdf['Client_Gender'] == 'XNA'].index, inplace = True)

    count = 0
    for x in initdf:
        if x in categ_dummy_list:
            count =len(data[x].value_counts())
        count = 0

    initdf['Accompany_Client'] = [x if x in ('Alone','Relative') else 'Other' for x in initdf['Accompany_Client']]
    initdf['Client_Income_Type'] = [x if x in ('Service','Commercial','Retired' , 'Student' , 'Unemployed') else 'Other' for x in initdf['Client_Income_Type']]
    initdf['Client_Education'] = [x if x in ('Secondary','Graduation') else 'Other' for x in initdf['Client_Education']]
    initdf['Client_Housing_Type'] = [x if x =='Home' else 'Other' for x in initdf['Client_Housing_Type']]
    initdf['Client_Marital_Status'] = [x if x =='M' else 'Other' for x in initdf['Client_Marital_Status']]
    initdf['Client_Occupation'] = [x if x in ('Laborers','Sales','Core','Managers','Drivers','High skill tech','Medicine') else 'Other' for x in initdf['Client_Occupation']]
    
    count = 0
    for x in initdf:
        if x in categ_dummy_list:
            count =len(data[x].value_counts())
        count = 0
    
    education = initdf['Client_Education'].unique()
    occupation = initdf['Client_Occupation'].unique()
    income_type = initdf['Client_Income_Type'].unique()

    initdf = encode_df(initdf, categ_dummy_list)

    education_label = initdf['Client_Education'].unique()
    occupation_label = initdf['Client_Occupation'].unique()
    income_type_label = initdf['Client_Income_Type'].unique()

    tot =0
    for x in initdf:
        tot = initdf[x].isnull().sum()
        if (tot/len(initdf.index)) > 0.3 :
            del initdf[x]
        tot=0

    #using Imputer in sklearn.preprocessing, impute missing values 

    imp = SimpleImputer(strategy='mean')
    imp.fit(initdf)
    initdf = pd.DataFrame(data=imp.transform(initdf),columns=initdf.columns)   

    # convert days to years
    days = ['Age_Days','Employed_Days','Registration_Days','ID_Days','Phone_Change']
    for var in days:
        initdf[var] = [math.ceil(x/365) if x != '' else null for x in initdf[var]]

    # Investigate all the variabl's max and min values

    for column in initdf:
        maxVal = initdf[column].max()
        minVal = initdf[column].min()

    #Remove unwanted rows 

    #Population_Region_Relative values must be between 1-0
    val=0
    for col in initdf['Population_Region_Relative']:
        if col>1:
            initdf.drop(initdf[initdf['Population_Region_Relative'] > 1].index, inplace = True)
            val+=1 
    val    
           
    val=0
    for col in initdf['Employed_Days']:
        if col>100:
            #print(col)
            val+=1 
    
    Finaldf=initdf[['Credit_Amount','Client_Income','Loan_Annuity','Age_Days','Phone_Change','ID_Days','Client_Education','Client_Occupation','Client_Gender','Client_Income_Type','Child_Count','Workphone_Working','Application_Process_Hour','Cleint_City_Rating','Homephone_Tag','Car_Owned','Default']]
    NonEncoded = [income_type , education ,   occupation]
    Encoded = [income_type_label , education_label , occupation_label]
    

    X = Finaldf.drop('Default', axis=1).values # Input features (attributes)
    Y = Finaldf['Default'].values # Target vector
    from imblearn.under_sampling import RandomUnderSampler 
    under = RandomUnderSampler()
    X, Y = under.fit_resample(X, Y)
    model=RandomForestClassifier(criterion='gini' , max_depth=20 )
    model.fit(X,Y)

    return model , Encoded , NonEncoded


def labelmaritalStatus(home_phone_tag , phone_working , car , Gender):

    if home_phone_tag == 'Yes':
        home_phone_tag = 1
    elif home_phone_tag == "No":
        home_phone_tag = 0 

    if phone_working == 'Yes':
        phone_working = 1
    elif phone_working == "No":
        phone_working = 0 

    if car == 'Yes':
        car = 1
    elif car == "No":
        car = 0


    if Gender == "Male":
        Gender = 1
    elif Gender == "Female":
        Gender = 0
    
    return home_phone_tag , phone_working ,  car ,  Gender

mainContainer = st.container()

with mainContainer:


    data = getData()
    model , Encoded , NonEncoded = ModelTraining(data)
    
    tab = st.table()

    tab1 = tab.form(key='my_form')
    tab1.header("Enter Following Details to predict Loan Default status")

    col1 , col2 = tab1.columns(2)

    fName = col1.text_input("Client Full Name : ")
    application = col2.text_input("Enter application ID : ")

    education = col1.selectbox("Enter Client Education : " , ('-' ,'Secondary', 'Graduation','Other' ) , on_change=None)
    occupation = col2.selectbox("Select client occupation : " , ('Laborers','Sales','Core','Managers','Drivers','High skill tech','Medicine' , 'Other'))

    income = col1.text_input("Enter Client Income : "  , value = 0 ,on_change=None)
    income_type = col2.selectbox("Enter Income Type : " , ('-' ,'Commercial', 'Service', 'Retired', 'Student', 'Unemployed', 'Other' ) , on_change=None)
                                
    loan_amount = col1.text_input("Enter Loan amount requested : " , on_change=None)
    loan_annuity = col2.text_input("Enter Loan annuity amount : " , on_change=None)

    Age = col1.slider("Enter Age : " , min_value = 20 , max_value= 60 , on_change=None)
    city_rating = col2.slider("Client living city rating : " , min_value = 1 , max_value= 3 , on_change=None)

    Gender = col1.selectbox("Enter Client Gender : " , ("-" , "Male" , "Female") , on_change=None)
    child_count = col2.selectbox("Client child count : " , ("-" ,0,1,2,3,4,5,6,7,8,9,10) , on_change=None)

    home_phone_tag = col1.selectbox("Homephone Number provided by Client ?" , ("-" ,"Yes" , "No") , on_change=None)
    phone_working = col2.selectbox("Mobile Number working ? " ,("-" ,"Yes" , "No") , on_change=None)

    phone_change = col1.slider("Years since phone changed : " , min_value = 0 , max_value= 15 , on_change=None)
    ID_change = col2.slider("Years since NIC issued : " , min_value = 0 , max_value= 60 , on_change=None)

    process_hour = col1.slider("Hour the application was processed : " , min_value = 0 , max_value= 23 , on_change=None)
    car_owned = col2.selectbox("Car owner ?" , ("-" , "Yes" , "No"))


    Submit = tab1.form_submit_button("Submit")
    if Submit:
        tab.empty()


        home_phone_tag , phone_working ,car_owned, Gender = labelmaritalStatus( home_phone_tag ,  phone_working ,  car_owned , Gender)

        count1 = 0
        for i in NonEncoded[0]:
            if income_type == i:
                break
            else:
                count1+=1
        
        income_type = Encoded[0][count1]
    # ==========================================================================================
        count2 = 0
        for i in NonEncoded[1]:
            if education == i:
                break
            else:
                count2+=1
       
        education = Encoded[1][count2]

    # =============================================================================================

        count3 = 0
        for i in NonEncoded[2]:
            if occupation == i:
                break
            else:
                count3+=1
        
        occupation = Encoded[2][count3]

        inputs = [loan_amount , income , loan_annuity , Age , phone_change , ID_change , education , occupation , Gender
                ,income_type , child_count, phone_working , process_hour , city_rating , home_phone_tag , car_owned]

        inputs = [inputs]

        st.write("Client Name : "+ fName)
        st.write("Application ID : "+ application)

        print(inputs)
        prediction  = model.predict(inputs)
        if prediction[0] == 0:
            st.success("The Above inquiry is safe to accept")
        else:
            st.error("The Above inquiry is not safe and has a 65% chance of being default")
            
        
        





    




  





