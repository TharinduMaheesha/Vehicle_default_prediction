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
st.set_page_config(page_title="Vehicle default", page_icon="https://www.sterlingbankasia.com/sites/default/files/2019-11/ultima%20inne-01.png", layout='wide', initial_sidebar_state='auto', menu_items=temp)

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
            
        
        





    




  





