import numpy as np
import pandas as pd
import math
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import math

from streamlit.state.session_state import Value
temp = {"About" : "Something"}
# st.set_page_config(page_title="Vehicle default", page_icon="https://www.sterlingbankasia.com/sites/default/files/2019-11/ultima%20inne-01.png", layout='wide', initial_sidebar_state='auto', menu_items=temp)

st.image("https://www.sterlingbankasia.com/sites/default/files/2019-11/ultima%20inne-01.png" , width = 500 , )


def getData():
    data =pd.read_csv("Data/Train_Dataset_1.csv") 
    return data

def preProcess(data):
    data = data[['Client_Income' , 'Active_Loan' , 'House_Own' , 'Child_Count' , 'Credit_Amount' , "Client_Income_Type" 
, "Client_Education" , 'Client_Marital_Status' , 'Client_Gender' , 'Loan_Contract_Type' , 'Age_Years' , 'Employed_Years'
 , 'Mobile_Tag' , 'Homephone_Tag' , 'Workphone_Working'  , 'Type_Organization' ,  'Default']]

    income_type = data['Client_Income_Type'].unique()
    education= data['Client_Education'].unique()
    organization= data['Type_Organization'].unique()

    day2Year = ['Age_Years','Employed_Years']
    for va1 in day2Year:
        data[va1] = data[va1].fillna(0)
        data[va1] = [math.ceil(x/365) if x != ' ' else null for x in data[va1]]

    data['Client_Income'] = data['Client_Income'].fillna(0)
    data['Active_Loan'] = data['Active_Loan'].fillna(-1)
    data['House_Own'] = data['House_Own'].fillna(-1)
    data['Child_Count'] = data['Child_Count'].fillna(-1)

    data['Client_Income_Type'] = data['Client_Income_Type'].fillna("Not mentioned")
    data['Client_Education'] = data['Client_Education'].fillna("Not mentioned")
    data['Client_Marital_Status'] = data['Client_Marital_Status'].fillna("Not mentioned")
    data['Client_Gender'] = data['Client_Gender'].fillna("Not mentioned")
    data['Credit_Amount'] = data['Credit_Amount'].fillna(-1)
    data['Type_Organization'] = data['Type_Organization'].fillna("Not mentioned")

    

    data = data.dropna()

    for x in data['Client_Gender']:
        if x == 'XNA':
            data.drop(data[data['Client_Gender'] == 'XNA'].index, inplace = True)

    le=LabelEncoder()
    data['Client_Income_Type']=le.fit(data['Client_Income_Type']).transform(data['Client_Income_Type'])
    data['Client_Education']=le.fit(data['Client_Education']).transform(data['Client_Education'])
    data['Client_Gender']=le.fit(data['Client_Gender']).transform(data['Client_Gender'])
    data['Type_Organization']=le.fit(data['Type_Organization']).transform(data['Type_Organization'])
    data['Loan_Contract_Type']=le.fit(data['Loan_Contract_Type']).transform(data['Loan_Contract_Type'])
    data['Client_Marital_Status']=le.fit(data['Client_Marital_Status']).transform(data['Client_Marital_Status'])

    income_type_label = data['Client_Income_Type'].unique()
    education_label= data['Client_Education'].unique()
    organization_label= data['Type_Organization'].unique()

    NonEncoded = [income_type , education ,   organization]
    Encoded = [income_type_label , education_label , organization_label]


    return data , NonEncoded , Encoded

def ModelTraining(data):

    X = data.drop('Default', axis=1).values# Input features (attributes)
    Y = data['Default'].values # Target vector
    from imblearn.under_sampling import RandomUnderSampler 
    under = RandomUnderSampler()
    X, Y = under.fit_resample(X, Y)
    model=RandomForestClassifier(criterion='gini' , max_depth=20 )
    model.fit(X,Y)

    return model

def labelmaritalStatus(marital_status , home_phone_tag , mobile_tag , phone_working , loan_type , house , active_loans , Gender):

    if marital_status == 'Married':
        marital_status = 1
    elif marital_status == 'Single':
        marital_status = 3
    elif marital_status == 'Divorced':
        marital_status = 0  
    elif marital_status == 'Widow':
        marital_status = 4
    elif marital_status == 'Not Mentioned':
        marital_status = 2


    if home_phone_tag == 'Yes':
        home_phone_tag = 1
    elif home_phone_tag == "No":
        home_phone_tag = 0 

    if mobile_tag == 'Yes':
        mobile_tag = 1
    elif mobile_tag == "No":
        mobile_tag = 0 

    if phone_working == 'Yes':
        phone_working = 1
    elif phone_working == "No":
        phone_working = 0 

    if loan_type == 'Cash Loan':
        loan_type = 0
    elif loan_type == "Revolving Loan":
        loan_type = 1

    if house == 'Yes':
        house = 1
    elif house == "No":
        house = 0
    elif house == 'Not Mentioned':
        house = -1

    if active_loans == 'Yes':
        active_loans = 1
    elif active_loans == "No":
        active_loans = 0
    elif active_loans == 'Not Mentioned':
        active_loans = -1

    if Gender == "Male":
        Gender = 1
    elif Gender == "Female":
        Gender = 0
    
    return marital_status , home_phone_tag , mobile_tag , phone_working , loan_type , house , active_loans , Gender

mainContainer = st.container()

with mainContainer:


    data = getData()
    data , NonEncoded , Encoded = preProcess(data)
    model = ModelTraining(data)
    
    tab = st.table()

    tab1 = tab.form(key='my_form')
    tab1.header("Enter Following Details to predict Loan Default status")

    col1 , col2 = tab1.columns(2)
    fName = col1.text_input("Client Full Name : ")
    application = col2.text_input("Enter application ID : ")

    Age = col1.slider("Enter Age : " , min_value = 20 , max_value= 60 , on_change=None)
    employed_years = col2.slider("Years before the application, the client started earning : " , min_value = 0 , max_value= 50 , on_change=None)



    Gender = col1.selectbox("Enter Client Gender : " , ("-" , "Male" , "Female") , on_change=None)
    marital_status = col2.selectbox("Client marital status : " , ("-" , "Not Mentioned","Married" , "Widow"  , 'Single'  , 'Divorced') , on_change=None)


    income = col1.text_input("Enter Client Income : "  , value = 0 ,on_change=None)
    income_type = col2.selectbox("Enter Income Type : " , ('-' , 'Not mentioned' , 'Commercial', 'Service', 'Retired', 'Govt Job',
                                                            'Student', 'Unemployed', 'Maternity leave', 'Businessman' ) , on_change=None)
    
    home_phone_tag = col1.selectbox("Homephone Number provided by Client ?" , ("-" ,"Yes" , "No") , on_change=None)
    mobile_tag = col2.selectbox("Mobile Number provided by Client ? " ,("-" ,"Yes" , "No") , on_change=None)

    phone_working = col1.selectbox("Mobile Number working ? " ,("-" ,"Yes" , "No") , on_change=None)
    loan_amount = col2.text_input("Enter Loan amount requested : " , on_change=None)

    
    
    loan_type = col1.selectbox("Client Loan Type  : " , ("-" , "Cash Loan" , "Revolving Loan") , on_change=None)
    Occupation_type = col2.selectbox("Select occupation type : " ,("-" , "Not Mentioned" , 'Self-employed', 'Government','Business Entity Type 3',
                                                                    'Other', 'Industry: type 3', 'Business Entity Type 2',
                                                                    'Business Entity Type 1', 'Transport: type 4', 'Construction',
                                                                    'Kindergarten', 'Trade: type 3', 'Industry: type 2',
                                                                    'Trade: type 7', 'Trade: type 2', 'Agriculture', 'Military',
                                                                    'Medicine', 'Housing', 'Industry: type 1', 'Industry: type 11',
                                                                    'Bank', 'School', 'Industry: type 9', 'Postal', 'University',
                                                                    'Transport: type 2', 'Restaurant', 'Electricity', 'Police',
                                                                    'Industry: type 4', 'Security Ministries', 'Services',
                                                                    'Transport: type 3', 'Mobile', 'Hotel', 'Security',
                                                                    'Industry: type 7', 'Advertising', 'Cleaning', 'Realtor',
                                                                    'Trade: type 6', 'Culture', 'Industry: type 5', 'Telecom',
                                                                    'Trade: type 1', 'Industry: type 12', 'Industry: type 8',
                                                                    'Insurance', 'Emergency', 'Legal Services', 'Industry: type 10',
                                                                    'Trade: type 4', 'Industry: type 6', 'Transport: type 1',
                                                                    'Industry: type 13', 'Religion', 'Trade: type 5') , on_change=None)
        

    education = col1.selectbox("Enter Client Education : " , ('-' , 'Not mentioned' , 'Secondary', 'Graduation', 'Graduation dropout',
                                                            'Junior secondary','Post Grad' ) , on_change=None)
    house = col2.selectbox("Client Owns a House? : " , ("-" , "Yes" , "No"  , "Not Mentioned") , on_change=None)

    active_loans = col1.selectbox("Any other active loans ? " , ("-" , "Not Mentioned" , "Yes" , "No") , on_change=None)
    child_count = col2.selectbox("Client child count : " , ("-" ,"Not Mentioned" ,0,1,2,3,4,5,6,7,8,9,10) , on_change=None)


    Submit = tab1.form_submit_button("Submit")
    if Submit:
        tab.empty()

        if child_count == 'Not Mentioned':
            child_count = -1

        marital_status , home_phone_tag , mobile_tag , phone_working , loan_type , house , active_loans , Gender = labelmaritalStatus(marital_status , home_phone_tag , mobile_tag , phone_working , loan_type , house , active_loans , Gender)

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
        print(NonEncoded[2])
        for i in NonEncoded[2]:
            if Occupation_type == i:
                break
            else:
                count3+=1
        
        Occupation_type = Encoded[2][count3]

        vals = [income , active_loans , house , child_count , loan_amount , income_type , education , marital_status , 
        Gender ,loan_type ,Age , employed_years , mobile_tag , home_phone_tag ,phone_working , Occupation_type,]

        vals = [vals]
        print(vals)

        st.write("Client Name : "+ fName)
        st.write("Application ID : "+ application)


        prediction  = model.predict(vals)
        if prediction[0] == 0:
            st.success("The Above inquiry is safe to accept")
        else:
            st.error("The Above inquiry is not safe and has a 65% chance of being default")
            
        
        





    




  





