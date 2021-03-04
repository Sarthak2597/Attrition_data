import pickle
import sklearn
import numpy
import pandas
import streamlit as st



st.write("""
# Attrition-Flag Application
""")


from PIL import Image
image = Image.open(r'****Credit_Card.jpg')
st.image(image, use_column_width=True)


st.write("Please provide the inputs accurately for each section.")

st.sidebar.write('Credit card attrition is, in simple terms, a reduction in credit card users for a company. Credit card attrition is enhanced by several factors. Some involve market conditions (changes in the economy) and some involve particulars at a specific company. In any case, reducing attrition and winning back customers requires strong action plans that are marketed specifically toward former customers')

st.sidebar.write('A good research team can be employed to extract the data of already attrited customers in order to study the patter or the likelihood of customers leaving the services on the similar ground.')

st.sidebar.write('This can further be enhanced by deploying a well trained machine learning algorithm (here SVM) to understand the trend and focus on crucial customers that are likely to discontinue.')


a=round(st.number_input("Total no. of products held by the customer"))
b=round(st.number_input("Number of months inactive in last 12 months"))
c=round(st.number_input("Number of times contacted in last 12 months"))
d=round(st.number_input("Total Revolving Balance on the Credit Card"))
e=st.number_input("Change in Transaction Amount (Q4 over Q1)")
f=round(st.number_input("Total Transaction amount in last 12 months"))
g=round(st.number_input("Total Transaction Count (Last 12 months)"))
h=st.number_input("Change in Transaction Count (Q4 over Q1)")
j=st.number_input("Average credit card utilization ratio")


def revolving(x):
    min_d=0.00
    max_d=2517.00
    if x < min_d:
        min_d = x
    if x > max_d:
        max_d = x
    return (x-min_d)/(max_d-min_d)

def chg_trans_amount(x):
    min_e=0.00
    max_e=3.397
    if x < min_e:
        min_e = x
    if x > max_e:
        max_e = x
    return (x-min_e)/(max_e-min_e)

def total_trans_amt(x):
    min_f=510.00
    max_f=18484.00
    if x < min_f:
        min_f = x
    if x > max_f:
        max_f = x
    return (x-min_f)/(max_f-min_f)

def trans_count_12(x):
    min_g=10.00
    max_g=139.00
    if x < min_g:
        min_g = x
    if x > max_g:
        max_g = x
    return (x-min_g)/(max_g-min_g)

def trans_count_chng(x):
    min_h=0.00
    max_h=3.714
    if x < min_h:
        min_h = x
    if x > max_h:
        max_h = x
    return (x-min_h)/(max_h-min_h)



input_lis=[a, b, c, revolving(d),chg_trans_amount(e),total_trans_amt(f),trans_count_12(g),trans_count_chng(h),j]


with open(r"***\SVM_Model.pkl", 'rb') as file:
    model = pickle.load(file)

arr=numpy.array(input_lis)
dff=pandas.DataFrame(arr)
dff=dff.transpose()

out=model.predict(dff)


if st.button('Predict'):
    if out[0] == 1:
        st.header('Possibility: Likely to Discontinue')
    elif out[0] == 0:
        st.header("Possibility: Likely to Stay")
    else:
        pass
else:
    st.write('Click to Predict')


    
st.write("Image Source [link](https://www.timesnownews.com/business-economy/personal-finance/planning-investing/article/lost-your-credit-card-do-this-immediately/620879)")
