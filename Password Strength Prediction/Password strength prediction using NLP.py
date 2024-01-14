'''

Password - 1000k unique values for password collected online

Strength - three values(0 , 1 , 2) i.e. 0 for weak, 1 for medium, 2 for strong..
Strength of the password based on rules(such as containing digits, special symbols , etc.)


The passwords used in our analysis are from 000webhost leak that is available online

'''

## importing all necessary libraries ..

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import string
import warnings
from warnings import filterwarnings
filterwarnings("ignore")

## we have your data into 'password_Data.sqlite' which has table has 'Users'
## now we can read data from this db file 'password_Data.sqlite' using sqlite3 & pandas
# from google.colab import drive
# drive.mount('/content/drive')

# very first u have to create a SQL connection to our SQLite database
con = sqlite3.connect(r"/content/drive/MyDrive/Hands on Python/Data Science Projects/Password Strength Prediction/password_data.sqlite")
#### check online its table name on https://sqliteonline.com/
data = pd.read_sql_query("SELECT * FROM Users" , con)

data.drop(["index"] , axis=1 , inplace=True)
data.isnull().any()

data[data["password"].str.isnumeric()].shape  ### only 26 people have set their password as only number !

data[data["password"].str.isalpha()].shape ### around 50 users have their password as alphabet letters only !

data[data["password"].str.isalnum()] ### most of the users around 97K have their password as alpha-numeric..

data[data["password"].str.istitle()] ### around 932 users have their password having first alphabet capital !


string.punctuation ## all punctuations defined in "string" package !

def find_semantics(row):
    for char in row:
        if char in string.punctuation:
            return 1
        else:
            pass

data["password"].apply(find_semantics)==1
data[data["password"].apply(find_semantics)==1]

## ie , 2663 observations have special characters in between them ..
## 2.6% people password actually uses special character in their password ..

'''

we have password strength so you can do a quick google search to check what features password depends on:-
It depends on 5 factors :

    Length of password
    Frequency of Lowercase Characters
    Frequency of Uppercase Characters
    Frequency of Numeric Characters
    Frequency of Special Characters

These will be the result of the google search to find factors effecting strength of password..


'''

'''

Q..->> why we are diving each value by its Total length or why we are normalizing frequency ?

Ans : Just  to get rid of some outliers bcz some passwords have huge length as we have seen , hence value of lowercase could
also be high , so lets normalise it in the range between 0 to 1


'''

data["length"] = data["password"].str.len()
def freq_lowercase(row):
    return len([char for char in row if char.islower()])/len(row)

def freq_uppercase(row):
    return len([char for char in row if char.isupper()])/len(row)

def freq_numerical_case(row):
    return len([char for char in row if char.isdigit()])/len(row)

data["lowercase_freq"] = np.round(data["password"].apply(freq_lowercase) , 3)

data["uppercase_freq"] = np.round(data["password"].apply(freq_uppercase) , 3)

data["digit_freq"] = np.round(data["password"].apply(freq_numerical_case) , 3)

def freq_special_case(row):
    special_chars = []
    for char in row:
        if not char.isalpha() and not char.isdigit():
            special_chars.append(char)
    return len(special_chars)
data["special_char_freq"] = np.round(data["password"].apply(freq_special_case) , 3) ## applying "freq_special_case" function
data["special_char_freq"] = data["special_char_freq"]/data["length"] ## noromalising "special_char_freq" feature

data[['length' , 'strength']].groupby(['strength']).agg(["min", "max" , "mean" , "median"]).reset_index()
cols = ['length', 'lowercase_freq', 'uppercase_freq',
       'digit_freq', 'special_char_freq']

for col in cols:
    print(col)
    print(data[[col , 'strength']].groupby(['strength']).agg(["min", "max" , "mean" , "median"]))
    print('\n')

'''

Just taking a rough look at the above stats I can say the following:-



->> Higher the length, Higher the strength

->> In case on alphabet frequency higher is not better.
    Probably because it'll not be a strong password if max portion is occupied by just alphabets..
    Password has more strength if the char types are spread in decent proportions.

'''


fig , ((ax1 , ax2) , (ax3 , ax4) , (ax5,ax6)) = plt.subplots(3 , 2 , figsize=(15,7))

sns.boxplot(x="strength" , y='length' , hue="strength" , ax=ax1 , data=data)
sns.boxplot(x="strength" , y='lowercase_freq' , hue="strength" , ax=ax2, data=data)
sns.boxplot(x="strength" , y='uppercase_freq' , hue="strength" , ax=ax3, data=data)
sns.boxplot(x="strength" , y='digit_freq' , hue="strength" , ax=ax4, data=data)
sns.boxplot(x="strength" , y='special_char_freq' , hue="strength" , ax=ax5, data=data)

plt.subplots_adjust(hspace=0.6)

'''
Insights :
Regarding the insights we can say that:-



->> Higher Lowercase frequency is seen in low strength passwords.
    For higher strength passwords ,  Lowercase frequency can be high too but that is probably effect of length.


->> In digit_freq there is a split of majority poplutation of strength 1 and 2
    but for 0 and 1 strength , there is overlap so no too much to say there.
    But we can say a nicely propotioned password is good..


->> In upper_freq , there is a trend if freq is around 0.5 then password is strong

->> Similar but stronger same trend as above in special_freq.

->> Higher strength passwords have more type breaks.


'''

#TF and IDF
#TF - Term Freq
#IDF - Inverse Document Frequency
dataframe = data.sample(frac=1) ### shuffling randomly for robustness of ML moodel
x = list(dataframe["password"])

from sklearn.feature_extraction.text import TfidfVectorizer ## import TF-IDF vectorizer to convert text data into numerical data

#### as password is a series of chars , we have to calculate TF_IDF values of each char
#### Thats why we have to split our password as-->>
#### kzde5577-->> ['k', 'z', 'd', 'e', '5', '5', '7', '7']
### then we can compute Tf-Idf value of each character like Tf-IDF value of k , Tf-IDF value of z

vectorizer = TfidfVectorizer(analyzer="char")
X = vectorizer.fit_transform(x)
X.shape
## ie (100000,1) passwords gets represented using (100000, 99)
## ie each password gets represented using 99 dimensions ..

## note : in your case , this dimension might be changed !

### returns feature/char_of_passwords/columns names

vectorizer.get_feature_names_out()

## ie these are the various chars to which different TF-IDF values are assigned for 100000 passwords ..
df2 = pd.DataFrame(X.toarray() , columns=vectorizer.get_feature_names_out())

df2["length"] = dataframe['length']
df2["lowercase_freq"] = dataframe['lowercase_freq']
y = dataframe["strength"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.20)

from sklearn.linear_model import LogisticRegression
## Apply Multinomial logistic Regression as have data have 3 categories in outcomes

clf = LogisticRegression(multi_class="multinomial")
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test) ## doing prediction on X-Test data
y_pred
from collections import Counter

Counter(y_pred)

password = "jdsddvio&^bhs"
sample_array = np.array([password])
sample_matrix = vectorizer.transform(sample_array)
sample_matrix.toarray()
sample_matrix.toarray().shape

### right now , array dim. is (1,99) so now we need to make it as : (1,101) so that my model will accept it as input..
### ie we need to add (length_of_password) & (total_lowercase_chars) in passsword

password
len(password)

new_matrix = np.append(sample_matrix.toarray() , (9,0.444)).reshape(1,101)
clf.predict(new_matrix)
def predict():
    password = input("Enter a password : ")
    sample_array = np.array([password])
    sample_matrix = vectorizer.transform(sample_array)

    length_pass = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()])/len(password)

    new_matrix2 = np.append(sample_matrix.toarray() , (length_pass , length_normalised_lowercase)).reshape(1,101)
    result = clf.predict(new_matrix2)

    if result == 0 :
        return "Password is weak"
    elif result == 1 :
        return "Password is normal"
    else:
        return "password is strong"
    
predict()

#### check Accuracy of the model using confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix ,  accuracy_score , classification_report
accuracy_score(y_test , y_pred)
confusion_matrix(y_test , y_pred)
print(classification_report(y_test , y_pred))