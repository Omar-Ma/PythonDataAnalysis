
# coding: utf-8

# In[15]:

import pandas as pd
daily_engagement=pd.read_csv('daily_engagement_full.csv')
len(daily_engagement['acct'].unique())


# In[37]:

#One dimensional data analysis in Pandas and Numpy
#In Pandas the basic one dimensional data structure is called Series, 
#and in Numpy, it's called Array.
#Pandas series have more features, while Numpy arrays are simpler.
#Pandas series are build on top upon Numpy arrays.

#Numpy arrays are very similar to python lists.
#But there are some differences:
# 1. Each element has to be of the same type.
# 2. More convenient functions
# 3. Can be muli-dimention (like list of list in python)
# First 20 countries with employment data
import numpy as np
countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])

# Employment data in 2007 for those 20 countries
employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])
#Find the country with highest employment rate.
def max_employment(countries, employment):
    i=employment.argmax()
    return (countries[i], employment[i])
max_employment(countries, employment)

print countries[:]#print all the elements
print countries.dtype#the output is S22, S means string and 22 mean max length
print employment.dtype#the output is float64
print np.array([0,1,2,3]).dtype#the output is int64
print np.array([1.0,1.5,2.0,2.5]).dtype#the output is float64
print np.array([True,False,True]).dtype#the output is bool
print np.array(['AL','AK','AZ','AR','CA']).dtype#the output is S2

for i in range(len(countries)):
    country = countries[i]
    country_employment=employment[i]
    print 'Country {} has employment {}'.format(country,country_employment)
    #the variables in format() will replace {}s.

#basic stats
print employment.mean()
print employment.std()
print employment.max()
print employment.sum()

#vectorized operations
print np.array([1,2,3])+np.array([4,5,6]) #vecter
print [1,2,3]+[4,5,6] #concatenate
print np.array([1,2,3])*3 #vecter
print [1,2,3]*3 #concatenate

#an example of vectorized operation
countries = np.array([
       'Algeria', 'Argentina', 'Armenia', 'Aruba', 'Austria','Azerbaijan',
       'Bahamas', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Bolivia',
       'Botswana', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi',
       'Cambodia', 'Cameroon', 'Cape Verde'
])

# Female school completion rate in 2007 for those 20 countries
female_completion = np.array([
    97.35583,  104.62379,  103.02998,   95.14321,  103.69019,
    98.49185,  100.88828,   95.43974,   92.11484,   91.54804,
    95.98029,   98.22902,   96.12179,  119.28105,   97.84627,
    29.07386,   38.41644,   90.70509,   51.7478 ,   95.45072
])

# Male school completion rate in 2007 for those 20 countries
male_completion = np.array([
     95.47622,  100.66476,   99.7926 ,   91.48936,  103.22096,
     97.80458,  103.81398,   88.11736,   93.55611,   87.76347,
    102.45714,   98.73953,   92.22388,  115.3892 ,   98.70502,
     37.00692,   45.39401,   91.22084,   62.42028,   90.66958
])

def overall_completion_rate(female_completion, male_completion):
    return (female_completion+male_completion)/2
overall_completion_rate(female_completion, male_completion)

def standardize_data(values):
    return (values-values.mean())/values.std()

a=np.array([1,2,3,4,5])
b=np.array([False, False, True, True, True])
print a[b] #output is [3,4,5]
print a[a>2]

#difference of + and +=
a=np.array([1,2,3,4])
b=a
a=a+np.array([1,1,1,1])
print b #b=[1 2 3 4]
a=np.array([1,2,3,4])
b=a
a+=np.array([1,1,1,1]) #+= operates in-place while + does not
print b #b=[2 3 4 5]|

a=np.array([1,2,3,4,5])
slice=a[:3]
slice[0]=100
print a #[100,2,3,4,5]


# In[84]:

#Pandas Series
#Similar to a Numpy array but with extra functionality
import pandas as pd

countries = ['Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
             'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
             'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
             'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia']

life_expectancy_values = [74.7,  75. ,  83.4,  57.6,  74.6,  75.4,  72.3,  81.5,  80.2,
                          70.3,  72.1,  76.4,  68.1,  75.2,  69.8,  79.4,  70.8,  62.7,
                          67.3,  70.6]

gdp_values = [ 1681.61390973,   2155.48523109,  21495.80508273,    562.98768478,
              13495.1274663 ,   9388.68852258,   1424.19056199,  24765.54890176,
              27036.48733192,   1945.63754911,  21721.61840978,  13373.21993972,
                483.97086804,   9783.98417323,   2253.46411147,  25034.66692293,
               3680.91642923,    366.04496652,   1175.92638695,   1132.21387981]

# Life expectancy and gdp data in 2007 for 20 countries
life_expectancy = pd.Series(life_expectancy_values)
gdp = pd.Series(gdp_values)
def variable_correlation(variable1, variable2):
    both_above=(variable1>variable1.mean() )& (variable2>variable2.mean())
    both_below=(variable1<variable1.mean())&(variable2<variable2.mean())
    is_same_direction =both_above|both_below
    num_same_direction=is_same_direction.sum()
    
    num_different_direction=len(variable1)-num_same_direction
    return(num_same_direction,num_different_direction)
variable_correlation(life_expectancy,gdp)

#The difference between pandas and numpy
a=np.array([1,2,3,4])
s=pd.Series([1,2,3,4])
s.describe() #numpy doesn't have this function

#Pandas have "index"
counntries=np.array(['Albania','Algeria','Andorra','Angola'])
life_expectancy_np=np.array([74.7, 75., 83.4, 57.6])
life_expectancy=pd.Series([74.7,75., 83.4, 57.6],
                          index=['Albania','Algeria','Andorra','Angola'])
print life_expectancy

#Numpy arrays are like souped-up Python lists
#A Oandas series is like a cross between a list and a dictionary
print life_expectancy.loc['Albania']
print life_expectancy.iloc[0]
print life_expectancy['Albania']
print life_expectancy.argmax()
print life_expectancy[life_expectancy.argmax()]

#vectorized operation for pandas series
# 1. the same index are in the same order
s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
s2=pd.Series([10,20,30,40],index=['a','b','c','d'])
print s1
print s2
print s1+s2
# 2. the same index are in the different order
s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
s2=pd.Series([10,20,30,40],index=['b','d','a','c'])
print s1+s2
# 3. different index
s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
s2=pd.Series([10,20,30,40],index=['c','d','e','f'])
print s1+s2#NaN will be generated

#removing and filling missing values
sum_result=s1+s2
sum_result.dropna()
sum_result=s1.add(s2,fill_value=0)
print sum_result

#apply function is very similar to apply functions in R
#states.apply(clean_state) clean_state is a function and states is a pandas series
names = pd.Series([
    'Andre Agassi',
    'Barry Bonds',
    'Christopher Columbus',
    'Daniel Defoe',
    'Emilio Estevez',
    'Fred Flintstone',
    'Greta Garbo',
    'Humbert Humbert',
    'Ivan Ilych',
    'James Joyce',
    'Keira Knightley',
    'Lois Lane',
    'Mike Myers',
    'Nick Nolte',
    'Ozzy Osbourne',
    'Pablo Picasso',
    'Quirinus Quirrell',
    'Rachael Ray',
    'Susan Sarandon',
    'Tina Turner',
    'Ugueth Urbina',
    'Vince Vaughn',
    'Woodrow Wilson',
    'Yoji Yamada',
    'Zinedine Zidane'
])

def reverse_name(name):
    split_name=name.split(" ")
    first_name=split_name[0]
    last_name=split_name[1]
    return last_name+", "+ first_name

def reverse_names(names):
    return names.apply(reverse_name)

print reverse_names(names)

import seaborn as sns

# The following code reads all the Gapminder data into Pandas DataFrames. You'll
# learn about DataFrames next lesson.

employment = pd.read_csv('employment_above_15.csv', index_col='Country')
female_completion = pd.read_csv('female_completion_rate.csv', index_col='Country')
male_completion = pd.read_csv('male_completion_rate.csv', index_col='Country')
life_expectancy = pd.read_csv('life_expectancy.csv', index_col='Country')
gdp = pd.read_csv('gdp_per_capita.csv', index_col='Country')

# The following code creates a Pandas Series for each variable for the United States.
# You can change the string 'United States' to a country of your choice.

employment_us = employment.loc['United States']
female_completion_us = female_completion.loc['United States']
male_completion_us = male_completion.loc['United States']
life_expectancy_us = life_expectancy.loc['United States']
gdp_us = gdp.loc['United States']
get_ipython().magic(u'pylab inline')
employment_us.plot()


# In[119]:

#Two dimensional data analysis in Pandas and Numpy
#In numpy we use 2D array, and in pandas we use data frame.

#1. numpy
#mean and std will operate on entire array
#riders of five stations on ten days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])
def mean_riders_for_max_station(ridership):
    #find the station with the maximum riders on the first day
    max_station=ridership[0,:].argmax()#the position of the maximum value on the first day
    #find the mean riders per day for that station
    mean_for_max=ridership[:,max_station].mean()
    #find the mean ridership overall for comparison
    overall_mean=ridership.mean()
    return (overall_mean, mean_for_max)
print mean_riders_for_max_station(ridership)

#operation along an axis
#in pandas data frame, you can use axis='index' or axis ='columns'

ridership.mean(axis=0)#mean of each column
ridership.mean(axis=1)#mean of each row
a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
print a.sum()
print a.sum(axis=0)
print a.sum(axis=1)

# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

def min_and_max_riders_per_day(ridership):
    '''
First, for each subway station, calculate the
    mean ridership per day. Then, out of all the subway stations, return the
    maximum and minimum of these values. That is, find the maximum
    mean-ridership-per-day and the minimum mean-ridership-per-day for any
    subway station.
    '''
    station_riders=ridership.mean(axis=0)
    
    max_daily_ridership = station_riders.max()     
    min_daily_ridership = station_riders.min  ()   
    
    return (max_daily_ridership, min_daily_ridership)
print min_and_max_riders_per_day(ridership)
ridership.mean(axis=0).max()

#data frame in Pandas
ridership_df = pd.DataFrame(
    data=[[   0,    0,    2,    5,    0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [  95,  229,  255,  496,  201],
          [   2,    0,    1,   27,    0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
)
print ridership_df
print ridership_df.mean()
print ridership_df.values
print ridership_df.values.mean()
print ridership_df.loc['05-01-11']
print ridership_df.iloc[0]
print ridership_df.iloc[0,3]
print ridership_df.loc['05-01-11','R006']
print ridership_df.loc['05-01-11','R006']
print ridership_df['R006']

def mean_riders_for_max_station(ridership):
    '''
    this function finds the station with the maximum riders on the
    first day, then return the mean riders per day for that station. Also
    return the mean ridership overall for comparsion.
    
    This is the same as a previous exercise, but this time the
    input is a Pandas DataFrame rather than a 2D NumPy array.
    '''
    maximum_station_first_day=ridership.iloc[0].argmax()
    overall_mean = ridership.values.mean()
    mean_for_max = ridership[maximum_station_first_day].mean()
    return (overall_mean, mean_for_max)
mean_riders_for_max_station(ridership_df)


# In[124]:

# read csv files into Pandas
subway_df=pd.read_csv('nyc_subway_weather.csv')
subway_df.head()
subway_df.describe()

#shift function
entries_and_exits = pd.DataFrame({
    'ENTRIESn': [3144312, 3144335, 3144353, 3144424, 3144594,
                 3144808, 3144895, 3144905, 3144941, 3145094],
    'EXITSn': [1088151, 1088159, 1088177, 1088231, 1088275,
               1088317, 1088328, 1088331, 1088420, 1088753]
})

entries_and_exits-entries_and_exits.shift(1)#calculate the increase between each row

#apply and applymap
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def convert_grade(grade):
    if grade>=90:
        return 'A'
    elif grade>=80:
        return 'B'
    elif grade>=70:
        return 'C'
    elif grade>=60:
        return 'D'
    else:
        return 'F'

def convert_grades(grades):
    '''
    The conversion rule is:
        90-100 -> A
        80-89  -> B
        70-79  -> C
        60-69  -> D
        0-59   -> F
    '''
    return grades.applymap(convert_grade)
convert_grades(grades_df)

def standardize_column(column):
    return (column-column.mean())/column.std()

def standardize(df):
    return df.apply(standardize_column)

#second largest number
df = pd.DataFrame({
    'a': [4, 5, 3, 1, 2],
    'b': [20, 10, 40, 50, 30],
    'c': [25, 20, 5, 15, 10]
})

def second_largest_in_column(column):
    sorted_column=column.sort_values(ascending=False)
    return sorted_column.iloc[1]
    
def second_largest(df):
    '''
    Fill in this function to return the second-largest value of each 
    column of the input DataFrame.
    '''
    return df.apply(second_largest_in_column)
second_largest(df)


# In[134]:

subway_df=pd.read_csv('nyc_subway_weather.csv')
subway_df.head()

get_ipython().magic(u'pylab inline')
import seaborn as sns
ridership_by_day=subway_df.groupby('day_week').mean()['ENTRIESn_hourly']
ridership_by_day.plot()


# In[136]:

# merger data frames
subway_df = pd.DataFrame({
    'UNIT': ['R003', 'R003', 'R003', 'R003', 'R003', 'R004', 'R004', 'R004',
             'R004', 'R004'],
    'DATEn': ['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
              '05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ENTRIESn': [ 4388333,  4388348,  4389885,  4391507,  4393043, 14656120,
                 14656174, 14660126, 14664247, 14668301],
    'EXITSn': [ 2911002,  2911036,  2912127,  2913223,  2914284, 14451774,
               14451851, 14454734, 14457780, 14460818],
    'latitude': [ 40.689945,  40.689945,  40.689945,  40.689945,  40.689945,
                  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ],
    'longitude': [-73.872564, -73.872564, -73.872564, -73.872564, -73.872564,
                  -73.867135, -73.867135, -73.867135, -73.867135, -73.867135]
})

weather_df = pd.DataFrame({
    'DATEn': ['05-01-11', '05-01-11', '05-02-11', '05-02-11', '05-03-11',
              '05-03-11', '05-04-11', '05-04-11', '05-05-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'latitude': [ 40.689945,  40.69132 ,  40.689945,  40.69132 ,  40.689945,
                  40.69132 ,  40.689945,  40.69132 ,  40.689945,  40.69132 ],
    'longitude': [-73.872564, -73.867135, -73.872564, -73.867135, -73.872564,
                  -73.867135, -73.872564, -73.867135, -73.872564, -73.867135],
    'pressurei': [ 30.24,  30.24,  30.32,  30.32,  30.14,  30.14,  29.98,  29.98,
                   30.01,  30.01],
    'fog': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'rain': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'tempi': [ 52. ,  52. ,  48.9,  48.9,  54. ,  54. ,  57.2,  57.2,  48.9,  48.9],
    'wspdi': [  8.1,   8.1,   6.9,   6.9,   3.5,   3.5,  15. ,  15. ,  15. ,  15. ]
})
subway_df.merge(weather_df,on=['DATEn','hour','latitude','longitude'],how='inner')#how can be inner,outter,left, or right


# In[147]:

subway_df = pd.read_csv('nyc_subway_weather.csv')
data_by_location=subway_df.groupby(['latitude','longitude'],as_index=False).mean()
scaled_entries=(data_by_location['EXITSn_hourly']/data_by_location['EXITSn_hourly'].std())
plt.scatter(data_by_location['latitude'],data_by_location['longitude'],s=scaled_entries)


# In[ ]:



