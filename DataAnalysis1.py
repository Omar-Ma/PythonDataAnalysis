
# coding: utf-8

# In[105]:

##Two options to read csv data into python as a list of rows
#1. Each row is a list
csv=[['A1','A2','A3'],['B1','B2','B3']]

#2. Each row is a dictionary
csv=[{'name1':'A1','name2':'A2','name3':'A3'},
     {'name1':'B1','name2':'B2','name3':'B3'},]


# In[106]:

import unicodecsv
import os

os.chdir('/Users/yujiema/Desktop/DataAnalysisPython')

#os.getcwd()
#os.listdir('/Users/yujiema/Desktop/DataAnalysisPython')

#enrollments=[]

#the following "with" statement help us shorten the code: we don't need f1.close 
#after extracting the data.
with open('enrollments.csv','rb') as f1:#'rb' means the file would be opened for reading
    reader=unicodecsv.DictReader(f1)#DictReader means each row will be a dictionary
    enrollments=list(reader)
    #the line above is a concise version of the following code:
    #for row in reader:
        #enrollments.append(row)   
#f1.close()
with open('project_submissions.csv','rb')as f1:#'rb' means the file would be opened for reading
    reader=unicodecsv.DictReader(f1)#DictReader means each row will be a dictionary
    project_submissions=list(reader)
with open('daily_engagement.csv','rb')as f1:#'rb' means the file would be opened for reading
    reader=unicodecsv.DictReader(f1)#DictReader means each row will be a dictionary
    daily_engagement=list(reader)


# In[120]:

from datetime import datetime as dt

def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date,"%Y-%m-%d")
    
def parse_maybe_int(i):
    if i == '':
        return None
    else:
        return int(i)


# In[121]:

for enrollment in enrollments:
    enrollment['cancel_date']=parse_date(enrollment['cancel_date'])
    enrollment['days_to_cancel']=parse_maybe_int(enrollment['days_to_cancel'])
    enrollment['is_canceled']=enrollment['is_canceled']=='True'
    enrollment['is_udacity']=enrollment['is_udacity']=='True'
    enrollment['join_date']=parse_date(enrollment['join_date'])
    
for engagement_record in daily_engagement:
    engagement_record['lessons_completed']=int(float(engagement_record['lessons_completed']))
    engagement_record['num_courses_visited']=int(float(engagement_record['num_courses_visited']))
    engagement_record['projects_completed']=int(float(engagement_record['projects_completed']))
    engagement_record['total_minutes_visited']=float(engagement_record['total_minutes_visited'])
    engagement_record['utc_date']=parse_date(engagement_record['utc_date'])
    
for submission in project_submissions:
    submission['completion_date']=parse_date(submission['completion_date'])
    submission['creation_date']=parse_date(submission['creation_date'])


# In[125]:

print len(enrollments)
print len(daily_engagement)
print len(project_submissions)


# In[126]:

for engagement_record in daily_engagement:
    engagement_record['account_key'] = engagement_record.pop('acct')


# In[127]:

def get_unique_students(data):
    unique_students=set()
    for data_point in data:
        unique_students.add(data_point['account_key'])
    return unique_students


# In[85]:

unique_enrolled_students=set()#set are unordered collections of unique elements
for enrollment in enrollments:
    unique_enrolled_students.add(enrollment['account_key'])
len(unique_enrolled_students)


# In[84]:

unique_engagement_students=set()#set are unordered collections of unique elements
for engagement_record in daily_engagement:
    unique_engagement_students.add(engagement_record['account_key'])
len(unique_engagement_students)


# In[91]:

unique_project_submitters = set()
for submission in project_submissions:
    unique_project_submitters.add(submission['account_key'])
len(unique_project_submitters)


# In[128]:

len(get_unique_students(project_submissions))


# In[129]:

for enrollment in enrollments:
    student=enrollment['account_key']
    if student not in unique_engagement_students:
        print enrollment
        break


# In[131]:

num_problem_students=0

for enrollment in enrollments:
    student=enrollment['account_key']
    if (student not in unique_engagement_students and enrollment['join_date'] != enrollment['cancel_date']):
        print enrollment
        num_problem_students += 1

num_problem_students


# In[ ]:



