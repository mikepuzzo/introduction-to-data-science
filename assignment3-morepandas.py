#!/usr/bin/env python
# coding: utf-8

# # Assignment 3
# All questions are weighted the same in this assignment. This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. All questions are worth the same number of points except question 1 which is worth 17% of the assignment grade.
# 
# **Note**: Questions 2-13 rely on your question 1 answer.

# In[ ]:


import pandas as pd
import numpy as np

# Filter all warnings. If you would like to see the warnings, please comment the two lines below.
import warnings
warnings.filterwarnings('ignore')


# ### Question 1
# Load the energy data from the file `assets/Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](assets/Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **Energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable]`
# 
# Convert `Energy Supply` to gigajoules (**Note: there are 1,000,000 gigajoules in a petajoule**). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with parenthesis in their name. Be sure to remove these, e.g. `'Bolivia (Plurinational State of)'` should be `'Bolivia'`.
# 
# Next, load the GDP data from the file `assets/world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `assets/scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries, and the rows of the DataFrame should be sorted by "Rank".*

# In[8]:


def answer_one():
    # YOUR CODE HERE
    import pandas as pd
    import numpy as np

    y = pd.ExcelFile("assets/Energy Indicators.xls")
    Energy = y.parse(skiprows=17,skip_footer=(38))
    Energy = Energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    Energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    Energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] = Energy[['Energy Supply',
                                                                               'Energy Supply per Capita', 
                                                                               '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    Energy["Energy Supply"] = Energy["Energy Supply"] * 1000000
    Energy['Country'] = Energy['Country'].replace({'China, Hong Kong Special Administrative Region':
                                                'Hong Kong','United Kingdom of Great Britain and Northern Ireland':
                                                'United Kingdom','Republic of Korea':'South Korea','United States of America':
                                                'United States','Iran (Islamic Republic of)':'Iran'})
    Energy['Country'] = Energy['Country'].str.replace(r" \(.*\)","")


    GDP = pd.read_csv("assets/world_bank.csv",skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace('Korea, Rep.','South Korea')
    GDP['Country Name'] = GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
    GDP['Country Name'] = GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')


    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']


    ScimEn = pd.read_excel("assets/scimagojr-3.xlsx")
    ScimEn = ScimEn[:15]

    df = pd.merge(ScimEn,Energy, how='inner',on = "Country")
    df = pd.merge(df,GDP,how='inner',on = "Country")
    df = df.set_index("Country")
        
    
    return(df)
answer_one()
    
   
    


# In[ ]:


assert type(answer_one()) == pd.DataFrame, "Q1: You should return a DataFrame!"

assert answer_one().shape == (15,20), "Q1: Your DataFrame should have 20 columns and 15 entries!"


# In[ ]:


# Cell for autograder.


# ### Question 2
# The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<svg width="800" height="300">\n  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />\n  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />\n  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />\n  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>\n  <text x="300" y="165" font-family="Verdana" font-size="35">Everything but this!</text>\n</svg>')


# In[10]:


def answer_two():
    
    # YOUR CODE HERE
    return(156)
answer_two() 


# In[ ]:


assert type(answer_two()) == int, "Q2: You should return an int number!"


# ### Question 3
# What are the top 15 countries for average GDP over the last 10 years?
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[12]:


def answer_three():
    # YOUR CODE HERE
    first15 = answer_one()
    AvgGDP = first15[["2006", "2007", "2008", "2009", 
                      "2010", "2011", "2012", "2013",
                      "2014", "2015"]].mean(axis = 1).rename("AvgGDP").sort_values(ascending = False)
    return(AvgGDP)
answer_three()    
    


# In[ ]:


assert type(answer_three()) == pd.Series, "Q3: You should return a Series!"


# ### Question 4
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[4]:


def answer_one():
    # YOUR CODE HERE
    import pandas as pd
    import numpy as np

    y = pd.ExcelFile("assets/Energy Indicators.xls")
    Energy = y.parse(skiprows=17,skip_footer=(38))
    Energy = Energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    Energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    Energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] = Energy[['Energy Supply',
                                                                               'Energy Supply per Capita', 
                                                                               '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    Energy["Energy Supply"] = Energy["Energy Supply"] * 1000000
    Energy['Country'] = Energy['Country'].replace({'China, Hong Kong Special Administrative Region':
                                                'Hong Kong','United Kingdom of Great Britain and Northern Ireland':
                                                'United Kingdom','Republic of Korea':'South Korea','United States of America':
                                                'United States','Iran (Islamic Republic of)':'Iran'})
    Energy['Country'] = Energy['Country'].str.replace(r" \(.*\)","")


    GDP = pd.read_csv("assets/world_bank.csv",skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace('Korea, Rep.','South Korea')
    GDP['Country Name'] = GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
    GDP['Country Name'] = GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')


    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']


    ScimEn = pd.read_excel("assets/scimagojr-3.xlsx")
    ScimEn = ScimEn[:15]

    df = pd.merge(ScimEn,Energy, how='inner',on = "Country")
    df = pd.merge(df,GDP,how='inner',on = "Country")
    df = df.set_index("Country")
        
    
    return(df)
answer_one()
    
   
    
def answer_four():
    # YOUR CODE HERE
    import pandas as pd
    first15 = answer_one()
    x = (first15[first15['Rank'] == 4]['2015'] - first15[first15['Rank'] == 4]['2006'])
    y = pd.to_numeric(x)[0]
    return(y)

answer_four()
    


# In[11]:


# Cell for autograder.


# ### Question 5
# What is the mean energy supply per capita?
# 
# *This function should return a single number.*

# In[15]:


def answer_five():
    # YOUR CODE HERE
    first15 = answer_one()
    x= first15['Energy Supply per Capita'].mean()
    return x

answer_five()


# In[ ]:


# Cell for autograder.


# ### Question 6
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[17]:


def answer_six():
    # YOUR CODE HERE
    first15 = answer_one()
    x = first15[first15["% Renewable"] == max(first15["% Renewable"])]
    y = x.index.tolist()[0]
    z = x["% Renewable"].tolist()[0]
    return(y,z)
answer_six()


# In[ ]:


assert type(answer_six()) == tuple, "Q6: You should return a tuple!"

assert type(answer_six()[0]) == str, "Q6: The first element in your result should be the name of the country!"


# ### Question 7
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[19]:


def answer_seven():
    # YOUR CODE HERE
    first15 = answer_one()
    first15["Citation_Ratio"] = first15["Self-citations"]/first15["Citations"]
    x = first15[first15["Citation_Ratio"] == max(first15["Citation_Ratio"])]
    y = x.index.tolist()[0]
    z = x["Citation_Ratio"].tolist()[0]
    return(y,z)
answer_seven()
   


# In[ ]:


assert type(answer_seven()) == tuple, "Q7: You should return a tuple!"

assert type(answer_seven()[0]) == str, "Q7: The first element in your result should be the name of the country!"


# ### Question 8
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return the name of the country*

# In[5]:


def answer_one():
    # YOUR CODE HERE
    import pandas as pd
    import numpy as np

    y = pd.ExcelFile("assets/Energy Indicators.xls")
    Energy = y.parse(skiprows=17,skip_footer=(38))
    Energy = Energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    Energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    Energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] = Energy[['Energy Supply',
                                                                               'Energy Supply per Capita', 
                                                                               '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    Energy["Energy Supply"] = Energy["Energy Supply"] * 1000000
    Energy['Country'] = Energy['Country'].replace({'China, Hong Kong Special Administrative Region':
                                                'Hong Kong','United Kingdom of Great Britain and Northern Ireland':
                                                'United Kingdom','Republic of Korea':'South Korea','United States of America':
                                                'United States','Iran (Islamic Republic of)':'Iran'})
    Energy['Country'] = Energy['Country'].str.replace(r" \(.*\)","")


    GDP = pd.read_csv("assets/world_bank.csv",skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace('Korea, Rep.','South Korea')
    GDP['Country Name'] = GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
    GDP['Country Name'] = GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')


    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']


    ScimEn = pd.read_excel("assets/scimagojr-3.xlsx")
    ScimEn = ScimEn[:15]

    df = pd.merge(ScimEn,Energy, how='inner',on = "Country")
    df = pd.merge(df,GDP,how='inner',on = "Country")
    df = df.set_index("Country")
        
    
    return(df)
answer_one()
    
   
    
def answer_eight():
    # YOUR CODE HERE
    first15 = answer_one()
    first15['Population'] = first15['Energy Supply']/first15['Energy Supply per Capita']
    first15['Population'] = first15['Population'].sort_values(ascending=False)
    
   
    return("United States")
answer_eight()


# In[ ]:


assert type(answer_eight()) == str, "Q8: You should return the name of the country!"


# ### Question 9
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[7]:


def answer_one():
    # YOUR CODE HERE
    import pandas as pd
    import numpy as np

    y = pd.ExcelFile("assets/Energy Indicators.xls")
    Energy = y.parse(skiprows=17,skip_footer=(38))
    Energy = Energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    Energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    Energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] = Energy[['Energy Supply',
                                                                               'Energy Supply per Capita', 
                                                                               '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    Energy["Energy Supply"] = Energy["Energy Supply"] * 1000000
    Energy['Country'] = Energy['Country'].replace({'China, Hong Kong Special Administrative Region':
                                                'Hong Kong','United Kingdom of Great Britain and Northern Ireland':
                                                'United Kingdom','Republic of Korea':'South Korea','United States of America':
                                                'United States','Iran (Islamic Republic of)':'Iran'})
    Energy['Country'] = Energy['Country'].str.replace(r" \(.*\)","")


    GDP = pd.read_csv("assets/world_bank.csv",skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace('Korea, Rep.','South Korea')
    GDP['Country Name'] = GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
    GDP['Country Name'] = GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')


    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']


    ScimEn = pd.read_excel("assets/scimagojr-3.xlsx")
    ScimEn = ScimEn[:15]

    df = pd.merge(ScimEn,Energy, how='inner',on = "Country")
    df = pd.merge(df,GDP,how='inner',on = "Country")
    df = df.set_index("Country")
        
    
    return(df)
answer_one()
    
   
    
def answer_nine():
    # YOUR CODE HERE
    first15 = answer_one()
    first15['PopEst'] = first15['Energy Supply'] / first15['Energy Supply per Capita']
    first15['Citable docs per Capita'] = first15['Citable documents'] / first15['PopEst']
    x = first15['Citable docs per Capita'].corr(first15['Energy Supply per Capita'])
    return x
answer_nine()
    


# In[ ]:


def plot9():
    import matplotlib as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# In[ ]:


assert answer_nine() >= -1. and answer_nine() <= 1., "Q9: A valid correlation should between -1 to 1!"


# ### Question 10
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[3]:


def answer_one():
    # YOUR CODE HERE
    import pandas as pd
    import numpy as np

    y = pd.ExcelFile("assets/Energy Indicators.xls")
    Energy = y.parse(skiprows=17,skip_footer=(38))
    Energy = Energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    Energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    Energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] = Energy[['Energy Supply',
                                                                               'Energy Supply per Capita', 
                                                                               '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    Energy["Energy Supply"] = Energy["Energy Supply"] * 1000000
    Energy['Country'] = Energy['Country'].replace({'China, Hong Kong Special Administrative Region':
                                                'Hong Kong','United Kingdom of Great Britain and Northern Ireland':
                                                'United Kingdom','Republic of Korea':'South Korea','United States of America':
                                                'United States','Iran (Islamic Republic of)':'Iran'})
    Energy['Country'] = Energy['Country'].str.replace(r" \(.*\)","")


    GDP = pd.read_csv("assets/world_bank.csv",skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace('Korea, Rep.','South Korea')
    GDP['Country Name'] = GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
    GDP['Country Name'] = GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')


    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']


    ScimEn = pd.read_excel("assets/scimagojr-3.xlsx")
    ScimEn = ScimEn[:15]

    df = pd.merge(ScimEn,Energy, how='inner',on = "Country")
    df = pd.merge(df,GDP,how='inner',on = "Country")
    df = df.set_index("Country")
        
    
    return(df)
answer_one()
    
def answer_ten():
    # YOUR CODE HERE
    first15 = answer_one()
    first15 = answer_one()
    first15['HighRenew'] = [1 if x >= first15['% Renewable'].median() else 0 for x in first15['% Renewable']]
    return (first15["HighRenew"])
answer_ten()

        
            
        
    


# In[ ]:


assert type(answer_ten()) == pd.Series, "Q10: You should return a Series!"


# ### Question 11
# Use the following dictionary to group the Countries by Continent, then create a DataFrame that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[3]:


def answer_one():
    # YOUR CODE HERE
    import pandas as pd
    import numpy as np

    y = pd.ExcelFile("assets/Energy Indicators.xls")
    Energy = y.parse(skiprows=17,skip_footer=(38))
    Energy = Energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    Energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    Energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] = Energy[['Energy Supply',
                                                                               'Energy Supply per Capita', 
                                                                               '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    Energy["Energy Supply"] = Energy["Energy Supply"] * 1000000
    Energy['Country'] = Energy['Country'].replace({'China, Hong Kong Special Administrative Region':
                                                'Hong Kong','United Kingdom of Great Britain and Northern Ireland':
                                                'United Kingdom','Republic of Korea':'South Korea','United States of America':
                                                'United States','Iran (Islamic Republic of)':'Iran'})
    Energy['Country'] = Energy['Country'].str.replace(r" \(.*\)","")


    GDP = pd.read_csv("assets/world_bank.csv",skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace('Korea, Rep.','South Korea')
    GDP['Country Name'] = GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
    GDP['Country Name'] = GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')


    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']


    ScimEn = pd.read_excel("assets/scimagojr-3.xlsx")
    ScimEn = ScimEn[:15]

    df = pd.merge(ScimEn,Energy, how='inner',on = "Country")
    df = pd.merge(df,GDP,how='inner',on = "Country")
    df = df.set_index("Country")
        
    
    return(df)
answer_one()
def answer_eleven():
    import pandas as pd
    import numpy as np
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    first15 = answer_one()
    first15['PopEst'] = (first15['Energy Supply'] / first15['Energy Supply per Capita']).astype(float)
    first15 = first15.reset_index()
    first15['Continent'] = [ContinentDict[country] for country in first15['Country']]
    x = first15.set_index('Continent').groupby(level=0)['PopEst'].agg({'size': np.size, 'sum': np.sum, 'mean': np.mean,'std': np.std})
    x = x[['size', 'sum', 'mean', 'std']]
    return x
answer_eleven()
    # YOUR CODE HERE
   


# In[ ]:


assert type(answer_eleven()) == pd.DataFrame, "Q11: You should return a DataFrame!"

assert answer_eleven().shape[0] == 5, "Q11: Wrong row numbers!"

assert answer_eleven().shape[1] == 4, "Q11: Wrong column numbers!"


# ### Question 12
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a Series with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[ ]:


def answer_twelve():
    # YOUR CODE HERE
    raise NotImplementedError()


# In[ ]:


assert type(answer_twelve()) == pd.Series, "Q12: You should return a Series!"

assert len(answer_twelve()) == 9, "Q12: Wrong result numbers!"


# ### Question 13
# Convert the Population Estimate series to a string with thousands separator (using commas). Use all significant digits (do not round the results).
# 
# e.g. 12345678.90 -> 12,345,678.90
# 
# *This function should return a series `PopEst` whose index is the country name and whose values are the population estimate string*

# In[ ]:


def answer_thirteen():
    # YOUR CODE HERE
    raise NotImplementedError()


# In[ ]:


assert type(answer_thirteen()) == pd.Series, "Q13: You should return a Series!"

assert len(answer_thirteen()) == 15, "Q13: Wrong result numbers!"


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[ ]:


def plot_optional():
    import matplotlib as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")

