#!/usr/bin/env python
# coding: utf-8

# In[2]:


from predict_language import LanguageDetectionResult
string = """Conceptually cream skimming has two basic dimensions - product and geography,
you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him,
One of our number will carry out your instructions minutely,
How do you know ? All this is their information again,
yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they 're getting up in the hundred dollar range,
my walkman broke so i 'm upset now i just have to turn the stereo up real loud,
But a few Christian mosaics survive above the apse is the Virgin with the infant Jesus , with the Archangel Gabriel to the right ( his companion Michael , to the left , has vanished save for a few feathers from his wings ),
( Read for Slate ' s take on Jackson 's findings,
Gays and lesbians  """
l1 = LanguageDetectionResult()
l1.predict_language(string)


# In[1]:


import pandas as pd
from predict_language import LanguageDetectionResult
df_new = pd.read_csv('Language_Test.csv')
print(df_new.columns)
df_new.drop(df_new[df_new.columns[0]])
l1 = LanguageDetectionResult()
predicted_results = l1.test_model(df_new['Text'])
df_new['predicted_language'] = predicted_results


# In[2]:


df_new


# In[79]:


model_accuracy = {}


# In[80]:


custom_model_accuracy = sum((df_new['Language'] == df_new['predicted_language']).to_list())/len(df_new)
model_accuracy['custom_model_accuracy'] = custom_model_accuracy


# ### TextBlob

# In[96]:


from textblob import TextBlob
text = "это компьютерный портал для гиков. It was a beautiful day" 
lang = TextBlob(text)
print(lang.detect_language())


# In[82]:


import pandas as pd
df_new = pd.read_csv('Language_Test.csv')
df['Language'].unique()


# In[83]:


lang_dict = {'en':'English','ml':'Malayalam','hi':'Hindi','ta':'Tamil','pt':'Portugeese','fr':'French','nl':'Dutch','es':'Spanish',
'el':'Greek','ru': 'Russian','da':'Danish','it':'Italian','tr':'Turkish','sv':'Sweedish','ar':'Arabic','de':'German',
'kn':'Kannada'}


# In[97]:


text_list = df_new['Text'].to_list()


text_blob_results = []
for text in text_list:
    lang = TextBlob(text)
    
    text_blob_results.append(lang_dict[lang.detect_language()])
    


# In[98]:


text_blob_results
df_new['text_blob_results'] = text_blob_results


# In[102]:


text_blob_accuracy=sum((df_new['Language'] == df_new['text_blob_results']).to_list())/len(df_new)
model_accuracy['text_blob_accuracy'] = text_blob_accuracy


# ### Langdetect

# In[55]:


from langdetect import DetectorFactory, detect, detect_langs
text = "English is the world wide spoken language"
print(detect(text)) 


# In[89]:


text_list = df_new['Text'].to_list()


lang_detect_results = []
for text in text_list:
    lang = detect(text)
    lang_detect_results.append(lang_dict[lang])

df_new['lang_detect_results'] = lang_detect_results


# In[91]:


lang_detect_accuracy=sum((df_new['Language'] == df_new['lang_detect_results']).to_list())/len(df_new)
model_accuracy['lang_detect_accuracy'] = lang_detect_accuracy


# ### Google Compact Language Detector

# In[92]:


import gcld3
detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, 
                                        max_num_bytes=1000)
result = detector.FindLanguage(text='A english')
print(result)


# In[93]:


text_list = df_new['Text'].to_list()


gcld_results = []
for text in text_list:
    lang = detector.FindLanguage(text=text)
    gcld_results.append(lang_dict[lang.language])

df_new['gcld_results'] = gcld_results


# In[94]:


gcld_accuracy=sum((df_new['Language'] == df_new['gcld_results']).to_list())/len(df_new)
model_accuracy['gcld_accuracy'] = gcld_accuracy


# In[103]:


print(model_accuracy)

