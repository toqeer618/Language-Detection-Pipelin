from predict_language import LanguageDetectionResult
import pandas as pd
obj = LanguageDetectionResult()
result=obj.predict("other.csv")#Your csv file
lang=list(set(result))
for i in result:
    print("The language is ", result[i])
# for simple string
