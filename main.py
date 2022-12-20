from speech_recognition import Recognizer,AudioFile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

#  nltk.download('vader_lexicon')

recognizer = Recognizer() 
with AudioFile('HERE_YOUR_AUDIO_FILE')as audio_file: 
  audio = recognizer.record(audio_file)

text = recognizer.recognize_google(audio)
print(text)

anzlyzar = SentimentIntensityAnalyzer()

text1 = text

if (anzlyzar.polarity_scores(text1)["compound"])>0:
    print("positive Text")

else:
    print("negative text")
anzlyzar.polarity_scores(text1)