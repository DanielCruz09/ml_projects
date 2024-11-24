I am showcasing my personal machine learning projects in this repository. I have tried to work on a diverse set of problems, ranging from regression analysis to natural language processing (NLP). I have provided a short description of each project below.

Cancer Cell Classification:
---------------------------
This is my first machine learning project that I completed without a guide. This project is short and simple -- classifying cancer cells as benign or malignant, and I wanted to practice the fundamental skills of fitting a machine learning model to some data. This project uses Scikit-Learn's built-in breast cancer dataset. You can learn more about using this dataset here: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html. 

SMS Spam Detection: 
-
I used a guide by GeeksforGeeks to complete this project. I implemented a neural network using TensorFlow's functional API to detect if an SMS message is spam. Although I followed an online guide, there were some issues I encountered while working on this project. Firstly, my version of the dataset included some missing values. I have some experience manipulating datasets in R, but Python was different. At this time, I did not know how to fully use NumPy or Pandas, so I relied a lot on trial and error. 
Eventually, I figured out how to handle missing values, but perhaps, using NumPy could have been more efficient. Another issue I encountered was that the neural network was overfitting the data. The accuracy score of the model was essentially 100%. I added some dropout layers and an L2 Regularizer to reduce the overfitting. In the end, I learned a lot about how to transform datasets in Python and construct effective models.

Vehicle Price Prediction: 
-
This project took me the longest since I worked on it individually. I acquired a lot of experience in using NumPy and Pandas to manipulate datasets. Also, I got a better hang of using Scikit-Learn to preprocess the data before feeding it to the model. Through this project, I learned how to use three essential regression estimators: linear regression, decision tree regression, and neural network regression. 
In retrospect, I encountered several bugs during the preprocessing and model building steps. I used Stack Overflow to fix most of the issues. Whatever I couldn't find an answer to, I tried to find an alternative solution or I used ChatGPT for debugging. After this long process, I managed to create a model with a Root Mean Squared Error of 1,565, which is an excellent score in the context of this dataset.

Email Phishing Detection: 
-
This is my most recent project. The goal of this project is to build a model to recognize phishing emails. I have noticed that I enjoy classification/detection models. However, I have also noticed that NLP projects can be some of the most difficult problems to work on since we are dealing with large texts. In this project, I learned how to convert texts to numerical sequences so that the model can learn meaningful patterns about the text (after all, the model can't derive meaning from words). 
I found TensorFlow's NLP tutorial on YouTube very handy in learning how to train a neural network for sentiment analysis. I achieved an accuracy score of 99%, and while this is impressive, some regularization may be needed.

Next Steps: 
-
I want to acquire more experience in computer vision since I have not had the opportunity to work with images. I plan to work on a computer vision project in the following months.
