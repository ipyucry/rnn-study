# **Small NLP Tasks: Last Chance for RNNs?**  

This repository contains the code and datasets used for the research project comparing **BiGRU** and **DistilBERT** on sentiment analysis tasks using the **IMDb** and **Sentiment140** datasets. The study evaluates the performance, efficiency, and practical feasibility of RNNs and Transformers under computational constraints.  

---

## **Project Structure**  

📂 **data/** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Preprocessed tokenized datasets  
&nbsp;&nbsp;&nbsp;&nbsp;──📂 **imdb/** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # IMDb tokenized data  
&nbsp;&nbsp;&nbsp;&nbsp;──📂 **sentiment140/** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Sentiment140 tokenized data  
📂 **models/** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Saved trained models  
📂 **results/** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Training logs and performance metrics  
📂 **scripts/** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Core scripts  
&nbsp;&nbsp;&nbsp;&nbsp;──📄 **imdb_preprocess.py** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Loads and preprocesses IMDb dataset  
&nbsp;&nbsp;&nbsp;&nbsp;──📄 **s140_preprocess.py** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Loads and preprocesses Sentiment140 dataset  
&nbsp;&nbsp;&nbsp;&nbsp;──📄 **check_imdb_preprocess.py** &nbsp;&nbsp; # Checks validity of IMDb tokenized files  
&nbsp;&nbsp;&nbsp;&nbsp;──📄 **check_s140_preprocess.py** &nbsp;&nbsp; # Checks validity of Sentiment140 tokenized files  
&nbsp;&nbsp;&nbsp;&nbsp;──📄 **train_eval_imdb_BiGRU.py** &nbsp;&nbsp;&nbsp;&nbsp; # Trains & evaluates BiGRU on IMDb  
&nbsp;&nbsp;&nbsp;&nbsp;──📄 **train_eval_imdb_DistilBERT.py** &nbsp; # Trains & evaluates DistilBERT on IMDb  
&nbsp;&nbsp;&nbsp;&nbsp;──📄 **train_eval_s140_BiGRU.py** &nbsp;&nbsp;&nbsp; # Trains & evaluates BiGRU on Sentiment140  
&nbsp;&nbsp;&nbsp;&nbsp;──📄 **train_eval_s140_DistilBERT.py** &nbsp; # Trains & evaluates DistilBERT on Sentiment140  
📄 **requirements.txt** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Python dependencies  

---
