

```python
import numpy as np
import praw
from praw.models import MoreComments
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```


```python
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return(score)

```


```python
reddit = praw.Reddit(client_id='j3uScZz6iIfIRw', \
                     client_secret='8AgY4HS_hW8l759pQvFlttU1A3c', \
                     user_agent='Sentiment', \
                     username='kilejohnson', \
                     password='sxXS!&65ZyZ!&3R9YotH')
```


```python
def comment_analyzer(submission_id):
    submission = reddit.submission(submission_id)
    submission.comments.replace_more(limit=None)
    comment_list = [comment.body for comment in submission.comments.list()]
    comment_sentiment = [sentiment_analyzer_scores(comment)['compound'] for comment in comment_list]
    return np.average(comment_sentiment)
```


```python
print(comment_analyzer('6gkd6v'))
```

    0.20068319088319087
    


```python
subreddit = reddit.subreddit('Ethtrader')
top_subreddit = subreddit.top(limit=1000)
```


```python
topics_dict = { "title":[], 
                "id":[],
                "url":[], 
               "submission_sentiment":[],
                "num_of_comms": [], 
               "comments_sentiment" : [],
                "created": [] 
              }
```


```python
for submission in top_subreddit:
    topics_dict["title"].append(submission.title)
    topics_dict["id"].append(submission.id)
    topics_dict["url"].append(submission.url)
    topics_dict["submission_sentiment"].append(sentiment_analyzer_scores(submission.title)['compound'])
    topics_dict["num_of_comms"].append(submission.num_comments)
    topics_dict['comments_sentiment'].append(comment_analyzer(submission.id))
    topics_dict["created"].append(submission.created)
```


```python
topics_data = pd.DataFrame(topics_dict)
```


```python
def get_date(created):
    return dt.datetime.fromtimestamp(created)
```


```python
_timestamp = topics_data["created"].apply(get_date)
```


```python
topics_data = topics_data.assign(timestamp = _timestamp)
topics_data = topics_data.drop('created', axis=1)
```


```python
reddit_data = topics_data
reddit_data.DateTimeIndex = reddit_data.timestamp
```


```python
reddit_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>id</th>
      <th>url</th>
      <th>submission_sentiment</th>
      <th>num_of_comms</th>
      <th>comments_sentiment</th>
      <th>timestamp</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-24 06:54:20</th>
      <td>Everytime Bitcoin drops</td>
      <td>7lusyi</td>
      <td>https://gfycat.com/defenselessmiserableiberian...</td>
      <td>0.0000</td>
      <td>349</td>
      <td>0.135494</td>
      <td>2017-12-24 06:54:20</td>
    </tr>
    <tr>
      <th>2018-01-16 16:16:23</th>
      <td>Here you go</td>
      <td>7qvmvq</td>
      <td>https://i.redd.it/tpypv1c7xha01.jpg</td>
      <td>0.0000</td>
      <td>311</td>
      <td>0.141956</td>
      <td>2018-01-16 16:16:23</td>
    </tr>
    <tr>
      <th>2017-06-11 04:20:21</th>
      <td>Welcome to r/ethtrader new people, let me save...</td>
      <td>6gkd6v</td>
      <td>http://i.imgur.com/RW0s5gB.gifv</td>
      <td>0.7351</td>
      <td>365</td>
      <td>0.200683</td>
      <td>2017-06-11 04:20:21</td>
    </tr>
    <tr>
      <th>2018-01-16 03:42:16</th>
      <td>Dips are just happy little accidents</td>
      <td>7qr0jq</td>
      <td>https://i.redd.it/m6jwe1ns6ea01.jpg</td>
      <td>0.4005</td>
      <td>212</td>
      <td>0.103242</td>
      <td>2018-01-16 03:42:16</td>
    </tr>
    <tr>
      <th>2018-01-18 12:24:03</th>
      <td>I'm a longterm hodler, but even i hate this su...</td>
      <td>7rba5b</td>
      <td>https://imgur.com/jjCNiyp</td>
      <td>-0.7227</td>
      <td>281</td>
      <td>0.140592</td>
      <td>2018-01-18 12:24:03</td>
    </tr>
  </tbody>
</table>
</div>




```python
reddit_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>submission_sentiment</th>
      <th>num_of_comms</th>
      <th>comments_sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>988.000000</td>
      <td>988.000000</td>
      <td>988.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.075117</td>
      <td>729.711538</td>
      <td>0.159767</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.305618</td>
      <td>1737.215455</td>
      <td>0.090964</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.935300</td>
      <td>9.000000</td>
      <td>-0.167769</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>72.000000</td>
      <td>0.106790</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>124.000000</td>
      <td>0.151311</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.226300</td>
      <td>224.500000</td>
      <td>0.207587</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.915300</td>
      <td>12740.000000</td>
      <td>0.606284</td>
    </tr>
  </tbody>
</table>
</div>




```python
reddit_data.dtypes
```




    title                           object
    id                              object
    url                             object
    submission_sentiment           float64
    num_of_comms                     int64
    comments_sentiment             float64
    timestamp               datetime64[ns]
    dtype: object




```python
fig=plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.scatter(x=reddit_data.index, y = reddit_data['submission_sentiment'], color = '#D52F52', edgecolors='black')
plt.scatter(x=reddit_data.index, y = reddit_data['comments_sentiment'], color = '#3396FF',edgecolors='black')
plt.ylim([-1, 1])
plt.grid(True)
plt.title("Reddit Sentiment Analysis")
plt.xlabel("Reddit Submission")
plt.ylabel("Sentiment")
plt.legend()
plt.show()
```


![png](output_16_0.png)



```python
date_data = reddit_data.groupby(pd.Grouper(freq="W")).mean()
date_date = date_data.dropna()
```


```python
fig=plt.figure(figsize=(14, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.scatter(x=date_data.index, y = date_data['submission_sentiment'], color = '#D52F52', edgecolors='black')
plt.scatter(x=date_data.index, y = date_data['comments_sentiment'], color = '#3396FF',edgecolors='black')
plt.ylim([-1, 1])
plt.grid(True)
plt.title("Reddit Sentiment Analysis")
plt.xlabel("Reddit Submission")
plt.ylabel("Sentiment")
plt.legend()
plt.show()
```


![png](output_18_0.png)



```python
comments = subreddit.comments(limit=10000)
```


```python
comment_dict = { 
                "link_id":[],
                "parent_id":[],
                "url":[], 
                "created": [], 
                "body":[],               
                "neg":[],
                  "neu":[],
                  "pos":[],
                    "compound":[]}
```


```python
for comment in comments:
    comment_dict['link_id'].append(comment.link_id)
    comment_dict['parent_id'].append(comment.parent_id)
    comment_dict['url'].append(comment.link_url)
    comment_dict['created'].append(comment.created)
    comment_dict['body'].append(comment.body)
    comment_dict['neg'].append(sentiment_analyzer_scores(comment.body)['neg'])
    comment_dict['neu'].append(sentiment_analyzer_scores(comment.body)['neu'])
    comment_dict['pos'].append(sentiment_analyzer_scores(comment.body)['pos'])
    comment_dict['compound'].append(sentiment_analyzer_scores(comment.body)['compound'])
```


```python
comment_data = pd.DataFrame(comment_dict)
comment_data = comment_data.assign(timestamp = _timestamp)
comment_data = comment_data.drop('created', axis=1)
comment_data.tail()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>link_id</th>
      <th>parent_id</th>
      <th>url</th>
      <th>body</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>compound</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>983</th>
      <td>t3_axhrgj</td>
      <td>t1_ehvq17r</td>
      <td>https://www.reddit.com/r/ethtrader/comments/ax...</td>
      <td>But Bitcoin is taking a nosedive</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>2018-08-20 07:31:58</td>
    </tr>
    <tr>
      <th>984</th>
      <td>t3_axqm45</td>
      <td>t1_ehve1kq</td>
      <td>https://i.redd.it/7sv49wt4edk21.jpg</td>
      <td>You've never seen a forex chart? Daily range f...</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>2018-04-12 02:12:16</td>
    </tr>
    <tr>
      <th>985</th>
      <td>t3_axqm45</td>
      <td>t3_axqm45</td>
      <td>https://i.redd.it/7sv49wt4edk21.jpg</td>
      <td>Eh. Its pegged to fiat tho, so the value is er...</td>
      <td>0.0</td>
      <td>0.723</td>
      <td>0.277</td>
      <td>0.7703</td>
      <td>2017-11-27 19:48:16</td>
    </tr>
    <tr>
      <th>986</th>
      <td>t3_axqm45</td>
      <td>t1_ehvszns</td>
      <td>https://i.redd.it/7sv49wt4edk21.jpg</td>
      <td>what are the pink blobs?</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>2017-11-17 13:14:21</td>
    </tr>
    <tr>
      <th>987</th>
      <td>t3_axq2ki</td>
      <td>t1_ehvpx69</td>
      <td>https://i.redd.it/2cdqqivl5dk21.jpg</td>
      <td>r/technicallythetruth</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>2017-07-07 16:42:05</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig=plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.scatter(x=comment_data.index, y = comment_data['compound'])
plt.xlim([-2, 1002])
plt.ylim([-1.01, 1.01])
plt.grid(True)
plt.title("Comment Sentiment Analysis")
plt.xlabel("Reddit Submission")
plt.ylabel("Sentiment")
plt.show()
```


![png](output_23_0.png)



```python
def word_cloud(comment):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in comment])
    wordcloud = WordCloud(
        background_color = 'white',
        stopwords = stopwords,
        width = 1600,
        height = 800,
        random_state = 21,
        colormap = 'jet',
        max_words = 50,
        max_font_size=200).generate(all_words)
 
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.show();      
    
```


```python
def word_list(df):
    wrd_list = []
    for row in df.itertuples():
        wrd_list.append(row.body)
    return wrd_list
    
```


```python
reddit_comments = word_list(comment_data)
```


```python
word_cloud(reddit_comments)
```


![png](output_27_0.png)

