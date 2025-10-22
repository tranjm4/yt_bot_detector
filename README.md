# Youtube Bot Detector

![Comic graphic of laptop observed by a magnifying glass. Inside of it is a robot face. A text bubble from the robot asks the question: "How to identify bots in Youtube?"](https://github.com/tranjm4/yt_bot_detector/blob/main/public/img.png?raw=true)

This is a machine learning module used to identify bots in Youtube comments from news outlets.

As the prevalence of AI 'slop' increases, it is important for data scientists and others in the world of data to be able to discern what is real or fake to improve the quality of data for future models. The problem of AI 'slop' is also important in the context of political discourse, as today's increasing polarity of political media incentivizes bad actors to automate divisive rhetoric via bots.

This project primarily focuses on **unsupervised learning models** for **anomaly detection**

***

## Todo:

- Manually review UMAP outputs and clusters for perceived patterns in clustering
- Explore graph features and potential coordinated user behavior

***

## Example outputs

### Oct 20, 2025

![UMAP visualization with colored labels; very stringy clusters, almost entirely labeled as 'normal', with a single cluster on the outer side labeled as 'anomaly'.](https://github.com/tranjm4/yt_bot_detector/blob/main/results/behavioral_model2/umap_if_visualization.png?raw=true)

This is the result of including additional features such as:
- `comment velocity`: the average rate of comments per day since account creation
- `account age`: the age of the account in days
- `emoji ratio`: the percentage of text consisting of emojis
- `caps ratio`: the percentage of words consisting of all capital letters

Using robust scalers (utilized above) to normalize the data also appeared to have more distinct separation. The UMAP below did not use as many robust scalers, which resulted in slightly more anomalies dispersed throughout the UMAP visualization, as seen in the left graph below:

![UMAP visualization with colored labels; very stringy clusters, almost entirely labeled as 'normal', with a single cluster on the outer side labeled as 'anomaly'.](https://github.com/tranjm4/yt_bot_detector/blob/main/results/behavioral_model1/umap_if_visualization.png?raw=true)

### Oct 18, 2025
I fitted an isolation forest on [data/basic_pipeline_v5_0](https://github.com/tranjm4/yt_bot_detector/blob/main/config/pipelines/basic_pipeline.yaml), using its labels to help visualize UMAP. I fitted both models independently to see if they are in agreement with each other. Below is the result:

![UMAP visualization with colored labels; many circular clusters, with one oblong-shaped cluster. The oblong cluster is labeled as an 'anomaly' based on isolation forest predictions. The anomalies appear to be strongly isolated from the normal points, with a few normal points included in the anomaly cluster.](https://github.com/tranjm4/yt_bot_detector/blob/main/results/model5/umap_if_visualization.png?raw=true)

The anomalies detected by the isolation forest (in red) are very clearly isolated in their own cluster by the UMAP model.

Upon comparing summary statistics between the anomalies and the overall data, the following differences is as follows:

- `duplicate_comment_count`, `unique_duplicate_comments`, `reply_count`, and `unique_channel_count` were relatively stronger indicators. The anomalies yielded a mean difference greater than 1-2 standard deviations from the overall mean.

I find this result particularly interesting, though I remain cautious and need to evaluate this model on more configurations of the features. It could be picking up the wrong kinds of patterns in the features.

***

## Data Sourcing

The data is collected from [**Youtube's Data API**](https://developers.google.com/youtube/v3/docs). It provides a very generous and well-documented API that allows collection of data regarding videos, comment threads, and individual user channel statistics.

This project focuses on the following objective: detecting anomalous behavior from individual channels within a focused community (political content). The data collection is driven by the following main questions:

- Can we identify abnormal commenting patterns from users?
- Can we identify coordinated efforts from subsets or communities of users?
- What features would be insightful for finding unusual behavior?

Thus, the data collected revolves around finding per-user behavior, described by the following relational database schema:

<u>**Versions**</u> (for data lineage)
| Fields | type |
| :- | :-: |
| versionName | str |
| createdAt | datetime |
| versionDescription | str |

<u>**Main Video Channels**</u>
| Fields | type |
| :------- | :-: |
| channelId | int |
| channelName | str |

<u>**Users**</u>
| Fields | type |
| :- | :-: |
| userId | str |
| username | str |
| createDate | datetime |
| subCount | int |
| videoCount | int |
| versionName -> **Versions** | str |
| updatedAt | datetime |

<u>**Videos**</u>
| Fields | type |
| :- | :-: |
| videoId | str |
| title | str |
| publishDate | datetime |
| channelId -> **Channels** | str |
| versionName -> **Versions** | str |
| updatedAt | datetime |

<u>**Comments**</u>
| Fields | type |
| :- | :-: |
| commentId | str |
| commenterId -> **Users** | str |
| videoId -> **Videos** | str |
| isReply | bool |
| threadId -> **Comments** | str |
| publishDate | datetime |
| editDate | datetime |
| likeCount | int |
| commentText | str |
| versionName -> **Versions** | str |
| updatedAt | datetime |


***

## Personal Challenges

As a learning programmer, data scientist, and machine learning engineer, here are some challenges and core questions I've faced (previously and currently) throughout my experiences working on this project:

### a. Unsupervised Learning

At the beginning of the project, I was heavily considering whether to approach my project as a ***semi-supervised learning problem*** (where I would manually label a small set of comments) or ***unsupervised learning problem***. The main reason against semi-supervised learning was that I would instill my own biases on what I perceive to be bot comments. Thus, unsupervised learning became my main objective.

### b. API Rate Limiting

Although the Youtube API offers a very generous 10,000 API units per day (e.g., 1 request = 1 unit), it still posed problems when trying to collect a sufficient amount of data.

The primary bottleneck I faced was having to collect individual user data (i.e., account creation date, video/sub counts), which originally required an individual API call per user. This was not only expensive in terms of API units, but also in terms of I/O costs with the API.

<u>**Solution**</u>: I was able to optimize my API collection modules to batch process users; the API allowed batch requesting up to 50 users per request. This reduced API costs by 50x. I was able to collect over 150k comments under the daily limits.

### c. Feature Engineering

Upon some initial exploratory data analysis, I found some interesting features. One example was **comment latency**, i.e., the delay from video upload to comment post.

Taking the standard deviation / variance of comment latency for each user and sorting in ascending order, we find some users who have commented multiple times with very low standard deviation in comment latency (some with 0). This is highly unusual behavior.

Normally, I would have thrown the latency into a standard normalization scaler and hoped the model learns the feature, but since we're working in an unsupervised learning context, I had to be more careful with handling the data. I wanted to scale this feature in a way that:

1. Isolates the std deviation when values are low
2. Differentiate values with low vs high sample size

Thus, I propose the following heuristic:

$$\frac{C_1log_b(C_2n)}{\sigma + \epsilon}$$

- $n$: sample size, i.e., comment count of a user
- $\sigma$: the standard deviation of comment latency
- $\epsilon$: a small constant epsilon used to prevent division by zero errors
- $b, C_1, C_2$: constants for modification

Further experimentation on the choice of constants is needed to observe effects on model performance.

**Oct 16, 2025**: Using $\epsilon = 10$ helped sufficiently dampen values to an acceptable scale. Otherwise, with small $\epsilon$, we see high values even after normalization
