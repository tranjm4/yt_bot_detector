# Youtube Bot Detector

[](https://github.com/tranjm4/yt_bot_detector/blob/main/public/img.png)

This is a machine learning module used to identify bots in Youtube comments from news outlets.

As the prevalence of AI 'slop' increases, it is important for data scientists and others in the world of data to be able to discern what is real or fake to improve the quality of data for future models.

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

### 1. Unsupervised Learning

At the beginning of the project, I was heavily considering whether to approach my project as a ***semi-supervised learning problem*** (where I would manually label a small set of comments) or ***unsupervised learning problem***. The main reason against semi-supervised learning was that I would instill my own biases on what I perceive to be bot comments. 

Thus, unsupervised learning was the main 

### 2. API Rate Limiting

Although the Youtube API offers a very generous 10,000 API units per day (e.g., 1 request = 1 unit), it still posed problems when trying to collect a sufficient amount of data.

The primary bottleneck I faced was having to collect individual user data (i.e., account creation date, video/sub counts), which originally required an individual API call per user. This was not only expensive in terms of API units, but also in terms of I/O costs with the API.

<u>**Solution**</u>: I was able to optimize my API collection modules to batch process users; the API allowed batch requesting up to 50 users per request. This reduced API costs by 50x. I was able to collect over 150k comments under the daily limits.

### 3. Feature Engineering

Upon 

***

## Setup