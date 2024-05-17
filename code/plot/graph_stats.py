import dateutil.parser
import pytz
import datetime
import pandas as pd
import matplotlib.pyplot as plt

event_time = datetime.datetime.strptime("2021-08-14 12:29:08", "%Y-%m-%d %H:%M:%S")
event_time = pytz.utc.localize(event_time)

death_df = pd.read_csv("output/tweetqa-robertaB-ti3-eb3-yearevent/death_tweets.csv")
injury_df = pd.read_csv("output/tweetqa-robertaB-ti3-eb3-yearevent/injury_tweets.csv")

d_times = []
deaths = []

i_times = []
injuries = []
for i, row in death_df.iterrows():
    tweet_time = dateutil.parser.isoparse(row["time"])
    date_diff = (tweet_time-event_time)
    hours, remainder = divmod(date_diff.seconds, 3600)
    hours += date_diff.days * 24
    minutes, seconds = divmod(remainder, 60)
    hours += minutes / 60
    hours += seconds / 3600

    d_times.append(hours)
    deaths.append(row["deaths"])


for i, row in injury_df.iterrows():
    tweet_time = dateutil.parser.isoparse(row["time"])
    date_diff = (tweet_time-event_time)
    hours, remainder = divmod(date_diff.seconds, 3600)
    hours += date_diff.days * 24
    minutes, seconds = divmod(remainder, 60)
    hours += minutes / 60
    hours += seconds / 3600

    i_times.append(hours)
    injuries.append(row["injuries"])

fig1 = plt.figure()
plt.scatter(d_times, deaths)
plt.title("Deaths over time")
plt.xlabel("Time (hours)")
plt.ylabel("Deaths")

fig2 = plt.figure()
plt.scatter(i_times, injuries)
plt.title("Injuries over time")
plt.xlabel("Time (hours)")
plt.ylabel("Injuries")

plt.show()
