import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from transformers import pipeline
from word2number import w2n
import os
import dateutil.parser

out_dir = "philippines22/tweetqa-robertaB-ti3-eb3-noyearevent"
data_dir = "data/us_tweets_philippines_12230727.csv"
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
type_classifier = AutoModelForSequenceClassification.from_pretrained("/home/james/Code/Simons22/models/type_imbalanced").eval()
event_classifier = AutoModelForSequenceClassification.from_pretrained("/home/james/Code/Simons22/models/event_balanced").eval()

qna_model = "saburbutt/roberta_base_tweetqa_model"
qna_pipeline = pipeline('question-answering', model=qna_model, tokenizer=qna_model)

event_year = 2021

def value_to_int(x):
    debris = [",", ".", "-", "!", "?", "ppl", "people", "dead", "deaths"]
    
    for item in debris:
        x = x.replace(item, "")

    try:
        val = int(x)
        return val
    except ValueError:
        y = x
        x = x.replace(" ", "")

        word_l = len(x)
        for i in range(word_l):
            try:
                int(x[i])
            except ValueError:
                if ('K' == x[i]) and (i==word_l-1):
                    return int(x.replace('K', '')) * 1000
                if ('k'==x[i]) and (i==word_l-1):
                    return int(x.replace('k', '')) * 1000
                if ('thousand' in x) and not ('thousands' in x) and ('t'==x[i]):
                    return int(x.replace('thousand', '')) * 1000
                if ('M'==x[i]) and (i==word_l-1):
                    return int(x.replace('M', '')) * 1000000
                if ('million' in x) and not ('millions' in x) and ('m'==x[i]):
                    return int(x.replace('million', '')) * 1000000
                if ('m'==x[i]) and (i==word_l-1):
                    return int(x.replace('b', '')) * 1000000
                if ('B'==x[i]) and (i==word_l-1):
                    return int(x.replace('B', '')) * 1000000000
                if ('b'==x[i]) and (i==word_l-1):
                    return int(x.replace('b', '')) * 1000000000
                if ('billion' in x) and not ('billions' in x) and ('b'==x[i]):
                    return int(x.replace('billion', '')) * 1000000000
                break
        
        try:
            return w2n.word_to_num(y)
        except:
            return -1
    return -1

def probe_info(text):
    answers = qna_pipeline(question=["How many dead?", "How many injured?", "Where?", "What year?", "What disaster?"], context=text)
    # answers = qna_pipeline(question=["How many are dead?", "How many are injured?", "Where"], context=text)

    deaths = value_to_int(answers[0]["answer"])
    injuries = value_to_int(answers[1]["answer"])
    location = answers[2]["answer"]

    info = {"deaths":[deaths, answers[0]["score"]], "injuries":[injuries, answers[1]["score"]], "location":location}

    if (len(answers[3]["answer"])==4) and (answers[3]["answer"] != event_year):
       return None
    elif ("quake" in text) or ("Quake" in text) or ("quake" in answers[4]["answer"]) or ("Quake" in answers[4]["answer"]):
        return info
    else:
        return None
    
    return info
    # question1 = "How many died?"
    # question2 = "How many injured?"
    # question3 = "Where is this?"

    # inputs1 = qna_tokenizer(question1, text, return_tensors="pt")
    # inputs2 = qna_tokenizer(question2, text, return_tensors="pt")
    # inputs3 = qna_tokenizer(question3, text, return_tensors="pt")

    # with torch.no_grad():
    #     outputs1 = qna_model(**inputs1)
    #     outputs2 = qna_model(**inputs2)
    #     outputs3 = qna_model(**inputs3)
    
    # start_index1 = outputs1.start_logits.argmax()
    # end_index1 = outputs1.end_logits.argmax()
    # start_index2 = outputs2.start_logits.argmax()
    # end_index2 = outputs2.end_logits.argmax()
    # start_index3 = outputs3.start_logits.argmax()
    # end_index3 = outputs3.end_logits.argmax()
    
    # answer1 = qna_tokenizer.decode(inputs1.input_ids[0, start_index1 : end_index1 + 1])
    # answer2 = qna_tokenizer.decode(inputs2.input_ids[0, start_index2 : end_index2 + 1])
    # answer3 = qna_tokenizer.decode(inputs3.input_ids[0, start_index3 : end_index3 + 1])
    # print(answer1)
    # print(answer2)
    # print(answer3)   

def filter_type(text):
    inputs = tokenizer.encode(text, return_tensors='pt')
    output = type_classifier(inputs)
    c = torch.argmax(output.logits, dim=-1).item()
    return c==0

def filter_event(text):
    inputs = tokenizer.encode(text, return_tensors='pt')
    output = event_classifier(inputs)
    c = torch.argmax(output.logits, dim=-1).item()
    return c==0

def process_tweet(text):
    is_eq = filter_event(text)
    is_stat = filter_type(text)
    
    if is_eq and is_stat:
        info = probe_info(text)
        return info
    else:
        return None

def main():
    df = pd.read_csv(data_dir)
    df = df[df["lang"]=="en"]

    filtered_t = {"time":[],"text":[]}
    death_t = {"time":[], "text":[], "deaths":[], "death_score":[], "injuries":[], "injury_score":[], "location":[], "tweet_loc":[]}
    injury_t = {"time":[], "text":[], "deaths":[], "death_score":[], "injuries":[], "injury_score":[], "location":[], "tweet_loc":[]}
    print("*"*50)
    for i, tweet in df.iterrows():
        if i % 500 == 0:
            print("%d / %d" %(i, df.shape[0]))

        info = process_tweet(tweet["text"])
        if info is None:
            filtered_t["time"].append(tweet["created_at"])
            filtered_t["text"].append(tweet["text"])
        elif (info["deaths"][0]==-1) and (info["injuries"][0]==-1):
            filtered_t["time"].append(tweet["created_at"])
            filtered_t["text"].append(tweet["text"])
        else:
            if info["deaths"][0] != -1:
                death_t["time"].append(tweet["created_at"])
                death_t["text"].append(tweet["text"])
                death_t["deaths"].append(info["deaths"][0])
                death_t["death_score"].append(info["deaths"][1])
                death_t["injuries"].append(info["injuries"][0])
                death_t["injury_score"].append(info["injuries"][1])
                death_t["location"].append(info["location"])
                death_t["tweet_loc"].append(tweet["geo.geo.bbox"])

                
            if info["injuries"][0] != -1:
                injury_t["time"].append(tweet["created_at"])
                injury_t["text"].append(tweet["text"])
                injury_t["deaths"].append(info["deaths"][0])
                injury_t["death_score"].append(info["deaths"][1])
                injury_t["injuries"].append(info["injuries"][0])
                injury_t["injury_score"].append(info["injuries"][1])
                injury_t["location"].append(info["location"])
                injury_t["tweet_loc"].append(tweet["geo.geo.bbox"])


    filtered_t = pd.DataFrame(filtered_t)
    death_t = pd.DataFrame(death_t)
    injury_t = pd.DataFrame(injury_t)

    if not  os.path.exists("output/%s" % out_dir):
        os.makedirs("output/%s" % out_dir)

    filtered_t.to_csv("output/%s/filtered_tweets.csv" % out_dir)
    death_t.to_csv("output/%s/death_tweets.csv" % out_dir)
    injury_t.to_csv("output/%s/injury_tweets.csv" % out_dir)

    print(filtered_t)
    print(death_t)
    print(injury_t)

if __name__ == '__main__':
    main()
    # print(probe_info("Today is the 16th anniversary of 2005 Kashmir #earthquake\n\nThere can't be a bigger tragedy than this earthquake which resulted in the death of more than 70,000 people. https://t.co/1tKIOpTcns"))
