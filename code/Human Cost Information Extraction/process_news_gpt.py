import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from transformers import pipeline, AutoModelForCausalLM, GPTJForCausalLM, AutoTokenizer, StoppingCriteria
from word2number import w2n
import os
import dateutil.parser
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DisasterNLP(object):
    def __init__(self, country_name, month, day, pretrain_token, pretrain_type, pretrain_event, pretrain_parsing):
        self.out_dir = country_name + str(month) + '_' + str(day) + 'news'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_token)
        self.type_classifier = AutoModelForSequenceClassification.from_pretrained(pretrain_type).eval()
        self.event_classifier = AutoModelForSequenceClassification.from_pretrained(pretrain_event).eval()
        self.type_classifier.to(device)
        self.event_classifier.to(device)
        self.generator = AutoModelForCausalLM.from_pretrained(pretrain_parsing, revision="float16", low_cpu_mem_usage=True)
        self.generator.to(device)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(pretrain_parsing)
        self.stop_id = self.gen_tokenizer("###", return_tensors="pt").input_ids.to(device)
        self.delimiter_id = self.gen_tokenizer("|", return_tensors="pt").input_ids.to(device)

    def value_to_int(self, x):
        if x is None:
            return -1
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

    def probe_info(self, text):
        prompt1 = """Extract casualty statistics from tweets.

    [Tweet]: #Moscow Russian earthquake death toll at 6,954, over 400k injured &amp; 1.2m people displaced this weekend. https://t.co/o4JC4HqNU4
    [Query]: |Deaths|Injured|City|Country|Earthquake|
    [Key]: |6954|400000|Moscow|Russia|yes|

    ###

    [Tweet]: Crushing damage in Peking. Until now, 31 reported deaths and 4000 displaced. #chinahurricane #shaking #beijing #china https://t.co/Z3sDfjEHjC4
    [Query]: |Deaths|Injured|City|Country|Earthquake|
    [Key]: |31|none|Beijing|China|no|

    ###

    [Tweet]: Injuries in massive #Anchorage flood jumps to 724: govt https://t.co/7asdfasdfaDh
    [Query]: |Deaths|Injured|City|Country|Earthquake|
    [Key]: |none|724|Anchorage|USA|no|

    ###

    [Tweet]: Sudden earthquake in Saudi Arabia 30,000 injured and 4090 killed. #saudi #earthquake https://t.co/6BJNYBN38
    [Query]: |Deaths|Injured|City|Country|Earthquake|
    [Key]: |4090|30000|none|Saudi Arabia|yes|

    ###

    [Tweet]: BREAKING: Earthquake of 5.9 magnitude in Nice, France, killing 600 and 4,000 injured. #NICE #quake
    [Query]: |Deaths|Injured|City|Country|Earthquake|
    [Key]: |600|4000|Nice|France|yes|

    ###

    [Tweet]:
    """
        prompt2 = """[Query]: |Deaths|Injuries|City|Country|Earthquake|
    [Key]:"""

        text = prompt1 + " " + text + "\n" + prompt2
        iids = self.gen_tokenizer(text, return_tensors="pt").input_ids.to(device)
        original_len = len(text)
        generated_ids = self.generator.generate(iids, do_sample=False, temperature=1.0, max_new_tokens=25, return_dict_in_generate=True, output_scores=True)
        
        og_token_len = iids[0].shape[0]
        #print(len(generated_ids.scores))
        #print(generated_ids.scores[0].shape)
        #print(generated_ids.scores[0])

        
        #print(generated_ids.sequences[0].shape)
        new_gen = generated_ids.sequences[0][og_token_len+1:]
        #print(len(new_gen))
        #print(new_gen,"heheha")
        #print(delimiter_id.item())
        generated_text = self.gen_tokenizer.decode(generated_ids.sequences[0])
        indices = (new_gen==self.delimiter_id.item()).nonzero(as_tuple=True)[0]
        if indices.shape[0]==5:
            death_dict = {}
            for i in range(0, indices[0]):
                current_token = generated_ids.sequences[0][i+og_token_len+1] 
                decoded = self.gen_tokenizer.decode(current_token)
                temp = {}
                token_scores = generated_ids.scores[i+1]
                token_scores = torch.softmax(token_scores, dim=-1)
                #print(token_scores)
                #s = 0
                #for i in range(token_scores[0].shape[0]):
                #    s += 0 if token_scores[0][i].item() < 0 else token_scores[0][i].item()
                #    if token_scores[0][i].item() < 0 and token_scores[0][i].item() > -1e6:
                #        print(token_scores[0][i].item())
                #print(s)
                #exit()


                top = torch.topk(token_scores, 5)
                top_k_indices = top.indices
                #print(top_k_indices)
                #print(decoded)
                #print("-----")
                for j in range(top_k_indices[0].shape[0]):
                    temp[self.gen_tokenizer.decode(top_k_indices[0][j])] = top.values[0][j].item()
                death_dict[decoded] = temp

            injuries_dict = {}
            for i in range(indices[0]+1, indices[1]):
                current_token = generated_ids.sequences[0][i+og_token_len+1]
                decoded = self.gen_tokenizer.decode(current_token)
                temp = {}
                token_scores = generated_ids.scores[i+1]
                token_scores = torch.softmax(token_scores, dim=-1)
                #print(token_scores)
                #s = 0
                #for i in range(token_scores[0].shape[0]):
                #    s += 0 if token_scores[0][i].item() < 0 else token_scores[0][i].item()
                #    if token_scores[0][i].item() < 0 and token_scores[0][i].item() > -1e6:
                #        print(token_scores[0][i].item())
                #print(s)
                #exit()


                top = torch.topk(token_scores, 5)
                top_k_indices = top.indices
                #print(top_k_indices)
                #print(decoded)
                #print("-----")
                for j in range(top_k_indices[0].shape[0]):
                    temp[self.gen_tokenizer.decode(top_k_indices[0][j])] = top.values[0][j].item()
                injuries_dict[decoded] = temp
            
            location_dict = {}
            for i in range(indices[1]+1, indices[2]):
                current_token = generated_ids.sequences[0][i+og_token_len+1]
                decoded = self.gen_tokenizer.decode(current_token)
                temp = {}
                token_scores = generated_ids.scores[i+1]
                token_scores = torch.softmax(token_scores, dim=-1)
                #print(token_scores)
                #s = 0
                #for i in range(token_scores[0].shape[0]):
                #    s += 0 if token_scores[0][i].item() < 0 else token_scores[0][i].item()
                #    if token_scores[0][i].item() < 0 and token_scores[0][i].item() > -1e6:
                #        print(token_scores[0][i].item())
                #print(s)
                #exit()


                top = torch.topk(token_scores, 5)
                top_k_indices = top.indices
                #print(top_k_indices)
                #print(decoded)
                #print("-----")
                for j in range(top_k_indices[0].shape[0]):
                    temp[self.gen_tokenizer.decode(top_k_indices[0][j])] = top.values[0][j].item()
                location_dict[decoded] = temp
        elif indices.shape[0]==4:
            print(generated_text)
            death_dict = {}
            for i in range(indices[0]+1, indices[1]):
                current_token = generated_ids.sequences[0][i+og_token_len+1] 
                decoded = self.gen_tokenizer.decode(current_token)
                temp = {}
                token_scores = generated_ids.scores[i+1]
                token_scores = torch.softmax(token_scores, dim=-1)
                #print(token_scores)
                #s = 0
                #for i in range(token_scores[0].shape[0]):
                #    s += 0 if token_scores[0][i].item() < 0 else token_scores[0][i].item()
                #    if token_scores[0][i].item() < 0 and token_scores[0][i].item() > -1e6:
                #        print(token_scores[0][i].item())
                #print(s)
                #exit()


                top = torch.topk(token_scores, 5)
                top_k_indices = top.indices
                print(top_k_indices)
                print(decoded)
                print("-----")
                for j in range(top_k_indices[0].shape[0]):
                    temp[self.gen_tokenizer.decode(top_k_indices[0][j])] = top.values[0][j].item()
                death_dict[decoded] = temp

            injuries_dict = {}
            for i in range(indices[1]+1, indices[2]):
                current_token = generated_ids.sequences[0][i+og_token_len+1]
                decoded = self.gen_tokenizer.decode(current_token)
                temp = {}
                token_scores = generated_ids.scores[i+1]
                token_scores = torch.softmax(token_scores, dim=-1)
                #print(token_scores)
                #s = 0
                #for i in range(token_scores[0].shape[0]):
                #    s += 0 if token_scores[0][i].item() < 0 else token_scores[0][i].item()
                #    if token_scores[0][i].item() < 0 and token_scores[0][i].item() > -1e6:
                #        print(token_scores[0][i].item())
                #print(s)
                #exit()


                top = torch.topk(token_scores, 5)
                top_k_indices = top.indices
                #print(top_k_indices)
                #print(decoded)
                #print("-----")
                for j in range(top_k_indices[0].shape[0]):
                    temp[self.gen_tokenizer.decode(top_k_indices[0][j])] = top.values[0][j].item()
                injuries_dict[decoded] = temp
            
            location_dict = {}
            for i in range(indices[2]+1, indices[3]):
                current_token = generated_ids.sequences[0][i+og_token_len+1]
                decoded = self.gen_tokenizer.decode(current_token)
                temp = {}
                token_scores = generated_ids.scores[i+1]
                token_scores = torch.softmax(token_scores, dim=-1)
                #print(token_scores)
                #s = 0
                #for i in range(token_scores[0].shape[0]):
                #    s += 0 if token_scores[0][i].item() < 0 else token_scores[0][i].item()
                #    if token_scores[0][i].item() < 0 and token_scores[0][i].item() > -1e6:
                #        print(token_scores[0][i].item())
                #print(s)
                #exit()


                top = torch.topk(token_scores, 5)
                top_k_indices = top.indices
                #print(top_k_indices)
                #print(decoded)
                #print("-----")
                for j in range(top_k_indices[0].shape[0]):
                    temp[self.gen_tokenizer.decode(top_k_indices[0][j])] = top.values[0][j].item()
                location_dict[decoded] = temp
        else:
            return None

        #print(gen_tokenizer.decode(new_gen))

        cleaned = generated_text[original_len:]
        cleaned = cleaned[:cleaned.rfind("###")]
        if cleaned.count("|") == prompt2.count("|"):
            cleaned = cleaned[2:cleaned.rfind("|")]
        
        info = cleaned.split("|")
        for i in range(len(info)):
            if(info[i]=="none"):
                info[i] = None
        if len(info) >= 5:
            info = {"deaths":[self.value_to_int(info[0]), death_dict], "injuries":[self.value_to_int(info[1]), injuries_dict], "city":[info[2], location_dict], "country":[info[3], -1], "earthquake":[info[4], -1]}
        else:
            info = None
        return info

    def filter_type(self, text):
        inputs = self.tokenizer.encode(text, return_tensors='pt').to(device)
        output = self.type_classifier(inputs)
        c = torch.argmax(output.logits, dim=-1).item()
        return c==0

    def filter_event(self, text):
        inputs = self.tokenizer.encode(text, return_tensors='pt').to(device)
        output = self.event_classifier(inputs)
        c = torch.argmax(output.logits, dim=-1).item()
        return c==0

    def process_tweet(self, text):
        is_eq = self.filter_event(text)
        is_stat = self.filter_type(text)
        
        if is_eq and is_stat:
            info = self.probe_info(text)
            #print(info)
            #exit()
            return info
        else:
            return None

def main():
    parser = argparse.ArgumentParser(description="Run analysis")
    parser.add_argument("--country_name", type=str, help="Country name of the earthquake")
    parser.add_argument("--month", type=int, help="Queried month datte")
    parser.add_argument("--day", type=int, help="Queried day")
    parser.add_argument("--pretrain_token", type=str, default='xlm-roberta-base', help="pretrained token model")
    parser.add_argument("--pretrain_type", type=str, default="models/xlm-r_type_imbalanced", help="pretrained type model")
    parser.add_argument("--pretrain_event", type=str, default="models/xlm-r_event_balanced", help="pretrained event model")
    parser.add_argument("--pretrain_parsing", type=str, default="EleutherAI/gpt-j-6B", help="pretrained parsing model")
    parser.add_argument("--news_data_dir", type=str, default="news_api_0728_philippines.csv", help="data file name")
    args = parser.parse_args()

    model = DisasterNLP(args.country_name, args.month, args.day, args.pretrain_token, args.pretrain_type, args.pretrain_event, args.pretrain_parsing)
   

    event_year = 2022

    
    #stop_criteria = StoppingCriteria(stop_id)

    data_dir = "data/"
    df = pd.read_csv(data_dir + args.news_data_dir)

    filtered_t = {'origin_id':[],"time":[],"text":[]}
    death_t = {'origin_id':[], "time":[], "text":[], "deaths":[], "death_score":[], "injuries":[], "injury_score":[], "city":[], "city_score":[], "country":[], "country_score":[], "earthquake":[], "earthquake_score":[]}
    injury_t = {'origin_id':[], "time":[], "text":[], "deaths":[], "death_score":[], "injuries":[], "injury_score":[], "city":[], "city_score":[], "country":[], "country_score":[], "earthquake":[], "earthquake_score":[]}
    print("*"*50)
    for i, tweet in df.iterrows():
        if i % 200 == 0:
            print("%d / %d" %(i, df.shape[0]))
        tweet["text"] = str(tweet["title"]) + str(tweet["content"])
        info = model.process_tweet(tweet["text"])
        tweet["created_at"] = tweet["publishedAt"]
        if info is None:
            filtered_t["time"].append(tweet["created_at"])
            filtered_t["text"].append(tweet["text"])
            filtered_t["origin_id"].append(i)
        elif (info["deaths"][0]==-1) and (info["injuries"][0]==-1):
            filtered_t["time"].append(tweet["created_at"])
            filtered_t["text"].append(tweet["text"])
            filtered_t["origin_id"].append(i)
        else:
            if info["deaths"][0] != -1:
                death_t["origin_id"].append(i)
                death_t["time"].append(tweet["created_at"])
                death_t["text"].append(tweet["text"])
                death_t["deaths"].append(info["deaths"][0])
                death_t["death_score"].append(info["deaths"][1])
                death_t["injuries"].append(info["injuries"][0])
                death_t["injury_score"].append(info["injuries"][1])
                death_t["city"].append(info["city"][0])
                death_t["city_score"].append(info["city"][1])
                death_t["country"].append(info["country"][0])
                death_t["country_score"].append(info["country"][1])
                death_t["earthquake"].append(info["earthquake"][0])
                death_t["earthquake_score"].append(info["earthquake"][1])
                #death_t["tweet_loc"].append(tweet["geo.geo.bbox"])

                
            if info["injuries"][0] != -1:
                injury_t["origin_id"].append(i)
                injury_t["time"].append(tweet["created_at"])
                injury_t["text"].append(tweet["text"])
                injury_t["deaths"].append(info["deaths"][0])
                injury_t["death_score"].append(info["deaths"][1])
                injury_t["injuries"].append(info["injuries"][0])
                injury_t["injury_score"].append(info["injuries"][1])
                injury_t["city"].append(info["city"][0])
                injury_t["city_score"].append(info["city"][1])
                injury_t["country"].append(info["country"][0])
                injury_t["country_score"].append(info["country"][1])
                injury_t["earthquake"].append(info["earthquake"][0])
                injury_t["earthquake_score"].append(info["earthquake"][1])
                #injury_t["tweet_loc"].append(tweet["geo.geo.bbox"])


    filtered_t = pd.DataFrame(filtered_t)
    death_t = pd.DataFrame(death_t)
    injury_t = pd.DataFrame(injury_t)

    if not  os.path.exists("output/%s" % model.out_dir):
        os.makedirs("output/%s" % model.out_dir)

    filtered_t.to_csv("output/%s/filtered_tweets.csv" % model.out_dir)
    death_t.to_csv("output/%s/death_tweets.csv" % model.out_dir)
    injury_t.to_csv("output/%s/injury_tweets.csv" % model.out_dir)

    print(filtered_t)
    print(death_t)
    print(injury_t)
    print(model.out_dir)

if __name__ == '__main__':
    main()
    #print(probe_info("A lot of damage in Okay. So far 29 reported deaths. #haitiearthquake #tranbleman #lescayes #ayiti #haiti https://t.co/ZPTHugLeD3"))
