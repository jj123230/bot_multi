# -*- coding: utf-8 -*-
print("Running bot_multi")
bot_name = 'bot_multi'
import os
import sys
try:
    mother_folder = sys.argv[1] ## 線上主機中的檔案位置
except:
    mother_folder = r"D:\test\Bots-2024\bot-chat" ## 線上主機中的檔案位置，線上，到客戶資料夾為止，EX:客戶資料夾名稱= test1

sys.path.append(f"{mother_folder}/{bot_name}")  ## 指到自己的Folder
import setting_config

test = setting_config.test

## SQL 位址
environment = setting_config.env
server = setting_config.DB_info['address']
database = ''
username = setting_config.DB_info['uid']  
password = setting_config.DB_info['pwd']

## ip
host_ip = setting_config.ip
port_ip = setting_config.port

try:
    font = f"{mother_folder}/{bot_name}/Models/JhengHei.ttf"
    if not os.path.exists(f'{mother_folder}/{bot_name}/Models'):
        os.makedirs(f'{mother_folder}/{bot_name}/Models')
except:
    pass

import math
e = math.e

from transformers import pipeline
from lingua import LanguageDetectorBuilder, Language, IsoCode639_1

import hanzidentifier
def detect_lang(text):
    if hanzidentifier.identify(text)==0 or hanzidentifier.identify(text)==3 or hanzidentifier.identify(text)==4:
        ## lang = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")(text)[0]['label']
        lang = detector.detect_language_of(text).iso_code_639_1.name.lower()
        if lang == 'zh':
            lang = 'zh-TW'
    elif hanzidentifier.identify(text)==1:
        lang = 'zh-TW'
    elif hanzidentifier.identify(text)==2:
        lang = 'zh-CN'
    return lang

import re
import pyodbc
import json
import pandas as pd
from datetime import datetime
import tensorflow as tf
from sqlalchemy import create_engine, NVARCHAR
from tqdm.contrib import tzip

try:
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    
    strategy = tf.distribute.MirroredStrategy(["GPU:0"])
except:
    pass

if tf.config.list_physical_devices('GPU'):
    print(f"Tensorflow on GPU: {tf.test.is_gpu_available()}")

import torch
print(f"Torch on GPU: {torch.cuda.is_available()}")

from transformers import BertTokenizerFast, AutoTokenizer, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification
model_name = 'xlm-roberta-base'#'google-bert/bert-base-multilingual-uncased'

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
compressor = FlashrankRerank(model = "ms-marco-MultiBERT-L-12")

from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
from sklearn.preprocessing import LabelEncoder

import random
import time
import jieba
from joblib import Parallel, delayed

language_codes = {
    "af": "afrikaans",
    "am": "amharic",
    "ar": "arabic",
    "az": "azerbaijani",
    "be": "belarusian",
    "bg": "bulgarian",
    "bn": "bengali",
    "bs": "bosnian",
    "ca": "catalan",
    "ceb": "cebuano",
    "co": "corsican",
    "cs": "czech",
    "cy": "welsh",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "eo": "esperanto",
    "es": "spanish",
    "et": "estonian",
    "eu": "basque",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "fy": "frisian",
    "ga": "irish",
    "gd": "scotsgaelic",
    "gl": "galician",
    "gu": "gujarati",
    "ha": "hausa",
    "haw": "hawaiian",
    "he": "hebrew",
    "hi": "hindi",
    "hmn": "hmong",
    "hr": "croatian",
    "ht": "haitiancreole",
    "hu": "hungarian",
    "hy": "armenian",
    "id": "indonesian",
    "ig": "igbo",
    "is": "icelandic",
    "it": "italian",
    "iw": "hebrew",  # Note: This is a duplicate of "he"
    "ja": "japanese",
    "jw": "javanese",
    "ka": "georgian",
    "kk": "kazakh",
    "km": "khmer",
    "kn": "kannada",
    "ko": "korean",
    "ku": "kurdish(kurmanji)",
    "ky": "kyrgyz",
    "la": "latin",
    "lb": "luxembourgish",
    "lo": "lao",
    "lt": "lithuanian",
    "lv": "latvian",
    "mg": "malagasy",
    "mi": "maori",
    "mk": "macedonian",
    "ml": "malayalam",
    "mn": "mongolian",
    "mr": "marathi",
    "ms": "malay",
    "mt": "maltese",
    "my": "myanmar(burmese)",
    "ne": "nepali",
    "nl": "dutch",
    "no": "norwegian",
    "ny": "chichewa",
    "or": "odia",
    "pa": "punjabi",
    "pl": "polish",
    "ps": "pashto",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sd": "sindhi",
    "si": "sinhala",
    "sk": "slovak",
    "sl": "slovenian",
    "sm": "samoan",
    "sn": "shona",
    "so": "somali",
    "sq": "albanian",
    "sr": "serbian",
    "st": "sesotho",
    "su": "sundanese",
    "sv": "swedish",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "tg": "tajik",
    "th": "thai",
    "tl": "filipino",
    "tr": "turkish",
    "ug": "uyghur",
    "uk": "ukrainian",
    "ur": "urdu",
    "uz": "uzbek",
    "vi": "vietnamese",
    "xh": "xhosa",
    "yi": "yiddish",
    "yo": "yoruba",
    "zh-CN": "chinese(simplified)",
    "zh-TW": "chinese(traditional)",
    "zu": "zulu",
    }

language_iso = {key: getattr(IsoCode639_1, key[:2].upper(), None) for key in language_codes}

'''
1. load SQL data
'''
def load_sql_data(faq_min, faq_max, intent_min, intent_max, env):
    global train_faq, train_intent, fit_faq, fit_faqsetting, match_faq, train_synonym, replace_synonyms, train_exclude, replace_exclude
    global thresh_dic, chat_dic, judge_dic, translate_dic, gen_dic, task, chat, lan_list, detector
    def get_target_len(df, col, left, right):
        return df[(df[col].str.len() >= left) & (df[col].str.len() <= right)]
    
    ## 1. traindata
    cnxn = pyodbc.connect(f'DRIVER=ODBC Driver 17 for SQL Server;SERVER={server}; DATABASE={database};UID={username};PWD={password}')
    data = json.load(open(f'{mother_folder}/{bot_name}/config.json'))[env]
    
    ## build synonym replace
    train_synonym = pd.read_sql(f"SELECT * FROM {data['synonym']}", cnxn)[['Text', 'SynonymId']]
    train_synonymE = pd.read_sql(f"SELECT * FROM {data['synonymE']}", cnxn)[['Text', 'SynonymId']]
    train_synonymS = pd.read_sql(f"SELECT * FROM {data['synonymS']}", cnxn)[['Text', 'SynonymId']]
    
    train_synonym = pd.concat([train_synonym, train_synonymE, train_synonymS], ignore_index=True)
    
    category_to_first_word = train_synonym.groupby('SynonymId')['Text'].first().to_dict()
    train_synonym['first_word'] = train_synonym['SynonymId'].map(category_to_first_word)

    def replace_synonyms(sentence, synonym, synonym_first=""):
        if type(synonym_first)== str:
            for i in synonym:
                sentence = sentence.replace(i, synonym_first)
        else:
            for i,j in zip(synonym, synonym_first):
                sentence = sentence.replace(i,j)
        return sentence
    
    ## build exclude replace
    train_exclude = pd.read_sql(f"SELECT * FROM {data['exclude']}", cnxn)[['Text']]
    train_excludeE = pd.read_sql(f"SELECT * FROM {data['excludeE']}", cnxn)[['Text']]
    
    train_exclude = pd.concat([train_exclude, train_excludeE])
                                
    def replace_exclude(sentence, exclude):
        text = ""
        for j in [i for i in jieba.cut(sentence) if i not in list(exclude)]:
            text+= j
        return text
    
    ## FAQ
    train_faq = pd.read_sql(f"SELECT * FROM {data['faq']}", cnxn).assign(LanguageCode='zh-TW')
    train_faqE = pd.read_sql(f"SELECT * FROM {data['faqE']}", cnxn).assign(LanguageCode='en')
    train_faqS = pd.read_sql(f"SELECT * FROM {data['faqS']}", cnxn)
    
    train_faq = pd.concat([train_faq, train_faqE, train_faqS], ignore_index=True)
    
    for i in set(train_faq['LanguageCode']):
        globals()[f'train_faq_{i}'] = train_faq[train_faq['LanguageCode']==i]
    
    train_faq['Text'] = [replace_synonyms(i, train_synonym['Text'], train_synonym['first_word']) for i in train_faq['Text']]
    ## train_faq['Text'] = Parallel(n_jobs=8)(delayed(replace_exclude)(i, train_exclude['Text']) for i in train_faq['Text'])
    train_faq = get_target_len(train_faq, 'Text', faq_min, train_faq['Text'].str.len().quantile(faq_max))
    train_faq = train_faq[~train_faq['Text'].duplicated()]
    
    lan_list = list(set(train_faq['LanguageCode']))
    for lan in lan_list:
        globals()[f'temp_{lan}'] = {}
        
    detect_list = [language_iso[i] for i in lan_list]
    detector = LanguageDetectorBuilder.from_iso_codes_639_1(*detect_list).build()
    '''
    ## Intent    
    for i in set(train_intent['LanguageCode']):
        globals()[f'train_intent_{i}'] = train_intent[train_intent['LanguageCode']==i]
    
    train_intent = train_intent[~train_intent.Rank.apply(lambda x : x==2)]
    train_intent['Text'] = Parallel(n_jobs=8)(delayed(replace_exclude)(i, train_exclude['Text']) for i in train_intent['Text']) 
    train_intent = get_target_len(train_intent, 'Text', intent_min, train_intent['Text'].str.len().quantile(intent_max))
    train_intent = train_intent[~train_intent['Text'].duplicated()]
    
    
    ## Emo
    train_emo = pd.read_sql(f"SELECT * FROM {data['emotionmodelupdate']}", cnxn)
        
    stp = OpenCC('s2twp')
    train_emo = train_emo.dropna(subset = ['EmotionName'])
    train_emo['Question'] = train_emo['Question'].apply(lambda x :stp.convert(x))
    train_emo['EmotionName'] = train_emo['EmotionName'].apply(lambda x :stp.convert(x))
    train_emo = get_target_len(train_emo, 'Question', emo_min, train_emo['Question'].str.len().quantile(emo_max))
    train_emo = train_emo[~train_emo['Question'].apply(lambda x : x == '')]
    train_emo = train_emo[~train_emo['Question'].duplicated()]
    
    def get_random_rows(group, row_n=1000):
        return group.sample(n=min(row_n, group.shape[0]), random_state=42)
    second_most = train_emo.groupby('EmotionName').size().nlargest().iloc[1]
    train_emo = train_emo.groupby('EmotionName').apply(get_random_rows, second_most).reset_index(drop=True)
    
    for i in list(set(train_emo['EmotionName'])):
        try:
            lack = train_emo.loc[train_emo['EmotionName']==i].sample(n=max(0, second_most-len(train_emo.loc[train_emo['EmotionName']==i])))
        except:
            lack = train_emo.loc[train_emo['EmotionName']==i]
        train_emo = pd.concat([train_emo, lack], ignore_index=True)
    '''
    def replace_dic_nan(dic):
        for key, value in dic.items():
            try:
                if pd.isna(value):
                    dic[key] = None
            except:
                pass
        return dic
    
    chat = pd.read_sql(f"SELECT * FROM {data['chat']}", cnxn)
    
    ## Threshold
    threshold = pd.read_sql(f"SELECT * FROM {data['Option']}", cnxn)
    thresh_dic = {i: float(threshold['Value'][threshold['Key']==i].values[0]) for i in threshold['Key']}
    
    ## chat_dic
    try:
        chat_dic = {'chatbot':int(thresh_dic['ChatBotKind'])}
    except:
        chat_dic = {'chatbot':0}
    
    ## judge_dic
    try:
        judge_dic = {'chatbot':int(thresh_dic['JudgeBotKind'])}
    except:
        judge_dic = {'chatbot':0}
    judge_dic['judgebot_score'] = chat[chat['Id'] == judge_dic['chatbot']]['Score'].values[0]
    
    ## translate_dic
    try:
        translate_dic = {'chatbot': int(thresh_dic['TranslBotKind'])}
    except:
        translate_dic = {'chatbot':0}
    
    ## gen_dic
    try:
        gen_dic = {'chatbot': int(thresh_dic['GenBotKind'])}
    except:
        gen_dic = {'chatbot':0}
    
    def get_dic_details(input_dic):
        try:
            input_dic['chatbot_name'] = chat[chat['Id'] == input_dic['chatbot']]['Name'].values[0]
            input_dic['chatbot_key'] = chat[chat['Id'] == input_dic['chatbot']]['APIKey'].values[0]
            input_dic['chatbot_memo'] = chat[chat['Id'] == input_dic['chatbot']]['Memo'].values[0]
            
            input_dic['chatbot_temperature'] = chat[chat['Id'] == input_dic['chatbot']]['Temperature'].values[0]
            input_dic['chatbot_token'] = chat[chat['Id'] == input_dic['chatbot']]['Token'].values[0]
                
            input_dic['chatbot_4090'] = list(chat[chat['Memo'] == "4090"]['Name'])
        except:
            input_dic['chatbot_name'] = 'gemma2'
            input_dic['chatbot_memo'] = '4090'
        input_dic = replace_dic_nan(input_dic)
        return input_dic
    
    chat_dic = get_dic_details(chat_dic)
    judge_dic = get_dic_details(judge_dic)
    translate_dic = get_dic_details(translate_dic)
    gen_dic = get_dic_details(gen_dic)
    
    '''
    ## load chatbot role
    role = pd.read_sql(f"SELECT * FROM {data['BotRole']}", cnxn)
    chat_dic['chatbot_role'] = role[role['Code'] == code]['RoleName'].values[0]
    
    chat_dic['chatbot_chat'] = role[role['Code'] == code]['Role_Chat'].values[0]
    chat_dic['chatbot_syn'] = role[role['Code'] == code]['Role_SynophaseCustomize'].values[0]
    '''
    task = pd.read_sql(f"SELECT * FROM {data['task']}", cnxn)
    
    ## judge_translate
    fit_faqsetting = pd.read_sql(f"SELECT * FROM {data['fitFaqSetting']}", cnxn)
    fit_faq = pd.read_sql(f"SELECT * FROM {data['fitFaq']}", cnxn)

    match_faq = pd.read_sql(f"SELECT * FROM {data['synophase']}", cnxn)
    print("輸入資料:完成")

def return_task_dic(model=""):
    input_dic = {}
    code = 'zh-TW'
    try:
        if model=="":
            input_dic['translate']=task[(task['Code']==code)&(task['TypeName']=='對話翻譯')&(task['BotModel']==translate_dic['chatbot_name'])]['Prompt'].values[0]
        else:
            input_dic['translate']=task[(task['Code']==code)&(task['TypeName']=='對話翻譯')&(task['BotModel']==model)]['Prompt'].values[0]
    except:
        input_dic['translate'] = task[(task['Code']==code) & (task['TypeName']=='對話翻譯') & (task['BotModel']=='AIBot')]['Prompt'].values[0]
        
    input_dic['translate_plus'] = "請把 {} 翻譯成 {}，請先對翻譯內容的品質進行自我評分 1-9分，分數超過 {} 分才通過，未達標則重新翻譯並重複以上流程。\
        回答內容只需提供文本，不需要額外的解釋也不需要分數，只要回傳翻譯內容就好。"
    try:
        if model=="":
            input_dic['gen'] = task[(task['Code']==code) & (task['TypeName']=='自動生成') & (task['BotModel']==gen_dic['chatbot_name'])]['Prompt'].values[0]
        else:
            input_dic['gen'] = task[(task['Code']==code) & (task['TypeName']=='自動生成') & (task['BotModel']==model)]['Prompt'].values[0]
    except:
        input_dic['gen'] = task[(task['Code']==code) & (task['TypeName']=='自動生成') & (task['BotModel']=='AIBot')]['Prompt'].values[0]
    try:
        if model=="":
            input_dic['judge_translate']=task[(task['Code']==code)&(task['TypeName']=='校稿')&(task['BotModel']==judge_dic['chatbot_name'])]['Prompt'].values[0]
        else:
            input_dic['judge_translate']=task[(task['Code']==code)&(task['TypeName']=='校稿')&(task['BotModel']==model)]['Prompt'].values[0]
    except:
        input_dic['judge_translate'] = task[(task['Code']==code) & (task['TypeName']=='校稿') & (task['BotModel']=='AIBot')]['Prompt'].values[0]
    try:
        if model=="":
            input_dic['chat'] = task[(task['Code']==code) & (task['TypeName']=='聊天') & (task['BotModel']==chat_dic['chatbot_name'])]['Prompt'].values[0]
        else:
            input_dic['chat'] = task[(task['Code']==code) & (task['TypeName']=='聊天') & (task['BotModel']==model)]['Prompt'].values[0]
    except:
        input_dic['chat'] = task[(task['Code']==code) & (task['TypeName']=='聊天') & (task['BotModel']=='AIBot')]['Prompt'].values[0]
    return input_dic
        
def load_chatbot():
    global vector, retriever, rag_chain, lan_list, lan_string
    ## 載入聊天機器人       
    # 載入語言與語料
    lan_list = list(set(train_faq['LanguageCode']))
    for lan in lan_list:
        globals()[f'temp_{lan}'] = {}
        
    # temp dic
    for lan in lan_list:
        for i,j,k,l in zip(train_faq.query(f"LanguageCode=='{lan}'")['IntentionName'],
                           train_faq.query(f"LanguageCode=='{lan}'")['Demand'],
                           train_faq.query(f"LanguageCode=='{lan}'")['Supply'],
                           train_faq.query(f"LanguageCode=='{lan}'")['Text']):
            try:
                try:
                    globals()[f'temp_{lan}'][i][j][1].append(l)
                except:
                    globals()[f'temp_{lan}'][i][j] = [k, []]
            except:
                globals()[f'temp_{lan}'][i] = {}

    lan_string = {}
    for lan in lan_list:
        for i in globals()[f'temp_{lan}'].items():
            lan_string[f'{lan}_{i[0]}'] = (", ".join(f"[Q: {j[0]} A: {j[1][0]}]" for j in i[1].items()))
            ## result.append(", ".join(f"[original question: {i[0]},Q: {i[1][1]} A: {i[1][0]}]" for i in globals()[f'temp_{lang}'].items()))
        
    ## load model
    for model in chat_dic['chatbot_4090']:
        embeddings = OllamaEmbeddings(model=model)
            
        ## original retriever
        documents = [Document(page_content= i[1], metadata={"language": i[0].split("_")[0]}) for i in lan_string.items()]
        
        try:
            vector.delete_collection()
        except:
            pass
        
        globals()[f'vector_{model}'] = Chroma.from_documents(documents= documents, embedding=embeddings, collection_name=f'vec_{model}')
        globals()[f'retriever_{model}'] = globals()[f'vector_{model}'].as_retriever(search_kwargs={"k": 5})
        
        '''
        for lang in lang_list:
            globals()[f'documents_{lang}'] = [Document(page_content= lang_string[lang], metadata={"source": "Q&A Dataset"})]
            
            try:
                globals()[f'vector_{model}_{lang}'].delete_collection()
            except:
                pass
            globals()[f'vector_{model}_{lang}'] = Chroma.from_documents(documents= globals()[f'documents_{lang}'], embedding=embeddings,
                                                                             collection_name=f'vec_{model}_{lang}')
            globals()[f'retriever_{model}_{lang}'] = globals()[f'vector_{model}_{lang}'].as_retriever()
            
            globals()[f'rag_chain_{model}_{lang}'] = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", 
                                                                                      retriever=globals()[f'retriever_{model}_{lang}'])
            '''
        print(f"已載入{chat_dic['chatbot_4090']}中的{model}")
    '''
    except:
        print("載入地端ollama失敗，需先載入資料 updatefaq")
    '''
        
'''
Build dic
'''
## 4. build dictionary
def set_random_seed():
    random.seed(420)
    tf.random.set_seed(420)
    np.random.seed(420)
    torch.manual_seed(420)

def build_dic():
    global train_faq, train_intent
    global maxlen_faq, maxlen_intent, dic_faq, dic_intent, subword_encoder_faq, subword_encoder_intent
    # global vocab_size_faq, vocab_size_intent, vocab_size_emo, x_train_faq, x_train_intent, x_train_emo, y_train_faq, y_train_intent, y_train_emo
    ## FAQ_train_data
    df_train_faq = tf.data.Dataset.from_tensor_slices(train_faq['Text'])
    subword_encoder_faq = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (i.numpy() for i in tf.data.Dataset.from_tensor_slices(train_faq['Text'])), 
        target_vocab_size=2**13,
        max_subword_length=1)

    vocab_size_faq = subword_encoder_faq.vocab_size

    encode_train_faq = [subword_encoder_faq.encode(i.numpy()) for i in df_train_faq]
    x_train_faq = keras.preprocessing.sequence.pad_sequences(encode_train_faq, padding='post')
    maxlen_faq = x_train_faq.shape[1]

    labeler = LabelEncoder()
    train_faq['label_faq'] = labeler.fit_transform(train_faq['Supply']) ## FaqSettingId
    train_faq.label_faq = pd.Categorical(train_faq.label_faq)
    dic_faq = dict(zip(range(len(labeler.classes_)), labeler.classes_))
    
    y_train_faq = np.array(train_faq.label_faq)

    ## intent_train_data
    df_train_intent = tf.data.Dataset.from_tensor_slices(train_faq['Text'])
    subword_encoder_intent = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (i.numpy() for i in tf.data.Dataset.from_tensor_slices(train_faq['Text'])), 
            target_vocab_size=2**13,
            max_subword_length=1)

    vocab_size_intent = subword_encoder_intent.vocab_size

    encode_train_intent = [subword_encoder_intent.encode(i.numpy()) for i in df_train_intent]
    x_train_intent = keras.preprocessing.sequence.pad_sequences(encode_train_intent, padding='post')
    maxlen_intent = x_train_intent.shape[1]  

    labeler = LabelEncoder()
    train_faq['label_intent'] = labeler.fit_transform(train_faq['IntentionName']) ## OriginalIntentId
    train_faq.label_intent = pd.Categorical(train_faq.label_intent)
    dic_intent = dict(zip(range(len(labeler.classes_)), labeler.classes_))

    y_train_intent = np.array(train_faq.label_intent)
    print("訓練用資料:完成")
    
    return vocab_size_faq, vocab_size_intent, x_train_faq, y_train_faq, x_train_intent, y_train_intent

'''
4. albert & gpt2
'''
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"

def encode_Bert(bert_tokenizer, datax, datay, len_n):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_l = []
    token_type_ids_l = []
    attention_mask_l = []
    label_l = []
    for i,j in zip(datax, datay):
        bert_input = bert_tokenizer.encode_plus(i, add_special_tokens=True, 
                                                max_length=len_n, 
                                                padding='max_length', 
                                                truncation=True, 
                                                return_attention_mask=True)
        input_ids_l.append(bert_input['input_ids'])
        attention_mask_l.append(bert_input['attention_mask'])
        label_l.append(j)
    
    return tf.data.Dataset.from_tensor_slices((input_ids_l, attention_mask_l, label_l))

## Set predictions
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def get_predictions(model, data, compute_acc=False):
    model.to(device)
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        for i in data:
            tokens_tensors, masks_tensors = [torch.tensor(j.numpy(), dtype=torch.long).to(device) for j in i[:2]]
            outputs = model(input_ids=tokens_tensors, attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            prob = logits
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = torch.tensor(i[2].numpy(), dtype=torch.long).to(device)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
                probability = prob
            else:
                predictions = torch.cat((predictions, pred))
                probability = torch.cat((probability, prob))
    
    if compute_acc:
        acc = correct / total
        predictions = predictions.tolist()
        probability = list(map(lambda x : softmax(x), probability.tolist()))
        return predictions, acc, probability
    
    predictions = predictions.tolist()
    probability = list(map(lambda x : softmax(x), probability.tolist()))
    return predictions, probability


def run_bert(now_model, bert_tokenizer, bert_model, func_name, epoch, x_train, y_train, test=0):
    if int(len(x_train)**(1/e))>30:
        batchsize = 30
    else:
        batchsize = int(len(x_train)**(1/e))
        
    bert_model.to(device)
    bert_model.config.pad_token_id = bert_model.config.eos_token_id
    bert_model.train()
    learning_rate= np.linspace(1e-5, 1e-6, epoch)
    
    maxlen = globals()[f'maxlen_{func_name}']
    if test==1:
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        
        BertTrain = encode_Bert(bert_tokenizer, x_train, y_train, maxlen).shuffle(6000).batch(batchsize)
        BertTest = encode_Bert(bert_tokenizer, x_test, y_test, maxlen).batch(batchsize)
        
        for i,l in zip(range(epoch), learning_rate):
            optimizer = torch.optim.Adam(bert_model.parameters(), lr=l)
            time_start = time.time()
            running_loss1, running_loss2 = 0.0, 0.0
            
            timer = 0
            for j in BertTrain:
                timer+=1
                
                tokens_tensor, masks_tensor, label = [torch.tensor(k.numpy(), dtype=torch.long).to(device) for k in j[:3]]  
                optimizer.zero_grad()
                outputs = bert_model(input_ids=tokens_tensor, attention_mask=masks_tensor, labels=label)
                
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                running_loss1 += loss.item()
                
                if timer == 10:
                    time.sleep(0.1)
                    timer=0
                
            for j in BertTest:
                timer+=1
                
                tokens_tensor, masks_tensor, label = [torch.tensor(k.numpy(), dtype=torch.long).to(device) for k in j[:3]]
                outputs = bert_model(input_ids=tokens_tensor, attention_mask=masks_tensor, labels=label)
                
                loss = outputs[0]
                running_loss2 += loss.item()
                
                if timer == 10:
                    time.sleep(0.1)
                    timer=0
            
            _, acc1,_ = get_predictions(bert_model, BertTrain, compute_acc=True)
            _, acc2,_ = get_predictions(bert_model, BertTest, compute_acc=True)
                
            print(f'[train_{func_name} {i+1}] loss: {round(running_loss1, 4)}, acc: {round(acc1, 4)}')
            print(f'[test_{func_name} {i+1}] loss: {round(running_loss2, 4)}, acc: {round(acc2, 4)}')
            
            time_end = time.time()
            time_c= time_end - time_start
            
            print("------------------------------------")
            print('time cost', time_c, 's')
            print('time cost hours', time_c/3600, 'hr')
            print("                                    ")
            
            with open(f"{mother_folder}/{bot_name}/Models/{now_model}_{func_name}", mode="wb") as f:
                torch.save(bert_model, f)
                
            print(f"{now_model}_{func_name} {i+1}:訓練完畢")
            torch.cuda.empty_cache()
            tf.keras.backend.clear_session()
           
    elif test==0:
        BertTrain = encode_Bert(bert_tokenizer, x_train, y_train, maxlen).shuffle(6000).batch(batchsize)
        
        for i,l in zip(range(epoch), learning_rate):
            optimizer = torch.optim.Adam(bert_model.parameters(), lr=l)
            time_start = time.time()
            running_loss1, running_loss2 = 0.0, 0.0
            
            timer = 0
            for j in BertTrain:
                timer+=1
                
                tokens_tensor, masks_tensor, label = [torch.tensor(k.numpy(), dtype=torch.long).to(device) for k in j[:3]]  
                optimizer.zero_grad()
                outputs = bert_model(input_ids=tokens_tensor, attention_mask=masks_tensor, labels=label)
                
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                running_loss1 += loss.item()
                
                if timer == 5:
                    time.sleep(0.1)
                    timer=0
            
            _, acc1,_ = get_predictions(bert_model, BertTrain, compute_acc=True)
                
            print(f'[train_{func_name} {i+1}] loss: {round(running_loss1, 4)}, acc: {round(acc1, 4)}')
            
            time_end = time.time()
            time_c= time_end - time_start
            
            print("------------------------------------")
            print('time cost', time_c, 's')
            print('time cost hours', time_c/3600, 'hr')
            print("                                    ")
            
            with open(f"{mother_folder}/{bot_name}/Models/{now_model}_{func_name}", mode="wb") as f:
                torch.save(bert_model, f)
                
            print(f"{now_model}_{func_name} {i+1}:訓練完畢")
            torch.cuda.empty_cache()
            tf.keras.backend.clear_session()
            
    elif test==2:
        bert_model = torch.load(f"{mother_folder}/{bot_name}/Models/{now_model}_{func_name}", map_location=torch.device(device_name))
    
    try:
        os.remove(f"{mother_folder}/{bot_name}/Models/{now_model}_{func_name}")
    except:
        pass
    
    with open(f"{mother_folder}/{bot_name}/Models/{now_model}_{func_name}", mode="wb") as f:
        torch.save(bert_model, f)
        
    print(f"{now_model}_{func_name} model:載入完畢")
    torch.cuda.empty_cache()
    tf.keras.backend.clear_session()
    
    return bert_model

'''
5. ChatBot
'''
def chatbot(text, func, n=5, original_lang="", translate_lang="", translate="", faq="", local_model=""):
    global content
    task_dic = return_task_dic(local_model)
    if func == 'chat':
        load_dic = chat_dic
        if original_lang=="":
            original_lang='繁體中文'
            
        content = task_dic['chat'].format(original_lang, text)
    elif func == 'v2':
        content = text
    elif func == 'gen':
        load_dic = gen_dic
        content = task_dic['gen'].format(n, text)
    elif func == 'translate':
        load_dic = translate_dic
        content = task_dic['translate'].format(text, translate_lang)
    elif func == 'translate_plus':
        load_dic = translate_dic
        content = task_dic['translate_plus'].format(text, translate_lang, judge_dic['judgebot_score'])
    elif func == 'multi_syn':
        load_dic = gen_dic
        content = task_dic['multi_syn'].format(n, text)
    elif func == 'judge_translate':
        load_dic = judge_dic
        content = task_dic['judge_translate'].format(text, translate)
    '''
    if 'ChatGPT' in load_dic['chatbot_memo']:
        from openai import OpenAI
        client = OpenAI(api_key=load_dic['chatbot_key'])
        
        history = [{"role": "assistant", "content": "聊天小幫手"},
                   {"role": "user", "content": content}
                   ]
        
        completion = client.chat.completions.create(model=load_dic['chatbot_name'], messages=history)
        return completion.choices[0].message.content
        
    elif "AI Translate - Gemini" in load_dic['chatbot_memo']:
        import google.generativeai as genai
        genai.configure(api_key=load_dic['chatbot_key'])
        model = genai.GenerativeModel(load_dic['chatbot_name'], system_instruction=load_dic['chatbot_role'])
        
        completion = model.generate_content(content)
        return completion.text
    '''
    if load_dic['chatbot_memo']=="4090":
        try:
            if local_model=="":
                try:
                    model = Ollama(model=load_dic['chatbot_name'], temperature=load_dic['chatbot_temperature'], num_predict=load_dic['chatbot_token'])
                except:
                    model = Ollama(model=load_dic['chatbot_name'])
                reranker = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=globals()[f'retriever_{load_dic["chatbot_name"]}'])
            else:
                try:
                    model = Ollama(model=local_model, temperature=load_dic['chatbot_temperature'], num_predict=load_dic['chatbot_token'])
                except:
                    model = Ollama(model=local_model)
                reranker = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=globals()[f'retriever_{local_model}'])
            
            prompt = PromptTemplate(template="""You are a multilingual assistant. Always answer in the same language as the question. 
                                    Use the context below to answer the question.:
                                        Context: {context}  
                                        Question: {question}
                                        Answer:""", input_variables=["context", "question"])
            
            rag_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever= reranker, chain_type_kwargs={"prompt": prompt})
            r = rag_chain.invoke(content)
            '''
            if lang=="":
            else:
                reranker = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=globals()[f'retriever_{lang}'])
                globals()[f'rag_chain_{lang}'] = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=reranker)
                r = globals()[f'rag_chain_{lang}'].invoke(content)
                '''
            return r['result']
        except:
            try:
                if local_model=="":
                    try:
                        model = Ollama(model=load_dic['chatbot_name'], temperature=load_dic['chatbot_temperature'], num_predict=load_dic['chatbot_token'])
                    except:
                        model = Ollama(model=load_dic['chatbot_name'])
                else:
                    try:
                        model = Ollama(model=local_model, temperature=load_dic['chatbot_temperature'], num_predict=load_dic['chatbot_token'])
                    except:
                        model = Ollama(model=local_model)
                r = model.invoke(content)
                return r
            except:
                return 'v2 chat setting error'
    elif load_dic['chatbot_memo']==None:
        return '聊天小幫手忙碌中'

'''
6. Algorithm
'''
def return_target_list(data, match, match_col, target_col, n=4):
    target = []
    for i in match:
        try:
            target.append(list(data.loc[(data[match_col]==i), target_col])[0])
            if len(target)==n:
                return target
        except:
            pass

def trans_float(n_list, r=3):
    return [round(i, r) for i in n_list]
    
def gen_old_label(text, lan=None):
    global now_faq, now_intent
    set_random_seed()    
    _, b_faq = get_predictions(now_faq, encode_Bert(tokenizer, [text], [""], maxlen_faq).batch(60))
    t_faq =b_faq[0]
    
    set_random_seed()
    _, b_intent = get_predictions(now_intent, encode_Bert(tokenizer, [text], [""], maxlen_intent).batch(60))
    t_intent =b_intent[0]
    
    '''
    lang = detect_lang(text)
    if lang not in list(set(train_faq['LanguageCode'])):
        lang = 'en'
        '''
    faq_answer = [dic_faq[i] for i in np.argsort(t_faq)[::-1]]
    
    if lan==None:      
        if hanzidentifier.identify(text)==0:
            lan = list(train_faq.loc[train_faq['Supply']==faq_answer[0], 'LanguageCode'])[0]
        elif hanzidentifier.identify(text)==1 or hanzidentifier.identify(text)==3 or hanzidentifier.identify(text)==4:
            if 'zh-TW' not in lan_list:
                lan = 'zh-CN'
            else:
                lan = 'zh-TW'
        elif hanzidentifier.identify(text)==2:
            if 'zh-CN' not in lan_list:
                lan = 'zh-TW'
            else:
                lan = 'zh-CN'
        
    faqid = return_target_list(globals()[f'train_faq_{lan}'], faq_answer, 'Supply', 'FaqSettingId')         
    similarity_faq = [float(i) for i in np.sort(t_faq)[::-1][:4]]
    referral = return_target_list(globals()[f'train_faq_{lan}'], faq_answer, 'Supply', 'Demand')
    
    ## intent
    intent = [dic_intent[i] for i in np.argsort(t_intent)[::-1]]
    
    ## intentid = return_target_list(train_intent, intent, 'IntentionName', 'OriginalIntentId', lang) 
    similarity_intent = [float(i) for i in np.sort(t_intent)[::-1][:4]]
    referral_intent = return_target_list(globals()[f'train_faq_{lan}'], faq_answer, 'Supply', 'IntentionName')
    
    if lan==None:
        pass
    else:
        faq_answer = return_target_list(globals()[f'train_faq_{lan}'], faqid, 'FaqSettingId', 'Supply')
        intent = return_target_list(globals()[f'train_faq_{lan}'], faqid, 'FaqSettingId', 'IntentionName')
    
    return referral_intent, intent[0], faqid[1:4], faqid[0], referral, similarity_faq, similarity_intent, faq_answer[0], lan
    
def gen_label(text):
    global now_faq, now_intent, lang
    set_random_seed()
    _, b_faq = get_predictions(now_faq, encode_Bert(tokenizer, [text], [""], maxlen_faq).batch(60))
    t_faq =b_faq[0]
    
    set_random_seed()
    _, b_intent = get_predictions(now_intent, encode_Bert(tokenizer, [text], [""], maxlen_intent).batch(60))
    t_intent =b_intent[0]
    
    lang = detect_lang(text)
    
    if lang not in list(set(train_faq['Lang'])):
        lang = 'en'
        
    faqid = [int(dic_faq[i]) for i in np.argsort(t_faq)[::-1][:4]]
    faq_similarity = [float(i) for i in np.sort(t_faq)[::-1][:4]]
    faq = [list(train_faq.loc[(train_faq['FaqSettingId'] == i) & (train_faq['Lang'] == lang), 'Demand'])[0] for i in faqid]
    faq_intent = [list(train_intent.loc[(train_intent['FaqSettingId'] == i) & (train_intent['Lang'] == lang), 'IntentionName'])[0] for i in faqid]
    faq_answer = list(train_faq.loc[(train_faq['FaqSettingId'] == faqid[0]) & (train_faq['Lang'] == lang), 'Supply'])[0]
    
    intentid = [int(dic_intent[i]) for i in np.argsort(t_intent)[::-1][:4]]
    intent = list(train_intent.loc[(train_intent['OriginalIntentId'] == intentid[0]) & (train_intent['Lang'] == lang), 'IntentionName'])[0]
    intent_similarity = [float(i) for i in np.sort(t_intent)[::-1][:4]]
    
    return faqid, faq_similarity, faq, faq_intent, faq_answer, intent, intent_similarity

def dscbot_reply(text, lan, mode, model=""):
    if mode=="dscbot" or mode=="v2" or mode=="rewrite":
        text = replace_synonyms(text, train_synonym['Text'], train_synonym['first_word'])
        text = replace_exclude(text, train_exclude['Text'])
        emotion = "無情緒"
        
        '''
        faqid, faq_similarity, faq, faq_intent, faq_answer, intent, intent_similarity = gen_label(text)
        faq_similarity, intent_similarity = trans_float(faq_similarity), trans_float(intent_similarity)
        '''
        try:
            Referral_intent, intent, Referral_faqid, faqid, Referral, Similarity_faq, Similarity_intent, faq_answer, lan = gen_old_label(text, lan)
        except:
            Referral_intent, intent, Referral_faqid, faqid, Referral, Similarity_faq, Similarity_intent, faq_answer, lan = gen_old_label(text, "en")
        Similarity_faq, Similarity_intent = trans_float(Similarity_faq), trans_float(Similarity_intent)
        
        faq_dict = {'Bot' : "FAQbot",
                    'Question' : Referral[0],
                    'Answer_faqsettingid' : faqid,
                    'Referral' : Referral[1:],
                    'Referral_faqsettingid' : Referral_faqid,
                    'Referral_intent' : Referral_intent[1:],
                    'Similarity' : Similarity_faq,
                    'emotion' : emotion,
                    'intent' : Referral_intent[0],
                    'type' : 1,
                    'end' : 1,
                    'max_prob' : Similarity_faq[0],
                    'pred_intent' : intent,
                    'pred_intent_prob' : Similarity_intent[0]
                    }
        
        error_dict = {'Bot' : "ChatBot",
                    'Question' : None,
                    'Answer_faqsettingid' : None,
                    'Referral' : None,
                    'Referral_faqsettingid' : None,
                    'Similarity' : None,
                    'emotion' : emotion,
                    'intent' : None,
                    'type' : 1,
                    'end' : 1,
                    'max_prob' : Similarity_faq[0],
                    'pred_intent' : intent,
                    'pred_intent_prob' : Similarity_intent[0]
                    }       
        
        if mode=="dscbot":
            try:
                if (float(Similarity_faq[0])<float(thresh_dic['SimilarityFAQBot'])) & (float(Similarity_intent[0])<float(thresh_dic['SimilarityIntentionBot'])):
                    if chat_dic['chatbot']==0:
                        if lan=='zh-TW':
                            error_dict['Answer'] ='很抱歉，您的問題不在我們的服務範圍內，我還能幫您做什麼?'
                        else:
                            error_dict['Answer'] = "I'm sorry, your question is not within the scope of our service. What else may I assist you?"
                        row_dict = error_dict
                    else:
                        if lan==None:
                            lang="繁體中文"
                        else:
                            lang=language_codes[lan]
                        error_dict['Answer'] = chatbot(text, 'chat', "", lang, "", "", "", model)
                        row_dict = error_dict
                else:
                    faq_dict['Answer'] = faq_answer
                    row_dict = faq_dict
            except:
                if lan=='zh-TW':
                    error_dict['Answer'] ='聊天小幫手忙碌中'
                else:
                    error_dict['Answer'] ="The chat assistant is currently busy"
                row_dict = error_dict
        elif mode=="v2" or mode=="rewrite":
            if mode=="rewrite":
                text = f"請將以下回答稍微改寫，{faq_answer}"
            faq_dict['Answer'] = chatbot(text, mode, "", "", "", "", "", model)
            row_dict = faq_dict
            
    elif mode=="chat":
        lang = lan
        r = chatbot(text, 'chat', "", lang, "", "", "", model)
        row_dict = {'Bot' : "ChatBot",
                    'Answer' : r,
                    'Question' : None,
                    'Answer_faqsettingid' : None,
                    'Referral' : None,
                    'Referral_faqsettingid' : None,
                    'Referral_intent' : None,
                    'Similarity' : None,
                    'emotion' : "無情緒",
                    'intent' : None,
                    'type' : 1,
                    'end' : 1,
                    'max_prob' : None,
                    'pred_intent' : None,
                    'pred_intent_prob' : None
                    }
    return row_dict

'''
7. Word Cloud
'''
from collections import Counter
from wordcloud import WordCloud
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def return_words(target, word_len, aws_ng=0.1, amount=10):
    stopwords = list(train_exclude['Text'])
    words = [word for sentence in [jieba.cut_for_search(text) for text in target] for word in sentence if word not in stopwords and len(word) == word_len]
    word_counts = Counter(words)
    top_words = [word for word, count in word_counts.most_common(amount)]
      
    return top_words

'''
8. Judge_translate
'''
def create_translation_df(func, fit_df, match_df, original_lang, translate_lang):
    if func=='faqsetting':
        match_factor = 'FaqSettingId'
        merged_df = pd.merge(fit_df, match_df, on=[match_factor, 'LanguageCode'], how='inner')
        merged_df = merged_df[['LanguageCode', 'FitId', match_factor, 'Demand', 'Supply']]
    elif func=='faq':
        match_factor = 'SynophaseDetailId'
        match_df = match_df.rename(columns={'Id': match_factor})
        merged_df = pd.merge(fit_df, match_df, on=[match_factor, 'LanguageCode'], how='inner')
        merged_df = merged_df[['LanguageCode', 'FitId', match_factor, 'Text']]
    merged_df = merged_df.drop_duplicates()
    
    # Use .loc with boolean indexing for clarity and to avoid the need for suffixes later
    original_lang_df = merged_df.loc[merged_df['LanguageCode'] == original_lang].copy()
    translate_lang_df = merged_df.loc[merged_df['LanguageCode'] == translate_lang].copy()

    # Merge based on FitId
    result_df = pd.merge(original_lang_df, translate_lang_df, on='FitId', how='inner', suffixes=('_original', '_translate'))

    if func=='faqsetting':
        result_df = result_df[[
            'FitId', 'FaqSettingId_translate', 'LanguageCode_original', 'Demand_original', 'Supply_original', 
            'LanguageCode_translate', 'Demand_translate', 'Supply_translate'
        ]].rename(columns={
            'FaqSettingId_translate': 'FaqSettingId',
            'LanguageCode_original': 'OriginalLang',
            'LanguageCode_translate': 'TranslateLang',
            'Demand_original': 'OriginalDemand',
            'Supply_original': 'OriginalSupply',
            'Demand_translate': 'TranslateDemand',
            'Supply_translate': 'TranslateSupply'
        })
        result_df["Original"] = result_df["OriginalDemand"] + result_df["OriginalSupply"]
        result_df["Translate"] = result_df["TranslateDemand"] + result_df["TranslateSupply"]
    elif func=='faq':
        result_df = result_df[[
            'FitId', 'SynophaseDetailId_translate', 'LanguageCode_original', 'Text_original', 'LanguageCode_translate', 'Text_translate'
        ]].rename(columns={
            'SynophaseDetailId_translate': 'SynophaseDetailId',
            'LanguageCode_original': 'OriginalLang',
            'LanguageCode_translate': 'TranslateLang',
            'Text_original': 'Original',
            'Text_translate': 'Translate'
        })
    return result_df

def judge_lang(original_text, translate_text):
    if detect_lang(original_text)==detect_lang(translate_text):
        return -1
    else:
        return 10

def judge_translate(func, fit_df, match_df, original_lang, translate_lang, env, model):    
    data = json.load(open(f'{mother_folder}/{bot_name}/config.json'))[env]
    process_df = create_translation_df(func, fit_df, match_df, original_lang, translate_lang)
    
    if func == 'faqsetting':
        match_factor = 'FaqSettingId'
        connect_table = data['recordFaqSetting']
    elif func == 'faq':
        match_factor = 'SynophaseDetailId'
        connect_table = data['recordFaq']
    
    cnxn = pyodbc.connect(f'DRIVER=ODBC Driver 17 for SQL Server;SERVER={server}; DATABASE={database};UID={username};PWD={password}')
    df = pd.read_sql(f"SELECT * FROM {connect_table}", cnxn)    
    df = pd.DataFrame(df, columns=[match_factor, "Original", "Translate", "State", "Similarity"])
    
    translation_lack = process_df[~process_df[match_factor].isin(df[match_factor])]
    
    merged = pd.merge(process_df, df, on=match_factor, suffixes=('_process', '_df'), how='inner')
    translation_diff = merged[
        (merged['Original_df'] != merged['Original_process']) |
        (merged['Translate_df'] != merged['Translate_process'])
    ].rename(columns={'Original_process': 'Original', 'Translate_process': 'Translate'})
    
    ## delete diff
    try:
        query = f"DELETE FROM {connect_table} WHERE {match_factor} IN ({', '.join(map(str, translation_diff[match_factor]))})"
        cursor = cnxn.cursor()
        
        cursor.execute(query)
        cnxn.commit()
        cnxn.close()
    except:
        print('No diff')
    
    translation_df = pd.concat([translation_lack, translation_diff], ignore_index=True)
    
    ## judge translate
    if len(translation_df)==0:
        return "無新資料，無需校稿"
    else:
        for i,ot,tt, ol, tl in tzip(translation_df[match_factor], translation_df['Original'], translation_df['Translate'], 
                                    translation_df['OriginalLang'], translation_df['TranslateLang']):
            judge = judge_lang(ot, tt)
            if judge==-1:
                result = [[i, ot, tt, ol, tl, judge, ""]]
            else:
                result = [[i, ot, tt, ol, tl, judge_lang(ot, tt), chatbot(ot, 'judge_translate', '', original_lang, translate_lang, tt, "", model)]]
    
            result = pd.DataFrame(result, columns=[match_factor, "Original", "Translate", "LanguageCode_O", "LanguageCode_T", "State", "Similarity"])
            result[[match_factor, 'State', 'Similarity']] = result[[match_factor, 'State', 'Similarity']].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            ## to sql
            connection_string = f"DRIVER=ODBC Driver 17 for SQL Server;SERVER={server};DATABASE={connect_table.split('.')[0].strip('[]')}; \
                UID={username};PWD={password}"
            engine = create_engine(f'mssql+pyodbc:///?odbc_connect={connection_string}')
            result.to_sql(connect_table.split('.')[-1].strip('[]'), con=engine, if_exists='append', index=False, dtype={'Original': NVARCHAR(2000),
                                                                                                                        'Translate': NVARCHAR(2000)})
        return f"Update {len(translation_df)} judges, in {connect_table}"

'''
9. API APP
'''    
import requests
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

app = FastAPI()
print("API:Ready to GO")

@app.get("/")
def hello():
    return 'Hello, I am Bot multi 20250409'

@app.get("/word_cloud")
def word_cloud():
    try:
        final = ' '.join([' '.join(return_words(train_faq.query("Lang=='zh-TW'")['Demand'], i)) for i in [2,3,4]])
        word = WordCloud(width=600, height=300, prefer_horizontal=1, collocations=False, min_font_size=10, colormap="tab10", font_path=font).generate(final)
        plt.figure(figsize=(20,10), facecolor='k')
        plt.imshow(word, interpolation ="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f'{mother_folder}/{bot_name}/Models/word.png', format='png', facecolor='k', bbox_inches='tight')
        plt.close()
        return f'Image saved in {mother_folder}/{bot_name}/Models/word.png'
    except:
        return 'updatefaq first'

@app.get("/updatefaq")
def updatefaq():
    start_time = datetime.now()
    load_sql_data(2, 0.8, 2, 0.8, environment)
    task_dic = return_task_dic()
    data_time = datetime.now()
    return [thresh_dic, chat_dic, judge_dic, translate_dic, gen_dic, task_dic, lan_list, start_time, data_time]

@app.get("/load_ollama")
def load_ollama():
    start_time = datetime.now()
    load_sql_data(2, 0.8, 2, 0.8, environment)
    load_chatbot()
    data_time = datetime.now()
    return [thresh_dic, chat_dic, judge_dic, lan_list, start_time, data_time]

@app.get("/updatemodel")
def train_model():
    global now_faq, now_intent, now_emo
    start_time = datetime.now()
    load_sql_data(2, 0.8, 2, 0.8, environment)
    data_time = datetime.now()
    vocab_size_faq, vocab_size_intent, x_train_faq, y_train_faq, x_train_intent, y_train_intent= build_dic()
    
    now_faq = run_bert('multi', tokenizer, XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(dic_faq)), 
                       'faq', 28, train_faq['Text'], train_faq.label_faq)
    faq_time = datetime.now()
    now_intent = run_bert('multi', tokenizer, XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(dic_intent)),
                          'intent', 16, train_faq['Text'], train_faq.label_intent)
    intent_time = datetime.now()
    gen_old_label('有問題')
    return f"Bot_multi train complete, lang:{lan_list}, start: {start_time}, data: {data_time}, FAQ: {faq_time}, intent: {intent_time}"

@app.get("/load_model")
def load_model():
    global now_faq, now_intent, now_emo
    start_time = datetime.now()
    load_sql_data(2, 0.8, 2, 0.8, environment)
    data_time = datetime.now()
    vocab_size_faq, vocab_size_intent, x_train_faq, y_train_faq, x_train_intent, y_train_intent= build_dic()
    
    temp = [f"lang:{lan_list}, start: {start_time}, data: {data_time}"]
    try:
        now_faq = run_bert('multi', tokenizer, XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(dic_faq)),
                           'faq', 15, train_faq['Text'], train_faq.label_faq, 2)
        print("Bot_multi faq model loaded.")
    except:
        print("Bot_multi faq model not exist, please train first.")
    
    try:
        now_intent = run_bert('multi', tokenizer, XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(dic_intent)), 
                              'intent', 8, train_faq['Text'], train_faq.label_intent, 2)
        print("Bot_multi intent model loaded.")
    except:
        print("Bot_multi intent model not exist, please train first.")
    gen_old_label('有問題')
    return temp

@app.get('/dscbot')
def dscbot_get(request: Request):
    global thresh_dic, row_dict
    text = request.query_params.get('text', '')
    lan = request.query_params.get('lan', None)
    
    row_dict = dscbot_reply(text, lan, "dscbot")
    return JSONResponse(content=row_dict, media_type="application/json")

@app.post('/dscbot')
async def dscbot_post(request:Request):
    global thresh_dic, row_dict
    data = await request.json()
    text = data.get('text', '')
    lan = data.get('lan', None)
    
    row_dict = dscbot_reply(text, lan, "dscbot")
    return JSONResponse(content=row_dict, media_type="application/json")

@app.get('/v2_dscbot')
def v2_dscbot_get(request: Request):
    global thresh_dic, row_dict
    text = request.query_params.get('text', '')
    model = request.query_params.get('model', '')
    lan = request.query_params.get('lan', None)
    
    row_dict = dscbot_reply(text, lan, "v2", model)
    return JSONResponse(content=row_dict, media_type="application/json")

@app.post('/v2_dscbot')
async def v2_dscbot_post(request:Request):
    global thresh_dic, row_dict
    data = await request.json()
    text = data.get('text', '')
    model = data.get('model', '')
    lan = data.get('lan', None)
    
    row_dict = dscbot_reply(text, lan, "v2", model)
    return JSONResponse(content=row_dict, media_type="application/json")

@app.get('/re_dscbot')
def re_dscbot_get(request: Request):
    global thresh_dic, row_dict
    text = request.query_params.get('text', '')
    model = request.query_params.get('model', '')
    lan = request.query_params.get('lan', None)
    
    row_dict = dscbot_reply(text, lan, "rewrite", model)
    return JSONResponse(content=row_dict, media_type="application/json")

@app.post('/re_dscbot')
async def re_dscbot_post(request:Request):
    global thresh_dic, row_dict
    data = await request.json()
    text = data.get('text', '')
    model = data.get('model', '')
    lan = data.get('lan', None)
    
    row_dict = dscbot_reply(text, lan, "rewrite", model)
    return JSONResponse(content=row_dict, media_type="application/json")

@app.get('/chat')
def chat(request: Request):
    global thresh_dic, row_dict
    text = request.query_params.get('text', '')
    model = request.query_params.get('model', '')
    lan = request.query_params.get('lan', 'English')
    try:
        lang = language_codes[lan]
    except:
        lang = "English"
        
    row_dict = dscbot_reply(text, lang, 'chat', model)
    return JSONResponse(content=row_dict, media_type="application/json")

@app.get('/gen')
def gen(request: Request):
    global thresh_dic, row_dict
    text = request.query_params.get('text', '')
    n = request.query_params.get('n', 5)
    model = request.query_params.get('model', '')
    
    try:
        r = chatbot(text, 'gen', n, "", "", "", "", model)
    except:
        r = '聊天小幫手忙碌中'
    return r

@app.get('/multi_syn')
def multi_syn(request: Request):
    global thresh_dic, row_dict
    faq = request.query_params.get('faq', '')
    text = request.query_params.get('text', '')
    n = request.query_params.get('n', 5)
    lan = request.query_params.get('lan', 'english')
    try:
        lang = language_codes[lan]
    except:
        pass
    model = request.query_params.get('model', '')
    
    try:
        r = chatbot(text, 'multi_syn', n, '', '', '', faq, model)
    except:
        r = '聊天小幫手忙碌中'
    return r

@app.get('/translate')
def translate(request: Request):
    global thresh_dic, row_dict
    text = request.query_params.get('text', '')
    original_lang = request.query_params.get('original_lang', '')
    translate_lang = request.query_params.get('translate_lang', '')
    '''
    try:
        lang = language_codes[lan]
    except:
        pass
    '''
    model = request.query_params.get('model', '')
    judge = int(request.query_params.get('judge', 0))
    
    try:
        if judge == 0:
            r = chatbot(text, 'translate', "", "", translate_lang, "", "", model)
        elif judge == 1:
            r = chatbot(text, 'translate_plus', "", "", translate_lang, "", "", model)
        return JSONResponse(content={"Answer": r}, media_type="application/json")
    except:
        return '聊天小幫手忙碌中'

@app.get('/judge_single')
def judge_single(request: Request):
    global thresh_dic, row_dict
    ot = request.query_params.get('original_text', '').lower()
    tt = request.query_params.get('translate_text', '').lower()
    original_lang = request.query_params.get('original_lang', '')
    translate_lang = request.query_params.get('translate_lang', '')
    
    model = request.query_params.get('model', '')
    
    if detect_lang(ot)== original_lang and detect_lang(tt)== translate_lang:
        try:
            return {'State': judge_lang(ot, tt), 
                    'Similarity': int(re.search(r"(\d+)/\d+", chatbot(ot, 'judge_translate', '',  "", "", tt, "", model)).group(1))}
        except Exception as error:
            return error
    else:
        return {'State': judge_lang(ot, tt), 
                'Similarity': 1}
    
@app.get('/judge_translate')
def judge_translate_sql(request: Request):
    global thresh_dic, row_dict
    func = request.query_params.get('func', '')
    original_lang = request.query_params.get('original_lang', '')
    translate_lang = request.query_params.get('translate_lang', '')
    env = request.query_params.get('env', '')
    
    model = request.query_params.get('model', '')
    
    load_sql_data(2, 0.8, 2, 0.8, env)
    try:
        if func=='faqsetting':
            return judge_translate('faqsetting', fit_faqsetting, train_faq, original_lang, translate_lang, env, model)
        elif func=='faq':
            return judge_translate('faq', fit_faq, match_faq, original_lang, translate_lang, env, model)
    except Exception as error:
        return error

@app.get('/lang_detect') 
def get_lang(request:Request):
    text = request.query_params.get('text', '')
    return detect_lang(text)


import uvicorn
if test:
    uvicorn.run(app, host='192.168.2.209', port=4000)
else:
    uvicorn.run(app, host=host_ip, port=port_ip)


