import argparse
import torch
from  Prompts  import get_prompt
import pandas as pd
from tqdm import tqdm
import warnings
import json
from typing import Any
from model import MORE_CL
import importlib
import texar.torch as tx
from langchain_openai import OpenAIEmbeddings
import time
import openai  # 确保安装了 openai 包
from openai import OpenAI
import os
import re
import csv
from sklearn.metrics import precision_score, recall_score,f1_score
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import faiss
from fuzzywuzzy import fuzz
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

###   API
api_key = "xxxxxxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
# base_url = os.environ.get("OPENAI_BASE_URL")
    )

def try_make_requests(question,model_name, stop_num,prompt):
    attempts = 0
    while attempts < stop_num:
        try:
            answers = make_requests(question,model_name,prompt).content
            return answers
        except openai.BadRequestError as e:
            attempts += 1
            print(f" {attempts}th Failure：{e}")
            if attempts == stop_num:
                print("Attempted twice but still failed, please check the input content.")
            time.sleep(1)
    return None

def make_requests(input,model_name,prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"input: {input}"}
        ],
        stream=False,
        temperature=0.1
    )
    answer =response.choices[0].message
    return answer
def evaluate_DAIC(list,answers,labels):
    count=0
    predicts = [1 if answers[i] == "DEPRESSION" else 0 for i in range(len(answers))]
    for i in range(len(predicts)):
        if predicts[i] != labels[i]:
            count+=1
            print(list[i],predicts[i],labels[i])
    print("Total number of error samples ：{0}".format(count))
    precision_depression= precision_score(labels, predicts,pos_label=1)
    precision_control = precision_score(labels, predicts, pos_label=0)
    f1_1 = f1_score(labels, predicts, pos_label=1)
    f1_0 = f1_score(labels, predicts, pos_label=0)
    macro_f1 = f1_score(labels, predicts, average='macro')
    print("\nTotal num:{0}, Current Precision_d:{1},Current Precision_c:{2},f1_1:{3},f1_0:{4},Macro_f1:{5}".format(
        len(labels), precision_depression, precision_control, f1_1, f1_0, macro_f1))

def extract_total_score(response):
    pattern1 = r"\*\*Total Score\*\*: (\d+)"
    pattern2 = r"Total Score\s*[:：]\s*(\d+)"
    # Find the total score using the regex pattern
    match1 = re.search(pattern1, response)
    match2 = re.search(pattern2, response)
    if match1:
        return match1.group(1)  # Return the total score as a string
    if match2:
        return match2.group(1)  # Return the total score as a string
    if match1 == None and match2 == None:
        return None  # Return None if no match is found
### use summary event to RAG
def ToM_Knowledge_Augmented_v2(index):
    kn_path_event_tuple = './Dataset/DAIC-WOZ-Major_event_summary/{0}_event-tuple-embed.csv'.format(index)
    output_file='./Dataset/DAIC-WOZ-Major_event_summary/{0}_event-tuple-embed-coke-evidences.csv'.format(index)
    event_summary_data = pd.read_csv(kn_path_event_tuple, low_memory=False)  #
    event_summary_df = pd.DataFrame(event_summary_data )
    ###  load ToM vector store
    index_store = faiss.read_index(f'index_{args.index_store}.index')
    Coke_dataset = './Dataset/COKE/train_thought.csv'
    coke_data = pd.read_csv(Coke_dataset,encoding="utf-8")
    coke_df = pd.DataFrame(coke_data)
    with open(kn_path_event_tuple, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        with open(output_file, mode='w+', newline='', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerow([ 'event', 'event_tuple','distance', 'situation', 'clue', 'thought', 'polarity'])
            for row in reader:
                embedding = torch.tensor(eval(row[2]))  # Tensor
                event = row[0]  #  'event'
                event_tuple = row[1]  #
                distances, indices = index_store.search(torch.tensor(embedding), args.M)
                for d, i in zip(distances[0], indices[0]):  # shape (1, N)
                    situation = coke_df.iloc[i]["situation"]
                    clue = coke_df.iloc[i]["clue"]
                    thought = coke_df.iloc[i]["thought"]
                    polarity = coke_df.iloc[i]["polarity"]
                    writer.writerow([event, event_tuple, d, situation, clue, thought, polarity])
    csv_data = pd.read_csv(output_file, low_memory=False)
    csv_df = pd.DataFrame(csv_data)
    all_examples = []
    for id in range(len(csv_df)):
        situation = csv_df.loc[id, "situation"]
        clue = csv_df.loc[id, "clue"]
        thought = csv_df.loc[id, "thought"]
        emotion = "positive" if   csv_df.loc[id, "thought"] else "negative"
        formatted_string = f"Example {id}:\nSituation: {situation} Clue:{clue} Thought: {thought} Emotion: {emotion}\n "
        all_examples.append(formatted_string)
    return all_examples
### use summary to RAG
def ToM_Knowledge_Augmented_v3(index):
    kn_path = './Retrieval_result/DAIC-ToM-k{0}/{1}_ToM_docs.txt'.format(args.M, index)
    kn_path_score = './Retrieval_result/DAIC-ToM-k{0}/{1}_ToM_docs_score.txt'.format(args.M, index)
    ###  check ToM dataset
    source_file = './Dataset/COKE/train_thought_combine.txt'
    source_path = os.path.join('./Dataset/COKE/train_thought.csv')
    if not os.path.exists(source_file):
        csv_data = pd.read_csv(source_path, low_memory=False)
        csv_df = pd.DataFrame(csv_data)
        content = csv_df[["situation", 'clue']].values
        str_content = ''
        for line in content:
            str_content += line[0] + line[1] + "\n"
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(str_content)
    else:
        str_content = open(source_file, 'r', encoding='utf-8').readlines()
    file = open(kn_path, 'r').readlines()
    ## ==================================few shots  Selection==================================
    total_shots = ""
    idx = 0
    str_content = open(source_file, 'r', encoding='utf-8').readlines()
    csv_data = pd.read_csv(source_path, low_memory=False)  # 防止弹出警告
    csv_df = pd.DataFrame(csv_data)
    simplify = set(file)
    simplify.discard('\n')
    for i in simplify:
        for (id, item) in enumerate(str_content):
            if i in item:
                idx += 1
                emotion = "positive\n" if csv_df[id + 1:id + 2]["polarity"].values[0] else "negative\n"
                total_shots += "Example:\nSituation: " + csv_df[id + 1:id + 2]["situation"].values[0] + " Clues: " + csv_df[id + 1:id + 2]["clue"].values[0] + " Thought: " + csv_df[id + 1:id + 2]["thought"].values[0] + " Emotion: " + emotion
    return total_shots
### use event summary --tuple -- embedding to RAG
def ToM_Knowledge_Augmented_final(index):
    Major_event_summary_path = './Dataset/DAIC-WOZ-Major_event_summary/{0}_summary.txt'.format(index)
    Major_event_tuple_path = './Dataset/DAIC-WOZ-Major_event_summary/{0}_tuple.json'.format(index)
    kn_path_event_tuple = './Dataset/DAIC-WOZ-Major_event_summary/{0}_event-tuple-embed.csv'.format(index)
    output_file='./Dataset/DAIC-WOZ-Major_event_summary/{0}_event-tuple-embed-coke-evidences.csv'.format(index)
    if not os.path.exists(output_file):
        total_summary = ""
        retrieved_docs_path = './Retrieval_result/DAIC-Personalized-k{0}/{1}_retrieval_com_docs.txt'.format( args.K, index)
        Evidences = open(retrieved_docs_path, 'r').readlines()
        clue_list = []
        temp_segment = []
        for item in Evidences :
            if item == "\n":
                if temp_segment:
                    combined = ''.join(temp_segment).strip()
                    if not combined.startswith("Query") and combined != "":
                        clue_list.append(combined)
                    temp_segment = []
            else:
                temp_segment.append(item)
        if temp_segment:
            combined = ''.join(temp_segment).strip()
            if not combined.startswith("Query") and combined != "":
                clue_list.append(combined)
        full_list = clue_list
        simplify = str(set(full_list))
        answer = try_make_requests(simplify, args.model_name, 2, prompt=get_prompt('major_event_prompt'))
        total_summary += str(answer) + "\n"
        if answer is None:
            print(f"\n=================================={index}'s  Major_event_summary is None ==================================")
        with open(Major_event_summary_path, 'w+', encoding='utf-8') as fw:
            fw.write(total_summary)
        ####  transfer into tuple
        Major_event_summary = open(Major_event_summary_path, 'r').readlines()
        event_tuples_dict = {}
        csv_filename=kn_path_event_tuple
        with open(csv_filename, mode="w+", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["event", "event_tuples","event_tuples_cls", "embedding", "embedding_mean", "embedding_var"])
            for idx, major_event in enumerate(Major_event_summary):
                major_event = re.sub(r'\d+\.\s*', '', major_event)
                event_tuples = try_make_requests(major_event, args.model_name, 2,prompt=get_prompt('event_extraction'))  ### evidence event tuple
                if event_tuples!="No":
                    for event_tuple in event_tuples.split("\n"):
                        if event_tuple!="":
                            event_tuples_dict[major_event] = event_tuple
                            event_tuple1 = re.sub(r'^\d+\.', "", event_tuple)  # Remove sequence numbers
                            event_tuple = re.sub(r',', '', event_tuple1)  # Remove commas
                            event_tuple = re.sub(r'[<>]', '', event_tuple)  # Remove angle brackets
                            event_tuple_cls = f"[CLS]{event_tuple} [SEP]"  # Add CLS SEP
                            length = len(event_tuple.split(" "))
                            ids = tokenizer.map_text_to_id(event_tuple)
                            embeddings, embeddings_mean, embeddings_var = model.encoder_q(torch.tensor(ids).unsqueeze(0).to(device),torch.tensor(length).unsqueeze(0).to(device))
                            writer.writerow([major_event, event_tuple1,event_tuple_cls, embeddings.tolist(), embeddings_mean.tolist(), embeddings_var.tolist()])
        with open(Major_event_tuple_path, 'w') as json_file:
            json.dump(event_tuples_dict, json_file)
    ### search part ###
        # event_summary_data = pd.read_csv(kn_path_event_tuple, low_memory=False)
        # event_summary_df = pd.DataFrame(event_summary_data )
        ###  load  ToM vector store
        index_store = faiss.read_index(f'index_{args.index_store}.index')
        ###  check COKE
        # Chunk_embed_path = './Chunk/size_500/{0}_chunk_event_embed.json'.format(index)
        Coke_dataset = './Dataset/COKE/train_thought.csv'
        coke_data = pd.read_csv(Coke_dataset,encoding="utf-8")
        coke_df = pd.DataFrame(coke_data)
        # with open(kn_path_event_tuple, 'r') as json_file:
        #         Chunk_embeddings = json.load(json_file)
        with open(kn_path_event_tuple, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)  #
            next(reader, None)
            with open(output_file, mode='w+', newline='', encoding='utf-8') as outputfile:
                writer = csv.writer(outputfile)
                writer.writerow([ 'event', 'event_tuple','distance', 'situation', 'clue', 'thought', 'polarity'])
                for row in reader:
                    #["event", "event_tuples","event_tuples_cls", "embedding", "embedding_mean", "embedding_var"]
                    embedding_mean = torch.tensor(eval(row[4]))
                    embedding= torch.tensor(eval(row[3]))
                    event = row[0]  #  'event'
                    event_tuple = row[1]
                    distances, indices = index_store.search(torch.tensor(embedding_mean).squeeze(0), args.M)
                    # index_raw = faiss.read_index(f'index_raw.index')
                    # distances, indices = index_store.search(torch.tensor(embedding), args.M)
                    for d, i in zip(distances[0], indices[0]):
                        situation = coke_df.iloc[i]["situation"]
                        clue = coke_df.iloc[i]["clue"]
                        thought = coke_df.iloc[i]["thought"]
                        polarity = coke_df.iloc[i]["polarity"]
                        writer.writerow([event, event_tuple, d, situation, clue, thought, polarity])
    ### construct ToM samples
    csv_data = pd.read_csv(output_file, low_memory=False)
    csv_df = pd.DataFrame(csv_data)
    all_examples = []
    for id in range(len(csv_df)):
        situation = csv_df.loc[id, "situation"]
        clue = csv_df.loc[id, "clue"]
        thought = csv_df.loc[id, "thought"]
        emotion = "positive" if  csv_df.loc[id, "thought"] else "negative"
        formatted_string = f"Example {id}:\nSituation: {situation} Clue:{clue} Thought: {thought} Emotion: {emotion}\n "
        all_examples.append(formatted_string)
    return all_examples

### ----------------------------------  ###
def parse_args():
    parser = argparse.ArgumentParser()
    # Adding arguments for each of the variables
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-08-06", help="Name of the model to be used.")
    parser.add_argument("--retry_num", type=int, default=2, help="Number of retries for the task.")
    parser.add_argument("--iteration", type=int, default=3, help="Number of iterations for the task.")
    parser.add_argument("--K", type=int, default=10, help="Number of top K documents to retrieve.")
    parser.add_argument("--M", type=int, default=2, help="Number of documents to select from the retrieved K.")
    parser.add_argument("--flag", type=int, default=0, help="Control flag for task behavior.")
    parser.add_argument("--mode", type=str, default="ToM_v4", choices=["ToM_v2", "ToM_v3", "ToM_v4"], help="Mode for the task.")
    parser.add_argument("--shots", type=str, default="few", choices=["few", "zero"], help="Few-shot or zero-shot learning mode.")
    parser.add_argument("--restart", type=bool, default=True, help="Flag to restart the process.")
    parser.add_argument("--show", type=bool, default=False, help="Flag to show additional output or not.")
    parser.add_argument("--out_filename", type=str, default="./results/{model_name}_{shots}without_selection_{mode}_{K}_{shots}_RAG_output_{shots}_check_final_2.txt", help="Output file name (with experiment data).")
    parser.add_argument("--out_filename_without_exp", type=str, default="./results/{model_name}_{shots}without_selection_{mode}_{K}_{shots}_RAG_output_{shots}_wo_exp_check_final_2.txt", help="Output file name (without experiment data).")
    parser.add_argument('--config-model',type=str,default="config_model", help="The model config.")
    parser.add_argument('--embedding_dim', type=int, default="500", help="embedding_dim for index .")
    parser.add_argument("--rag_type", type=str, choices=["Iterative_Personalized","Disposable_Personalized","Disposable_Naive"],default="Iterative_Personalized", help="Choose the type of RAG model to use: Personalized, Naive RAG, or ToM selection.") # Default is Personalized, you can change it
    parser.add_argument('--ToM', type=bool, default=True, help="decide whether to use ToM selection.")
    parser.add_argument("--fuzzy_match", type=bool, default=False,help="Enable or disable fuzzy matching.")
    parser.add_argument("--logs", type=bool, default=True, help="help record the details of the experiment.")
    parser.add_argument("--index_store", type=str, default="raw", help="choose what kind of index store to use.")
    return parser.parse_args()

def Evidence_RAG(match_list,index):  #
        if args.rag_type == 'Personalized':
            retrieved_docs_path = './Retrieval_result/DAIC-Personalized-k{0}/{1}_retrieval_com_docs.txt'.format(args.K,index)
            query_path=r"./Dataset/Personalized_information/{0}_retrieval_per_query.txt".format(index)
        else:
            retrieved_docs_path = './Retrieval_result/DAIC-k{0}/{1}_retrieval_com_docs.txt'.format(args.K, index)
            query_path =r'./common_query.txt'
        if not os.path.exists(retrieved_docs_path):
            path = os.path.join(prefix, "Dataset/DAIC-WOZ_Pre" + '/{0}_TRANSCRIPT.csv'.format(index))
            file = open(path, 'r', encoding='utf-8').readlines()
            content = ''
            for line in file:
                content += line
            ### retriever
            texts = text_splitter.create_documents([content])
            KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                texts, embeddings, distance_strategy=DistanceStrategy.COSINE
            )
            if args.rag_type == 'Personalized':
                if os.path.exists(query_path):  ### check whether personalized_querys exists
                    querys = open(query_path, 'r').readlines()
                else:
                    personalized_information_summary_path= r"./Dataset/Personalized_information/{0}_retrieval_per_inf.txt".format(index)
                    if os.path.exists(personalized_information_summary_path):  ### check whether personalized_information exists
                        per_information = open(personalized_information_summary_path, 'r').readlines()
                    else:
                        answer = try_make_requests(file, args.model_name, 2, prompt=get_prompt('personal_information_generate'))
                        with open(personalized_information_summary_path, 'w') as f:
                            f.write(answer)
                        per_information = open(personalized_information_summary_path, 'r').readlines()
                    answer = try_make_requests(per_information, args.model_name, 2, prompt=get_prompt('personal_query_generate'))
                    personal_query_path=r"./Dataset/Personalized_information/{0}_retrieval_per_query.txt".format(index)
                    with open(personal_query_path, 'w') as f:
                        f.write(answer)
                    querys = open(personal_query_path, 'r').readlines()
            else:
                querys = open(query_path, 'r').readlines() ### common querys
                
            ### retrieval    
            total_results = ""
            for query in querys:
                retrieved_results = ""
                idx = 0
                retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=query, k=args.K)
                for i in retrieved_docs:
                    retrieved_results += ('\nClue {0}:\n'.format(idx) + i.page_content)
                    idx += 1
                total_results += f"\nQuery: {query}" + "{" + retrieved_results + "\n}\n"
            with open(retrieved_docs_path, 'w') as f:
                f.write(total_results)
            evidences = open(retrieved_docs_path, 'r').readlines()
        else:
            evidences = open(retrieved_docs_path, 'r').readlines()
        return evidences
def generate_unique_filename(filename):
    base, ext = os.path.splitext(filename)  #
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"  #
        counter += 1
    return new_filename

def query_gengerate_llm(query,evidences):
    """generate new query based on owned information"""
    if evidences !=None:
       input=  "**Existing Query**: "+f'{query}\n' +"**Evidences**:" f'{evidences}'
       query  =  try_make_requests(input,args.model_name, 2, prompt=get_prompt('Iterative_query_generate'))
    return query
def check_evidence_sufficient(symptom,evidences):
    """check whether evidences are sufficient"""
    input = "**Symptom**: " + f'{symptom}\n' + "**Evidences**:" f'{evidences}'
    response=try_make_requests(input, args.model_name, 2, prompt=get_prompt('check_evidence_sufficient'))
    return response
def get_new_evidence(retrieved_docs, existing_evidences):
    for doc in retrieved_docs:
        if doc not in existing_evidences:
            return doc
        else:
            pass
def Evidence_Iterative_Personal_RAG(index):
    retrieved_docs_path = './Retrieval_result/DAIC-Iterative-Personalized-k{0}/{1}_retrieval_com_docs.txt'.format(args.K, index)
    retrieval_process_path = './Retrieval_result/DAIC-Iterative-Personalized-k{0}/{1}_retrieval_process.json'.format(args.K, index)
    # check whether retrieval docs is ready  default k=10
    if not os.path.exists(retrieved_docs_path):
        path = os.path.join(prefix, "Dataset/DAIC-WOZ_Pre" + '/{0}_TRANSCRIPT.csv'.format(index))
        file = open(path, 'r', encoding='utf-8').readlines()
        content = ''
        for line in file:
            content += line
        ### retriever
        texts = text_splitter.create_documents([content])
        KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            texts, embeddings, distance_strategy=DistanceStrategy.COSINE
        )
        common_querys_path=r'./common_query.txt'
        common_querys=open(common_querys_path, encoding='utf-8').readlines()
        symptoms= ["Little interest or pleasure in doing things","Feeling down, depressed, or hopeless","Trouble falling or staying asleep, or sleeping too much","Feeling tired or having little energy", "Changes in appetite",
                                         "Feeling bad about yourself", "Trouble concentrating",
                                         "Moving or speaking slowly or being fidgety/restless"]
        total_process_log ={}
        with open(retrieved_docs_path, "a") as evidence_file:
            for idx,query in enumerate(common_querys):  ### eight aspects
                evidence_file.write("Query: "+query+ "\n")
                retrieved_results = []
                max_retries=args.K
                current_query = query
                retry_count = 0
                total_process_log[idx] = []  # List to hold the process logs for each step
                while retry_count < max_retries:
                    # Step 1: Perform search based on current query (simulating with a placeholder)
                    print(f"Searching for: {current_query}")
                    # Here you would perform the actual search and retrieve evidence
                    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=query,k=args.K)###only collect the first one
                    retrieved_docs=[i.page_content for i in retrieved_docs]
                    evidence_collected=get_new_evidence(retrieved_docs, retrieved_results)
                    retrieved_results.append(evidence_collected)
                    evidence_file.write(evidence_collected + "\n\n")
                    result = check_evidence_sufficient(symptoms[idx],retrieved_results)
                    # Save the intermediate variables in the process log
                    total_process_log[idx] .append({
                        "query": current_query,
                        "evidence": evidence_collected,
                        "sufficiency_result": result,
                        "retry_count": retry_count
                    })
                    # Step 3: If evidence is sufficient, break the loop, otherwise generate a new query
                    if result == "Yes":
                        print("Evidence is sufficient. Assessment can be made.")
                        break
                    else:
                        print("Evidence is insufficient. Generating new query.")
                        current_query = query_gengerate_llm(current_query, evidence_collected)
                        retry_count += 1
                        time.sleep(2)  # Simulate time between queries
                        if retry_count >= max_retries:
                            print("Reached maximum number of retries. Cannot gather sufficient evidence.")
                            break
                    # Save the final process log to a JSON file for later review
        with open(retrieval_process_path, 'w') as f_process:
            json.dump(total_process_log, f_process, indent=4)
        evidences = open(retrieved_docs_path, 'r').readlines()
    else:
        evidences = open(retrieved_docs_path, 'r').readlines()
    return evidences
def DAIC(mode,start_idx,index_list):
    predictions=[]
    scores = []
    if mode:
        with (open(out_filename, 'a+') as fw):
            with open(out_filename_res_only, 'a+') as ff:
                with open(out_filename_score_only, 'a+') as fs:
                    for index in tqdm(index_list[start_idx:],total=len(index_list[start_idx:])):
                        start_time = time.time()
                        ## retrieved_evidences
                        evidences =Evidence_Iterative_Personal_RAG(index)
                        #  preliminary_assessment
                        preliminary_assessment_path ='./results/Tempanswer/{0}_answers.txt'.format(index)
                        flag1 = 1  # control
                        flag2 = 1  # control
                        while (flag1 and flag2 ):
                            ### step1  preliminary_assessment
                            if not os.path.exists(preliminary_assessment_path):
                                first_answer = try_make_requests(evidences, args.model_name, 2, prompt=get_prompt('preliminary_assessment'))
                                with open(preliminary_assessment_path, 'w', encoding='utf-8') as fir:  # save
                                    fir.write(first_answer)
                            else:
                                first_answer = str(open(preliminary_assessment_path, 'r').readlines())
                            flag1 = 0
                            first_analysis = first_answer.split("**Step 2 Output:**")[0].split("**Step 1 Output:**")[1]
                            ## step2  ToM Sample RAG
                            ### fuzzy matching
                            match_list = []
                            match_set = ["Little interest or pleasure in doing things",
                                         "Feeling down, depressed, or hopeless",
                                         "Trouble falling or staying asleep, or sleeping too much",
                                         "Feeling tired or having little energy", "Changes in appetite",
                                         "Feeling bad about yourself", "Trouble concentrating",
                                         "Moving or speaking slowly or being fidgety/restless"]
                            if args.fuzzy_match == True:
                                rag_sample = first_answer.split("**Step 2 Output:**")[1]
                                rag_sample = rag_sample.split(",")
                                for ele in rag_sample:  ### match
                                    most_similar_name = max(match_set, key=lambda x: fuzz.ratio(x, ele), default=None)
                                    match_list.append(match_set.index(most_similar_name))
                            else:
                                match_list= match_set  ## full
                            if args.ToM == True:
                                # get ToM samples
                                ToM_Sample = ToM_Knowledge_Augmented_v3( index)
                                # summary of the whole transcript
                                Major_event_path = './Dataset/DAIC-WOZ-paraphrase_event_extraction-simplify/{0}_summary.txt'.format(index)
                                Major_event = open(Major_event_path, 'r').readlines()
                                # In-context generation  few shots
                                input = "Preliminary symptom assessment result:\n" + str(first_analysis) + "Major_event about the participant:\n" + str(Major_event)\
                                + "\n**For reference, here are some samples that might guide your reasoning:**\n" + str(ToM_Sample)
                                prompt = get_prompt('ToM_prompt')
                            else:  ## zero shot
                                input = evidences
                                prompt = get_prompt('zero_shot_prompt')
                            ## step3  re-answer
                            answer = try_make_requests(input,args.model_name, 2, prompt=prompt)
                            # Extract answer
                            if ("DEPRESSION" in answer or "Depression" in answer) and ("CONTROL" not in answer) and ("Control" not in answer):
                                temp_result = "DEPRESSION"
                                flag1 = 0
                            elif ("CONTROL" in answer or "Control" in answer) and ("DEPRESSION" not in answer) and ("Depression" not in answer):
                                temp_result = "CONTROL"
                                flag1 = 0
                            else:
                                print(f"\n=================================={index} answer is not formal ==================================")
                                print(answer)

                            score = extract_total_score(answer)
                            if score != None and int(score) in [i for i in range(0, 25)]:
                                scores.append(score)
                            else:
                                flag2 = 0
                            if args.logs:
                                logs_path = './logs/{0}_log.txt'.format(index)
                                if os.path.exists(logs_path):
                                    os.remove(logs_path)
                                #Foramt
                                if args.rag_type :
                                    log = "\n==================prompt=======================\n" + prompt \
                                          + "\n===================input======================\n" + str(input) \
                                          + "\n===================first_analysis======================\n" + str(first_analysis) \
                                          + "\n===================Samples======================\n" + str(ToM_Sample) \
                                          + "\n==================answer=======================\n" + answer
                                else:
                                    log = "\n==================prompt=======================\n" + prompt \
                                          + "\n===================input======================\n" + str(input) \
                                          + "\n==================answer=======================\n" + answer
                                with open(logs_path, 'w', encoding='utf-8') as log_file:
                                    log_file.write(log)
                        predictions.append(temp_result)
                        fw.write(str(index)+str(answer))
                        fw.write("\n\n")
                        fs.write(str(index) + " " + str(score))
                        fs.write("\n")
                        ff.write(str(index) + str(temp_result))
                        ff.write("\n")
                        print("\n=================================={0} Time cost:{1}==================================".format(index,(time.time()-start_time)))




    if len (predictions)!= len(index_list):
        predictions=[]
        f=open(out_filename_res_only,'r').readlines()
        if len(index_list) ==35:
            for i in f[0:len(index_list)]:
                predictions.append(i[3:-1])
        if len(index_list) ==107:
            for i in f[-len(index_list):]:
                predictions.append(i[3:-1])
    return predictions


if __name__ == '__main__':
    args = parse_args()
    dev_answers=[]
    train_answers = []
    warnings.filterwarnings("ignore")

    prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
    MARKDOWN_SEPARATORS = [
        "[Ellie]",
    ]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=MARKDOWN_SEPARATORS,
    )

    out_filename = r"./results/" + args.model_name +"_"+"_{0}_{1}_{2}_RAG_{3}_iterative.txt".format(args.mode,args.K,args.shots,"Personalized")
    # out_filename_res_only = r"./results/" + args.model_name + "_" + "_{0}_{1}_{3}_RAG_output_{2}_wo_exp_final_iterative_version_v4.txt".format(args.mode, args.K,args.shots,"Personalized")
    out_filename_res_only = r"./results/" + args.model_name + "_" + "_{0}_{1}_{2}_RAG_{3}_wo_exp_iterative.txt".format(args.mode, args.K,args.shots,"Personalized")
    out_filename_score_only = r"./results/" + args.model_name + "_" + args.shots + "_{0}_{1}_RAG_output_{2}_socre_s11_final_iterative_version_v4.txt".format(args.mode, args.K,args.shots)


    openai.api_key = api_key
    # Output the parsed arguments for debugging
    print("\n====================================================================")
    print(f"Model Name: {args.model_name}")
    print(f"RAG Type: {args.rag_type}")
    print(f"Number of Retrieved Docs (K): {args.K}")
    print(f"Whether to use ToM samples: {args.ToM}")
    print(f"Out_filename: {out_filename}")
    print(f"Whether start from zero: {args.restart}")
    print("\n====================================================================")

    user_querys=[]
    with open('./common_query.txt',"r") as f:
        for line in f.readlines():
            user_querys.append(line)
    with open('./dev_label_DAIC.txt', 'r') as f:
        temp = eval(f.readlines()[0])
        dev_list = list(temp.keys())
        dev_labels=list(temp.values())
    with open('./train_label_DAIC.txt', 'r') as f:
        temp = eval(f.readlines()[0])
        train_list = list(temp.keys())
        train_labels=list(temp.values())

    if args.restart:
        if os.path.exists(out_filename):
            os.remove(out_filename)
        if os.path.exists(out_filename_res_only):
            os.remove(out_filename_res_only)
        if os.path.exists(out_filename_score_only):
            os.remove(out_filename_score_only)
        train_idx = 0
        dev_idx = 0
        dev = True
        train = True
    else:
        f = open(out_filename_res_only, "r").readlines()
        index = int(f[-1][0:3])
        if index in dev_list:
            dev_idx = dev_list.index(index)+1
            train_idx = 0
            dev = True
            train = True
        else:
            dev_idx = len(dev_list)
            train_idx = train_list.index(index)+1
            dev = False
            train = True
    tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
    pad_token_id = tokenizer.map_token_to_id(tokenizer.pad_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_model: Any = importlib.import_module('config_model')
    config_data: Any = importlib.import_module('config_data')
    model = MORE_CL(config_model=config_model, config_data=config_data)
    model.to(device)
    # f_relation = Func_Relation_Attention(config_model.gauss_dim)
    # f_relation.to(device)
    ckpt = torch.load('best_model/checkpoint_best.pt')
    model.load_state_dict(ckpt['model'])

    ### dev dataset
    dev_predictions = DAIC(dev,dev_idx,dev_list)
    evaluate_DAIC(dev_list,dev_predictions, dev_labels)

    ### train dataset
    train_predictions= DAIC(train, train_idx, train_list)
    evaluate_DAIC(train_list,train_predictions, train_labels)
    labels = dev_labels + train_labels
    evaluate_DAIC(dev_list+train_list,dev_predictions+train_predictions, labels)