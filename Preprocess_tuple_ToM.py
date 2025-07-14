import texar.torch as tx
import argparse
import importlib
import torch
import csv
from  Prompts  import get_prompt
import pandas as pd
from tqdm import tqdm
import json
import time
import openai
from openai import OpenAI
import os
import re
from typing import Any
from model import MORE_CL
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss

###  API
api_key = "*****************************************"
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
def try_make_requests(question,model_name, stop_num,prompt):
    attempts = 0
    while attempts < stop_num:
        try:
            answers = make_requests(question,model_name,prompt).content
            return answers
        except openai.BadRequestError as e:
            attempts += 1
            print(f"第 {attempts} 次失败：{e}")
            if attempts == stop_num:
                print("重试两次仍然失败，请检查输入内容。")
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

### -------------parameter------------------  ###
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
    parser.add_argument("--out_filename", type=str, default="./results/{model_name}_{shots}without_selection_{mode}_{K}_{shots}_RAG_output_{shots}_check_final.txt", help="Output file name.")
    parser.add_argument("--out_filename_without_exp", type=str, default="./results/{model_name}_{shots}without_selection_{mode}_{K}_{shots}_RAG_output_{shots}_wo_exp_check_final.txt", help="Output file name.")
    parser.add_argument('--config-model',type=str,default="config_model", help="The model config.")
    parser.add_argument('--embedding_dim', type=int, default="500", help="embedding_dim for index .")
    parser.add_argument("--rag_type", type=str,choices=["Iterative_Personalized","Disposable_Personalized","Disposable_Naive"],default="Iterative_Personalized",help="Choose the type of RAG model to use: Personalized, Naive RAG, or ToM selection.") # Default is Personalized, you can change it
    parser.add_argument('--ToM', type=bool, default=False, help="decide whether to use ToM selection.")
    parser.add_argument("--fuzzy_match", type=bool, default=False,help="Enable or disable fuzzy matching.")
    parser.add_argument("--logs", type=bool, default=True, help="help record the details of the experiment.")
    parser.add_argument("--index_store", type=str, default="mean", help="choose what kind of index store to use.")
    return parser.parse_args()

def CHUNK(index):     #### chunk into event tuple
    MARKDOWN_SEPARATORS = [
        "[Ellie]",
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=MARKDOWN_SEPARATORS,
    )
    ##### chunk check ####
    Chunk_path = './Chunk/size_500/{0}_chunk.txt'.format(index)  ## transcription -- chunk
    Chunk_event_path = './Chunk/size_500/{0}_chunk_event.json'.format(index) ## chunk -- tuple

    if os.path.exists(Chunk_event_path):
        with open(Chunk_event_path, 'r') as jso:
            Chunk_event = json.load(jso)
    else:
        docs_path = './Dataset/DAIC-WOZ_Pre' + '/{0}_TRANSCRIPT.csv'.format(index)
        dialogues = open(docs_path, 'r').readlines()  ##dialogues
        content = ''
        for line in dialogues:
            content += line
        chunks = text_splitter.create_documents([content])
        chunk_event_dict={}
        chunk_event_list = []
        for idx,chunk in enumerate(chunks):
            chunk_event = try_make_requests(chunk.page_content, args.model_name, 2, prompt=get_prompt('event_extraction'))  ### evidence event tuple
            chunk_event_list.append(chunk.page_content)
            chunk_event_dict[f"{idx}"]=chunk_event

        # save chunk list type
        with open(Chunk_path, 'w', encoding='utf-8') as f_chunks:
        # Iterate over each element in the chunk_event_list and write it to the file
            for event in chunk_event_list:
                f_chunks.write(str(event) + '\n')
        # save chunk_event  dict type
        with open(Chunk_event_path, "w") as json_file:
            json.dump(chunk_event_dict, json_file)

    return Chunk_event


def Major_event_tuple(index):     #### evidences into summary into event tuple
    Major_event_summary_path = './Dataset/DAIC-WOZ-Major_event_summary/{0}_summary.txt'.format(index)
    Major_event_tuple_path = './Dataset/DAIC-WOZ-Major_event_summary/{0}_tuple.json'.format(index)

    if not os.path.exists('./Dataset/DAIC-WOZ-Major_event_summary/{0}_summary.txt'): #Major_event_summary_path
        # check  whether Major_event_summary exists
        if  os.path.exists(Major_event_summary_path):
            Major_event_summary = open(Major_event_summary_path, 'r').readlines()
        else:
            total_summary = ""
            if  args.rag_type == "Disposable_Personalized":
                retrieved_docs_path = './Retrieval_result/DAIC-Personalized-k{0}/{1}_retrieval_com_docs.txt'.format(args.K,index)
                dialogue_evidences = open(retrieved_docs_path, 'r').readlines()  ##evidences_dialogue
                query = str(dialogue_evidences).split("Query: ")[1:]
                full_list = []
                s = ""
                for i in range(args.K):
                    if i != (args.K - 1):
                        s += "Clue {0}:|".format(i)
                    else:
                        s += "Clue {0}:".format(i)
                for clues in query:
                    full_list += (re.split(s, clues)[1:])
                full_list = [re.sub(r'[\'\"{}]+', "", i) for i in full_list]
                full_list = [re.sub(r'(  )', "", i) for i in full_list]
            elif args.rag_type == "Iterative_Personalized":
                retrieved_docs_path = './Retrieval_result/DAIC-Iterative-Personalized-k{0}/{1}_retrieval_com_docs.txt'.format(args.K, index)
                Evidences = open(retrieved_docs_path, 'r')
                #### deal with
                segments = []
                temp_segment = []
                # 遍历数据列表，分隔并拼接内容
                for item in Evidences:
                    if item == "\n":
                        if temp_segment:
                            segments.append(''.join(temp_segment))  # 将收集到的内容拼接成一段，添加到segments
                            temp_segment = []  # 重置临时段
                    else:
                        temp_segment.append(item)  # 如果不是 "\n"，则把内容添加到临时段中
                # 如果最后一段没有被添加（避免末尾无 "\n" 的情况）
                if temp_segment:
                    segments.append(''.join(temp_segment))
                full_list = segments[1:]
            else:
                retrieved_docs_path = './Retrieval_result/DAIC-k{0}/{1}_retrieval_com_docs.txt'.format(args.K, index)
                dialogue_evidences = open(retrieved_docs_path, 'r').readlines()  ##evidences_dialogue
                query = str(dialogue_evidences).split("Query: ")[1:]
                full_list = []
                s = ""
                for i in range(args.K):
                    if i != (args.K - 1):
                        s += "Clue {0}:|".format(i)
                    else:
                        s += "Clue {0}:".format(i)
                for clues in query:
                    full_list += (re.split(s, clues)[1:])
                full_list = [re.sub(r'[\'\"{}]+', "", i) for i in full_list]
                full_list = [re.sub(r'(  )', "", i) for i in full_list]

            simplify = str(set(full_list))
            answer = try_make_requests(simplify, args.model_name, 2, prompt=get_prompt('major_event_prompt'))
            total_summary += str(answer) + "\n"
            if answer is None:
                print(
                    f"\n=================================={index}'s  Major_event_summary is None ==================================")
            with open( Major_event_summary_path, 'w+', encoding='utf-8') as fw:
                fw.write(total_summary)
            Major_event_summary = open( Major_event_summary_path, 'r').readlines()

        event_tuples_dict={}
        csv_filename = './Dataset/DAIC-WOZ-Major_event_summary/{0}_event-tuple-embed.csv'.format(index)
        with open(csv_filename, mode="w+", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(["event", "event_tuples", "embedding", "embedding_mean", "embedding_var"])
            for idx, major_event in enumerate(Major_event_summary):
                major_event = re.sub(r'\d+\.\s*', '',major_event)
                event_tuples = try_make_requests(major_event, args.model_name, 2,prompt=get_prompt('event_extraction'))  ### evidence event tuple
                event_tuples_dict[major_event] = event_tuples

                event_tuples = re.sub(r'^\d+\.', "", event_tuples)   # Remove sequence numbers
                event_tuples = re.sub(r',', '', event_tuples)  # Remove commas
                event_tuples = re.sub(r'[<>]', '', event_tuples) # Remove angle brackets
                event_tuples = f"[CLS]{event_tuples} [SEP]"  # Add CLS SEP
                length = len(event_tuples.split(" "))
                ids = tokenizer.map_text_to_id(event_tuples)
                embeddings, embeddings_mean, embeddings_var = model.encoder_q(torch.tensor(ids).unsqueeze(0).to(device),torch.tensor(length).unsqueeze(0).to(device))

                writer.writerow([major_event, event_tuples, embeddings.tolist(),embeddings_mean.tolist(),embeddings_var.tolist()])

        with open(Major_event_tuple_path, 'w') as json_file:
            json.dump(event_tuples_dict, json_file)




def REFORMAT(evt):
    # evt ='<cls> ' + ' <sep> '.join(evt) + ' <sep>'
    # evt = [evt[1], evt[0], evt[2]]
    # evt = ' '.join(evt)
    evt_length=len(evt.split(" "))
    evt = evt.replace(",", "")
    evt = re.sub(r'\d+\.\s*', '', evt)
    return f"[CLS] {evt} [SEP]" ,evt_length+2
def ENCODER(index ):
    Chunk_embed_path = './Chunk/size_500/{0}_chunk_event_embed.json'.format(index)
    Chunk_event_path = './Chunk/size_500/{0}_chunk_event.json'.format(index)

    ### check chunk embedding
    if os.path.exists(Chunk_embed_path):
        with open(Chunk_embed_path, 'r') as js:
            Chunks_embed = json.load(js)
    else:
        Chunks_embed={}
        ### format
        with open(Chunk_event_path, 'r') as json_file:
            Chunk_event = json.load(json_file)

        for idx in Chunk_event.keys():
            chunk_e=Chunk_event[idx].split('\n')
            chunk_e = [re.sub(r'^\d+\.', "", i) for i in chunk_e] # Remove sequence numbers
            chunk_e = [re.sub(r',', '', i) for i in chunk_e] # Remove commas
            chunk_e = [re.sub(r'[<>]', '', i) for i in chunk_e] # Remove angle brackets
            chunk_e = [f"[CLS]{i} [SEP]" for i in chunk_e]  # Add CLS SEP
            chunk_e_length = [len(i.split(" ")) for i in chunk_e]  # Add CLS SEP
            # Chunk_event[idx]=chunk_e

            for id in range(len(chunk_e)):
                ids=  tokenizer.map_text_to_id(chunk_e[id])
                embeddings, embeddings_mean, embeddings_var  = model.encoder_q(torch.tensor(ids).unsqueeze(0).to(device),torch.tensor(chunk_e_length[id]).unsqueeze(0).to(device))
                Chunks_embed[chunk_e[id]] = embeddings_mean.tolist()
        with open(Chunk_embed_path, 'w') as json_file:
            json.dump(Chunks_embed, json_file)

    return Chunks_embed

def Preprocess_DAIC(args,start_idx,index_list):
    """
    1. Personal_RAG evidence event tuple
    2.

    """
    if args.mode:
        with (open(args.out_filename, 'a+') as fw):
            with open(args.out_filename_without_exp, 'a+') as ff:
                for index in tqdm(index_list[start_idx:],total=len(index_list[start_idx:])):
                    start_time = time.time()
                    ## check whether Personalized RAG evidence is ready
                    Major_event_tuple(index)
                    # evidence= CHUNK(index)
                    embeddings = ENCODER( index)
                    print(
                        "\n=================================={0} Time cost:{1}==================================".format(
                            index, (time.time() - start_time)))


def COKE_vector_store(model):
    # Create Chroma vector store
    # vector_store_path = "COKE_vector_store_db"  # Directory where embeddings are saved
    # vectorstore = Chroma(persist_directory=vector_store_path, embedding_function=model.encoder_q)
    ### load COKE data
    coke_path = os.path.join('./Dataset/COKE/train_thought.csv') ## download
    coke_path_sc = './Dataset/COKE/train_thought_combine.txt'  ## modify situation-clues only
    ### coke_event
    coke_event_path = './Dataset/COKE/train_thought_combine_event.txt'  ## situation and clues
    coke_event_json = './Dataset/COKE/train_thought_combine_event.json' ## different type
    ### coke_path_sc file check
    if not os.path.exists(coke_path_sc):
        print(f"File {coke_path_sc} does not exist.")
        csv_data = pd.read_csv(coke_path, low_memory=False)  # 防止弹出警告
        csv_df = pd.DataFrame(csv_data)
        content = csv_df[["situation", 'clue']].values
        str_content = ''
        for line in content:
            str_content += line[0] + line[1] + "\n"
        with open(coke_path_sc, 'w', encoding='utf-8') as f:
            f.write(str_content)
    else:
        coke_sc = open(coke_path_sc, 'r', encoding='utf-8').readlines()
    ###  coke_event
    if not os.path.exists(coke_event_json):
        print(f"File {coke_event_json} does not exist.")
        coke_event_list=[]
        coke_event_dict={}
        for coke in tqdm(coke_sc, total=len(coke_sc)):   ### LLM coke_event_extraction
            coke_event = try_make_requests(coke, args.model_name, 2, prompt=get_prompt('coke_event_extraction'))
            coke_event_list.append(re.sub(r'[<>]', '',coke_event))

        ### save as dict type
        for index,coke in enumerate(coke_event_list):
            coke_event_dict[f"{index}"] = coke
        with open(coke_event_json, "w") as json_file:
            json.dump(coke_event_dict, json_file)

        ### save coke_event as list type
        with open(coke_event_path, 'w', encoding='utf-8') as f_coke_event:
            for coke  in coke_event_list:
                f_coke_event.write(str(coke))
    #
    # hard_data = data_loader.HardData(config_data.hard_hparams, device=device)
    # hard_data_iterator = tx.data.DataIterator(hard_data)
    # for batch in hard_data_iterator:
    #     evt_embedding = model.encoder_q(batch.evt_ids, batch.evt_lengths)

    # with open(coke_event_json, "r") as file:
    #     coke_events = json.load(file)
    # for idx in range(len(coke_events)):
    #      ### r
    #     evt = REFORMAT( coke_events[f'{idx}'])
    #     evt_ids = tokenizer.map_text_to_id(evt)
    #
    #
    ### create  Fassis embedding dataset
    index_mean_path = 'index_mean.index'
    index_var_path = 'index_var.index'
    index_raw_path = 'index_raw.index'
    # Check if the file exists
    if not os.path.exists(index_raw_path) :
        print(f"'{index_raw_path}' not exists.")
        with open(coke_event_json, "r") as file:
            coke_events = json.load(file)
        index_raw = faiss.IndexIDMap2(faiss.IndexFlatL2(768))
        index_mean = faiss.IndexIDMap2(faiss.IndexFlatL2(500))
        index_var = faiss.IndexIDMap2(faiss.IndexFlatL2(500))
        index_mean_var_combine = faiss.IndexIDMap2(faiss.IndexFlatL2(1000))
        for idx in tqdm(range(len(coke_events)),total=len(coke_events)):
            evt= coke_events[f'{idx}']
            evt,evt_length=REFORMAT(evt)
            evt_ids = tokenizer.map_text_to_id(evt)
            embeddings,embeddings_mean,embeddings_var = model.encoder_q(torch.tensor(evt_ids).unsqueeze(0).to(device),torch.tensor(evt_length).unsqueeze(0).to(device))
            ## 768d ## 500d ## 500d
            index_raw.add_with_ids(torch.squeeze(embeddings, dim=1).cpu().detach().numpy(), idx)
            index_mean.add_with_ids(torch.squeeze(embeddings_mean,dim=1).cpu().detach().numpy(), idx)
            index_var.add_with_ids(torch.squeeze(embeddings_var,dim=1).cpu().detach().numpy(), idx)
            # index_mean_var_combine.add_with_ids(torch.squeeze(torch.cat((embeddings_mean, embeddings_var), dim=2), dim=1).cpu().detach().numpy(), idx)

        ### save index
        if idx+1 == len(coke_events):
            faiss.write_index(index_raw, 'index_raw.index')
            faiss.write_index(index_mean, 'index_mean.index')
            faiss.write_index(index_var, 'index_var.index')
            faiss.write_index(index_mean_var_combine, 'index_mean_var_combine.index')
    else:

        index_mean = faiss.read_index(index_mean_path)
        index_var = faiss.read_index(index_var_path)

    return index_mean,index_var


if __name__ == '__main__':

    args = parse_args()
    with open('./dev_label_DAIC.txt', 'r') as f:
        temp = eval(f.readlines()[0])
        dev_list = list(temp.keys())
        dev_labels=list(temp.values())
    with open('./train_label_DAIC.txt', 'r') as f:
        temp = eval(f.readlines()[0])
        train_list = list(temp.keys())
        train_labels=list(temp.values())

    if args.restart:
        if os.path.exists(args.out_filename):
            os.remove(args.out_filename)  # 删除旧的输出文件
        if os.path.exists(args.out_filename_without_exp):
            os.remove(args.out_filename_without_exp)
        train_idx = 0
        dev_idx = 0
        dev = True
        train = True
    else:
        f = open(args.out_filename_without_exp, "r").readlines()
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

    index_mean,index_var = COKE_vector_store(model)  ## return index

    Preprocess_DAIC(args, dev_idx, dev_list)
    Preprocess_DAIC(args, train_idx, train_list)


