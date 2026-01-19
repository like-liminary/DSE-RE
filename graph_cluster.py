from typing import Set, Dict, List, Tuple
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

import matplotlib.pyplot as plt


from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI 

client = OpenAI(base_url = "",api_key  = "")

model_name=""
device="cuda"



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
if device:
    model.to(device)
model.eval()

def load_bert(model_name="", device="cuda"):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device:
        model.to(device)
    model.eval()
    return tokenizer, model

def encode_words(words, tokenizer, model, device="cpu", max_length=16, batch_size=32):
    
    
    words = sorted(words, key=len)

    bat = tokenizer(words, padding=True, truncation=True, return_length=True)
    max_len = int(max(bat["length"]))  
    print(f"max_len: {max_len}")
    all_embs = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state 

        
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)         
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)       
        mean_pooled = summed / counts                          

        all_embs.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embs)


def get_category_from_llm(cluster_words: List[str], item_type_name) -> str:
    
    words_str = ", ".join(cluster_words)
    
    promptzh = f"""
    这是一组知识图谱中的类型：[{words_str}]。
    请分析这些类型的共同含义，概括出一个最精准、简短的新的类型来代表这组类型。
    要求：
    1. 仅输出类型名称，不要包含任何解释或其他文字。
    2. 类型名称最好是一个更通用的上位词。
    3. 输出尽可能短，不要超过5个字。
    """
    promptzh_entity = f"""
    这是一组知识图谱中的实体类型：[{words_str}]。
    请分析这些类型的共同含义，概括出一个最精准、简短的新的实体类型来代表这组类型。
    要求：
    1. 仅输出实体类型名称，不要包含任何解释或其他文字。
    2. 实体类型名称最好是一个更通用的上位词。
    3. 输出尽可能短，不要超过5个字。
    """
    promptzh_rel = f"""
    这是一组知识图谱中的关系类型：[{words_str}]。
    请分析这些类型的共同含义，概括出一个最精准、简短的新的关系类型来代表这组类型。
    要求：
    1. 仅输出关系类型名称，不要包含任何解释或其他文字。
    2. 关系类型名称最好是一个更通用的上位词。
    3. 输出尽可能短，不要超过5个字。
    """
    
    if item_type_name == "实体类型":
        prompt = promptzh_entity
    elif item_type_name == "关系类型":
        prompt = promptzh_rel
    else:
        prompt = promptzh

    """You are a professional taxonomist and terminology standardization assistant."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "作为一个专业的知识图谱实体类型和关系类型归纳专家,按照要求对实体类型或关系类型进行归纳。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, 
        )
        category = response.choices[0].message.content.strip()
        
        category = category.replace("。", "").replace("类别：", "").replace("类", "").strip()
        return category
    except Exception as e:
        print(f"LLM调用出错: {e}")
        
        return cluster_words[0] if cluster_words else "Unknown"


def cluster_and_represent(words, embeddings, threshold=0.7, max_workers=8, item_type_name=None):
    
    words = sorted(words, key=len)
    N = len(words)
    sim_mat = cosine_similarity(embeddings)  # (N, N)

   

    
    G = nx.Graph()
    G.add_nodes_from(words)
    for i in range(N):
        for j in range(i+1, N):
            sim = float(sim_mat[i, j])
            if sim >= threshold:
                G.add_edge(words[i], words[j], weight=sim)

    
    clusters = list(nx.connected_components(G))

    
    mapping = {}
    
    def process_single_cluster(comp_set):
        comp_list = list(comp_set)
        
        if len(comp_list) == 1:
            return comp_list, comp_list[0]
        
        
        label = get_category_from_llm(comp_list, item_type_name)
        return comp_list, label

    print(f"{item_type_name}开始处理 {len(clusters)} 个簇，使用 {max_workers} 个线程...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        futures = [executor.submit(process_single_cluster, comp) for comp in clusters]
        
        
        for future in as_completed(futures):
            comp_list, label = future.result()
            for w in comp_list:
                mapping[w] = label

    return mapping, clusters


def clauster_graph(items, item_type_name):
    
    words = items
    threshold = 0.7
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embs = encode_words(words, tokenizer, model, device=device)
    mapping, clusters = cluster_and_represent(words, embs, threshold=threshold, max_workers=10, item_type_name=item_type_name)

    return mapping


if __name__ == "__main__":
    sample = {
    "上有",
    "齐平"
}
    result = clauster_graph(sample)
    print(result)
    print("-----------------")
    print(json.dumps(result, ensure_ascii=False, indent=4))


