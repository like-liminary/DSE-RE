import json
import os
import random  
import matplotlib.pyplot as plt
from collections import defaultdict
import time  
from openai import OpenAI
from prompt import prompt_template_CN, prompt_template_EN
import re
from sklearn.cluster import DBSCAN

from graph_cluster import clauster_graph

GPTclient = OpenAI(base_url = "",api_key  = "")
GPTMODEL = "gpt-4o"

from sentence_transformers import SentenceTransformer
import numpy as np
import hdbscan

encoder=''

filename = ""

client = GPTclient
model_name = GPTMODEL  

def mock_llm_call(sys_prompt, user_prompt):



    response = client.chat.completions.create(
    model=model_name,  # 填写需要调用的模型编码
    temperature=0.2,
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
        ],
    )
    return response.choices[0].message.content


def parse_llm_response(response_text):
    if not response_text:
        print("警告: LLM响应为空。")
        return None

    
    match = re.search(r"\{[\s\S]*\}", response_text)
    if not match:
        print(f"警告: 未能从响应中提取JSON对象: {response_text}")
        return None

    json_text = match.group(0)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        print(f"警告: 无法解析LLM响应为JSON: {json_text}")
        return None
    
    return data

def save_result(data, filename="extraction_results.json"):
    
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            # 将 data 转为带缩进的 JSON 字符串
            pretty_json = json.dumps(data, ensure_ascii=False, indent=4)
            f.write(pretty_json)
            f.write('\n\n')  # 对象之间空一行，便于区分
    except IOError as e:
        print(f"错误: 无法写入文件 {filename}: {e}")

def load_results_old(filename="extraction_results.json"):
    
    results = []
    if not os.path.exists(filename):
        return results  # 如果文件不存在，返回空列表
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"警告: 跳过无法解析的行: {line.strip()}")
    except IOError as e:
        print(f"错误: 无法读取文件 {filename}: {e}")
    return results

def load_results(filename="extraction_results.json"):
    
    results = []
    if not os.path.exists(filename):
        return results  # 文件不存在时返回空列表
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        # 以双换行分割各条记录，过滤空段
        chunks = [chunk for chunk in content.split('\n\n') if chunk.strip()]
        for idx, chunk in enumerate(chunks, 1):
            try:
                results.append(json.loads(chunk))
            except json.JSONDecodeError as e:
                print(f"警告: 第 {idx} 条记录解析失败: {e}")
    except IOError as e:
        print(f"错误: 无法读取文件 {filename}: {e}")
    return results

def update_schema(entity_types_set, relation_types_set, extraction_result):
    
    
    if not extraction_result:
        return

    triples = extraction_result.get('triple')
    if not isinstance(triples, list):
        print("警告: 抽取结果中的 'triple' 不是列表格式。")
        return

    for t in triples:
        head_type = t.get('head_type')
        tail_type = t.get('tail_type')
        relation = t.get('relation')

        if head_type:
            entity_types_set.add(head_type)
        if tail_type:
            entity_types_set.add(tail_type)
        if relation:
            relation_types_set.add(relation)

def build_prompt(text, current_entity_types, current_relation_types):
    
    # 将集合转换为列表并排序，以便提示词相对稳定
    entity_type_list = sorted(list(current_entity_types))
    relation_type_list = sorted(list(current_relation_types))
    entity_types = ",".join(entity_type_list)
    relation_types = ",".join(relation_type_list)

    
    user_prompt, sys_prompt = prompt_template_CN(text, entity_types, relation_types)

    return sys_prompt, user_prompt


def cluster_schema_items(items_set, item_type_name):

    print(f"\n--- 开始对 {item_type_name} 进行聚类 ---")
    items_list = sorted(list(items_set))  # 排序以保证代表选择的某种一致性
    num_items = len(items_list)
    print(f"聚类前的数量: {num_items}")

    mapping = clauster_graph(items_set, item_type_name)

    return mapping

def update_results_after_clustering_old(results_list, entity_type_mapping, relation_mapping,filename="extraction_results.jsonl"):
    
    print("\n--- 开始更新已保存的抽取结果以反映聚类变更 ---")
    updated_count = 0
    updated_results_list = []  # 用于存储更新后的结果

    for result in results_list:  # 遍历内存中的结果列表
        # original_result_extraction_str = json.dumps(result.get('extraction')) # 用于比较的原始抽取部分字符串
        changed_in_this_result = False  # 标记此条记录是否发生变化
        if 'extraction' in result and result['extraction']:
            extraction = result['extraction']  # 获取抽取数据字典
    
            original_head_type = extraction.get('head_type')
            original_tail_type = extraction.get('tail_type')
            original_relation = extraction.get('relation')

            # 从映射中获取新的类型/关系；如果不在映射中，则使用原始值
            new_head_type = entity_type_mapping.get(original_head_type, original_head_type)
            new_tail_type = entity_type_mapping.get(original_tail_type, original_tail_type)
            new_relation = relation_mapping.get(original_relation, original_relation)

            if new_head_type != original_head_type:
                extraction['head_type'] = new_head_type
                changed_in_this_result = True
            if new_tail_type != original_tail_type:
                extraction['tail_type'] = new_tail_type
                changed_in_this_result = True
            if new_relation != original_relation:
                extraction['relation'] = new_relation
                changed_in_this_result = True

            if changed_in_this_result:
                updated_count += 1
                # print(f"  记录更新: {original_result_extraction_str} -> {json.dumps(extraction)}") # 可选：打印详细变化

        updated_results_list.append(result)  # 添加更新后的（或未改变的）结果到新列表

    print(f"内存中共有 {updated_count} 条记录的类型/关系被更新。")

    
    if updated_count > 0 or True:  # 如果有任何更新，或者即使没有，也重写以确保一致性（例如，如果聚类改变了代表词但未减少总数）
        print(f"将更新后的 {len(updated_results_list)} 条结果写回文件: {filename}")
        temp_filename = filename + ".tmp"  # 使用临时文件保证写入安全
        try:
            with open(temp_filename, 'w', encoding='utf-8') as f:
                for res_entry in updated_results_list:  # 写入更新后的内存列表
                    json.dump(res_entry, f, ensure_ascii=False)
                    f.write('\n')
            # 替换原文件
            os.replace(temp_filename, filename)
        except IOError as e:
            print(f"错误: 写入更新后的结果文件失败: {e}")
            # 如果发生错误，尝试删除临时文件
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            print(f"错误: 更新文件时发生意外错误: {e}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    print("--- 结果更新结束 ---")
    return updated_results_list  # 返回更新后的结果列表

def update_results_after_clustering(all_results, entity_type_mapping, relation_mapping, filename="extraction_results.json"):
   
    print("\n--- 开始更新已保存的抽取结果以反映聚类变更 ---")
    updated_count = 0
    updated_results = []

    for result in all_results:
        changed = False
        extraction = result.get('extraction')
        # 仅当存在提取三元组时才更新
        if extraction and isinstance(extraction.get('triple'), list):
            for triple in extraction['triple']:
                orig_head_type = triple.get('head_type')
                orig_tail_type = triple.get('tail_type')
                orig_relation = triple.get('relation')

                # 应用映射
                new_head_type = entity_type_mapping.get(orig_head_type, orig_head_type)
                new_tail_type = entity_type_mapping.get(orig_tail_type, orig_tail_type)
                new_relation = relation_mapping.get(orig_relation, orig_relation)

                # 如果发生变化，则更新三元组并标记
                if new_head_type != orig_head_type:
                    triple['head_type'] = new_head_type
                    changed = True
                if new_tail_type != orig_tail_type:
                    triple['tail_type'] = new_tail_type
                    changed = True
                if new_relation != orig_relation:
                    triple['relation'] = new_relation
                    changed = True

            if changed:
                updated_count += 1

        updated_results.append(result)

    print(f"内存中共有 {updated_count} 条记录的三元组类型/关系被更新。")

    # 重写结果文件，使用美化格式以便阅读
    temp_file = filename + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            for entry in updated_results:
                # 使用 indent 参数展开 JSON
                json.dump(entry, f, ensure_ascii=False, indent=4)
                f.write('\n')  # 每条记录后保留一个空行，可根据需要改为 '\n\n'
        os.replace(temp_file, filename)
        print(f"已将 {len(updated_results)} 条结果以可读格式写回文件: {filename}")
    except Exception as e:
        print(f"错误: 无法写入结果文件: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print("--- 结果更新结束 ---")
    return updated_results

def plot_schema_growth(history):
    """
    绘制实体类型和关系类型数量随处理文本数量变化的折线图。
    """
    if not history:
        print("没有历史数据可供绘图。")
        return

    steps = [item['step'] for item in history]
    entity_type_counts = [item['entity_types'] for item in history]
    relation_type_counts = [item['relation_types'] for item in history]

    # 确保绘图库使用支持中文的字体
    plt.rcParams['font.sans-serif'] = ['FandolSong']  # 例如：SimHei, Microsoft YaHei
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

    plt.figure(figsize=(12, 7))  # 调整图形大小
    plt.plot(steps, entity_type_counts, marker='o', linestyle='-', label='Number of Entity Types')
    plt.plot(steps, relation_type_counts, marker='x', linestyle='--', label='Number of Relation Types')

    plt.xlabel("Extraction Step")
    plt.ylabel("Count")
    plt.title("Trend of Entity and Relation Type Counts")
    plt.legend()
    plt.grid(True)
    if steps:  # 设置x轴刻度，使其更易读
        plt.xticks(range(min(steps), max(steps) + 1, max(1, len(steps) // 10 if len(steps) > 10 else 1)))
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    plot_filename = f""
    try:
        plt.savefig(plot_filename)
        print(f"\nSchema 增长图已保存为 {plot_filename}")
    except Exception as e:
        print(f"错误: 保存图形失败: {e}")
    # plt.show() # 如果需要在脚本运行时显示图形，取消此行注释

def split_to_sentences(text, max_len=150):
    
    # 先按句末标点分割，并保留标点
    pieces = re.split(r'(?<=[。！？])\s*', text)
    sentences = []
    
    for sent in pieces:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_len:
            sentences.append(sent)
        else:
            # 长句再按逗号拆
            subpieces = re.split(r'(?<=[，,])\s*', sent)
            buffer = ''
            for sub in subpieces:
                if len(buffer) + len(sub) <= max_len:
                    buffer += sub
                else:
                    if buffer:
                        sentences.append(buffer.strip())
                    # 如果单个 sub 也超过 max_len，就硬切
                    if len(sub) <= max_len:
                        buffer = sub
                    else:
                        # 按 max_len 切片
                        for i in range(0, len(sub), max_len):
                            sentences.append(sub[i:i+max_len].strip())
                        buffer = ''
            if buffer:
                sentences.append(buffer.strip())
    return sentences


def load_all_txt_sentences(folder_path, max_len=150, encoding='utf-8'):
   
    all_sentences = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.txt'):
            continue
        fullpath = os.path.join(folder_path, fname)
        with open(fullpath, 'r', encoding=encoding) as f:
            text = f.read()
        sents = split_to_sentences(text, max_len=max_len)
        all_sentences.extend(sents)
    return all_sentences


if __name__ == "__main__":
    
    file_path = f''

    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts_to_process = [line.strip() for line in f if line.strip()]
    
    print(f"加载了 {len(texts_to_process)} 条文本。")

    output_filename = f""
    
    
    all_results = load_results(output_filename)  # 这是内存中的所有结果列表
    current_entity_types = set()
    current_relation_types = set()
    # 从已加载的结果中恢复 schema
    for res in all_results:
        if 'extraction' in res and res['extraction']:
            update_schema(current_entity_types, current_relation_types, res['extraction'])

    print(f"初始加载 {len(all_results)} 条结果。")
    print(f"初始实体类型数量: {len(current_entity_types)}, 初始关系类型数量: {len(current_relation_types)}")

    schema_growth_history = []  # 记录 schema 增长历史
    
    
    processed_text_count_offset = len(all_results)


    # --- 修改部分：为实体类型和关系类型分别初始化聚类阈值 ---
    entity_clustering_threshold = 15
    relation_clustering_threshold = 15
    print(f"初始实体类型聚类阈值: {entity_clustering_threshold}")
    print(f"初始关系类型聚类阈值: {relation_clustering_threshold}")


    # 2. 批量处理文本
    for i, text in enumerate(texts_to_process):
        current_step_number = processed_text_count_offset + i + 1  # 当前处理的总步骤号
        print(f"\n--- 处理文本 {i + 1}/{len(texts_to_process)} (总步骤: {current_step_number}) ---")
        print(f"文本: {text}")

        # 2.1 构建当前提示词
        sys_prompt, user_prompt = build_prompt(text, current_entity_types, current_relation_types)
        # prompt += f"\n\n文本内容:\n{text}"

        # 2.2 调用LLM进行抽取 (使用模拟函数)
        llm_response_text = mock_llm_call(sys_prompt, user_prompt)

        # 2.3 解析LLM响应
        extraction_data = parse_llm_response(llm_response_text)

        # 2.4 处理和保存结果
        result_entry = {
            "text": text,
            "llm_response": llm_response_text,  # 保存原始响应，便于调试
            "extraction": extraction_data,  # 保存解析后的五元组，如果解析失败则为None
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step_processed": current_step_number  # 记录处理步骤
        }
        save_result(result_entry, output_filename)  # 追加到文件
        all_results.append(result_entry)  # 更新内存中的列表

        # 2.5 更新 schema (实体类型和关系类型)
        if extraction_data:
            update_schema(current_entity_types, current_relation_types, extraction_data)
            print(f"抽取成功: {extraction_data}")
        else:
            print("本次未能成功抽取或解析五元组。")

        print(f"抽取后，当前实体类型数量: {len(current_entity_types)}, 关系类型数量: {len(current_relation_types)}")


        # --- 记录 schema 增长历史 (在本次抽取后，聚类前) ---
        schema_growth_history.append({
            "step": current_step_number,  # 使用总步骤号
            "entity_types": len(current_entity_types),
            "relation_types": len(current_relation_types)
        })

        # --- 修改部分：具有独立阈值的条件性聚类逻辑 ---
        any_clustering_triggered_this_step = False  # 标记本轮是否有任何聚类被触发
        schema_changed_by_clustering = False  # 标记本轮是否有任何聚类实际改变了Schema

        # 初始化映射为恒等映射，如果某类型不聚类，则使用此映射
        entity_type_mapping = {item: item for item in current_entity_types}
        relation_mapping = {item: item for item in current_relation_types}

        # --- 实体类型聚类判断与执行 ---
        if len(current_entity_types) > entity_clustering_threshold:
            any_clustering_triggered_this_step = True
            print(f"\n--- 触发实体类型聚类 (当前实体阈值: {entity_clustering_threshold}) ---")
            print(f"当前实体类型数: {len(current_entity_types)}")

            entity_types_before_clustering = current_entity_types.copy()
            # 更新 entity_type_mapping 为实际聚类结果
            entity_type_mapping = cluster_schema_items(current_entity_types, "实体类型")
            current_entity_types = set(entity_type_mapping.values())  # 更新为聚类后的类型集合

            # entity_clustering_threshold = len(current_entity_types) + 10  # 增加实体类型聚类阈值
            # print(f"新的实体类型聚类阈值: {entity_clustering_threshold}")
            
            if len(current_entity_types) < len(entity_types_before_clustering):
                schema_changed_by_clustering = True  # 标记Schema已改变
                print("实体类型聚类导致Schema项目数量减少。")
                entity_clustering_threshold = len(current_entity_types) + 5  # 增加实体类型聚类阈值
                print(f"新的实体类型聚类阈值: {entity_clustering_threshold}")
            else:
                print("实体类型聚类执行完毕，但项目数量未减少，实体类型阈值不变。")
            print(f"实体类型聚类后数量: {len(current_entity_types)}")
        else:
            print(f"实体类型数量 ({len(current_entity_types)}) 未达到其聚类阈值 ({entity_clustering_threshold})。跳过实体类型聚类。")
            # entity_type_mapping 已是恒等映射

        # --- 关系类型聚类判断与执行 ---
        if len(current_relation_types) > relation_clustering_threshold:
            any_clustering_triggered_this_step = True
            print(f"\n--- 触发关系类型聚类 (当前关系阈值: {relation_clustering_threshold}) ---")
            print(f"当前关系类型数: {len(current_relation_types)}")

            relation_types_before_clustering = current_relation_types.copy()
            # 更新 relation_mapping 为实际聚类结果
            relation_mapping = cluster_schema_items(current_relation_types, "关系类型")
            current_relation_types = set(relation_mapping.values())  # 更新为聚类后的关系集合

            # relation_clustering_threshold = len(current_relation_types) + 10  # 增加关系类型聚类阈值
            # print(f"新的关系类型聚类阈值: {relation_clustering_threshold}")

            if len(current_relation_types) < len(relation_types_before_clustering):
                schema_changed_by_clustering = True  # 标记Schema已改变
                print("关系类型聚类导致Schema项目数量减少。")
                relation_clustering_threshold = len(current_relation_types) + 5  # 增加关系类型聚类阈值
                print(f"新的关系类型聚类阈值: {relation_clustering_threshold}")
            else:
                print("关系类型聚类执行完毕，但项目数量未减少，关系类型阈值不变。")
            print(f"关系类型聚类后数量: {len(current_relation_types)}")
        else:
            print(f"关系类型数量 ({len(current_relation_types)}) 未达到其聚类阈值 ({relation_clustering_threshold})。跳过关系类型聚类。")
            # relation_mapping 已是恒等映射

        # --- 如果任一聚类被触发且导致Schema变化，则更新结果 ---
        if schema_changed_by_clustering:  # 检查总的改变标记
            print("\n由于Schema变更，开始更新已保存的抽取结果...")
            # 更新内存中的 all_results 列表并重写文件
            # entity_type_mapping 和 relation_mapping 包含了各自聚类的结果（或恒等映射）
            all_results = update_results_after_clustering(all_results, entity_type_mapping, relation_mapping,output_filename)
            # 各自的阈值已在上面独立更新
            print(f"Schema已更新。当前实体阈值: {entity_clustering_threshold}, 当前关系阈值: {relation_clustering_threshold}")
        elif any_clustering_triggered_this_step:  # 聚类被触发但未改变Schema
            print("\n聚类执行完毕，但Schema项目数量未减少（或已是最优状态）。")

        if any_clustering_triggered_this_step:  # 如果至少有一个聚类过程被触发
            print("--- 本轮聚类检查结束 ---")


    print("\n--- 所有文本处理完成 ---")

    # 3. 绘制 Schema 增长图 (使用在整个处理过程中收集的历史数据)
    plot_schema_growth(schema_growth_history)
    #保存schema_growth_history到json文件
    with open(f"", 'w', encoding='utf-8') as f:
        json.dump(schema_growth_history, f, ensure_ascii=False, indent=4)

    print("\n--- 系统运行结束 ---")
    print(f"最终实体类型 ({len(current_entity_types)}): {sorted(list(current_entity_types))}")
    print(f"最终关系类型 ({len(current_relation_types)}): {sorted(list(current_relation_types))}")
    #将current_entity_types和current_relation_types写入txt文件
    with open(f"", 'w', encoding='utf-8') as f:
        for item in sorted(list(current_entity_types)):
            f.write(f"{item}\n")
    with open(f"", 'w', encoding='utf-8') as f:
        for item in sorted(list(current_relation_types)):
            f.write(f"{item}\n")
    print(f"最终实体类型聚类阈值: {entity_clustering_threshold}")
    print(f"最终关系类型聚类阈值: {relation_clustering_threshold}")