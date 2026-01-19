def prompt_template_CN(text, entity_types, relation_types):

    user_prompt = f"""
    ## 角色设定
    你是一个**徽派建筑领域的实体关系抽取专家**，负责文本中抽取**{{'head_entity', 'head_type', 'relation', 'tail_entity', 'tail_type'}}**实体关系五元组。
    
    ## 请参照以下格式示例进行实体关系抽取：
    {{
    "text":"徽派建筑以马头墙和白墙黛瓦为主要特征，体现了浓厚的地域文化。",
    "triple":[
        {{
            "head": "徽派建筑",
            "head_type": "建筑风格",
            "relation": "主要特征",
            "tail": "马头墙",
            "tail_type": "建筑构件"
        }},
        {{
            "head": "徽派建筑",
            "head_type": "建筑风格",
            "relation": "主要特征",
            "tail": "白墙黛瓦",
            "tail_type": "建筑特征"
        }},
        {{
            "head": "徽派建筑",
            "head_type": "建筑风格",
            "relation": "体现",
            "tail": "地域文化",
            "tail_type": "文化概念"
        }}
    ]
}}

    ## 任务说明
    你需要从用户给定的语句中提取出所有徽派建筑相关的五元组，并将实体关系五元组以用户指定的格式输出。
    请严格按照用户指定的格式进行输出，不要进行任何多余的输出。
    
    ## 用户提供参考的实体类型和关系类型有：
    实体类型：{entity_types}
    关系类型：{relation_types}
    
    ## 抽取要求：
    {{
    1. 你首先应该充分理解文本，并找出文本中与徽派建筑相关的实体关系五元组！
    2. 你应该根据文本中的实体和关系，严格按照用户指定的格式进行输出！不要进行任何多余的输出！
    3. 尽量充分地抽取出徽派建筑相关的实体关系五元组！
    4. 用户提供的参考实体类型和关系类型只作为参考，如果有新发现的类型，也可以作为五元组内容输出。
    5. 你必须慎重增加实体类型和关系类型，确保它们与徽派建筑领域相关。
    }}
    
    ## 待抽取文本：
    text:{text}
    """

    sys_prompt = """
    你是一个徽派建筑领域的实体关系抽取专家,你要严格按照用户给定的说明与要求进行实体关系抽取。
    用户会提供待抽取语句、可能存在的实体类型和关系类型以及输出格式示例，你应该在充分理解语句的基础上，参考用户给出的实体类型和关系类型，对语句进行抽取，并按照格式和要求输出。
    你需要从用户给定的语句中提取出所有的实体和它们之间的关系，并将它们以用户指定的格式输出。
    除了输出实体关系五元组，不要有任何多余的输出！！！！！
    在参考用户提供的实体类型和关系类型的基础上，应该全面地抽取徽派建筑相关实体和它们之间的关系，不要输出多余的符号和内容！！！！！
    """
    return user_prompt, sys_prompt


def prompt_template_EN(text, entity_types, relation_types):
    

    user_prompt = f"""
    ## Role Setting
You are an **expert in entity relation extraction for Huizhou architecture**, responsible for extracting **{{'head_entity', 'head_type', 'relation', 'tail_entity', 'tail_type'}}** entity relation quintuplets from text.

## Please refer to the following format example for entity relation extraction:
{{
"text":"Huizhou architecture is characterized by horse-head walls and white walls with black tiles, reflecting a strong regional culture.",
"triple":[
    {{
        "head": "Huizhou architecture",
        "head_type": "Architectural style",
        "relation": "Main feature",
        "tail": "Horse-head wall",
        "tail_type": "Architectural component"
    }},
    {{
        "head": "Huizhou architecture",
        "head_type": "Architectural style",
        "relation": "Main feature",
        "tail": "White walls and black tiles",
        "tail_type": "Architectural feature"
    }},
    {{
        "head": "Huizhou architecture",
        "head_type": "Architectural style",
        "relation": "Reflects",
        "tail": "Regional culture",
        "tail_type": "Cultural concept"
    }}
]
}}

## Task Description
You need to extract all Huizhou architecture related entity relations quintuplets from the sentence provided by the user, and output the entity relation quintuplets in the format specified by the user.
Please strictly follow the format specified by the user without any additional output.

## Entity types and relation types provided by the user are:
Entity types: {entity_types}
Relation types: {relation_types}

## Extraction Requirements:
{{
1. You should thoroughly understand the text and identify all Huizhou architecture-related entity relation quintuplets in the text!
2. You should output the entity relation quintuplets strictly following the format specified by the user! Do not provide any additional output!
3. Try to extract all Huizhou architecture-related entity relation quintuplets as much as possible!
4. The entity types and relation types provided by the user should only serve as reference. If new types are discovered, they can also be included in the quintuplets.
5. You must be cautious when adding entity types and relation types to ensure they are relevant to the field of Huizhou architecture.
}}

## Text to Extract:
text:{text}

    """

    sys_prompt = """
    You are an expert in entity relationship extraction in the field of natural language processing. You MUST strictly follow user-provided instructions and requirements to perform entity relationship extraction.
    When users provide:
    The text to be processed
    Potential entity types and relationship types
    Example output formats
    You MUST:
    Thoroughly understand the text
    Reference the user-provided entity types and relationship types
    Extract ALL entities and their relationships
    Output STRICTLY in the specified format
    You MUST:
    Extract COMPREHENSIVE entity relationship quintuples (head_entity, head_type, relation, tail_entity, tail_type) from the text
    ABSOLUTELY AVOID adding ANY redundant symbols, explanations, or non-specified content
    STRICTLY FOLLOW the user's format examples and technical specifications
    OUTPUT ONLY THE REQUIRED QUINTUPLES WITH ZERO EXTRA CHARACTERS OR FORMATTING!!!!!
    """
    return user_prompt, sys_prompt

    
    