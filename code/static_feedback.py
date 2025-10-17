import re
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from generator import build_question_and_sample

res_path="../res/baseine"
fixed_path=res_path+"_fixed/"
qus_path="..\data\questions.jsonl"
model_name=""
model_path=""
known_apis=['ASM', 'A0', 'A1', 'AB', 'ABS', 'ABSTRACT', 'ACOS', 'ACTION', 'AD', 'ADD', 'ALTERNATIVE_BRANCH', 'AND', 'Any_Array', 'Any_BCD', 'Any_Bit', 'Any_Block', 'Any_Char', 'Any_Chars', 'Any_CodeBlock', 'Any_DataBlock', 'Any_Date', 'Any_Duration', 'Any_Elementary', 'Any_Int', 'Any_Magnitude', 'Any_Num', 'Any_Ordered', 'Any_Pointer', 'Any_Real', 'Any_Reference', 'Any_Signed', 'Any_String', 'Any_Struct', 'Any_Structured', 'Type', 'Any_TypeBlock', 'Any_TypedReference', 'Any_UnOrdered', 'Any_UnSigned', 'Any_UnTypedRe', 'ference', 'AR1', 'AR2', 'ASIN', 'AT', 'ATAN', 'AUTHOR', 'AW', 'BEGIN', 'BIE', 'BR', 'BROWSERINFO', 'BR', 'BY', 'CALL', 'CASE', 'CAUSE', 'CAUSE_GROUP', 'CC0', 'CC1', 'CEIL', 'CLASS', 'CODE_VERSION1', 'COMM_BLOCK', 'CONCAT', 'CONFIGURATION', 'CONST', 'CONSTANT', 'CONTINUE', 'COS', 'DATA_BLOCK', 'DATATYPE', 'DB', 'DB_SPECIFIC', 'DBB', 'DBD', 'DBLG', 'DBNO', 'DBW', 'DBX', 'DCHAR', 'DELETE', 'DI', 'DIB', 'DID', 'DILG', 'DINO', 'DIV', 'DIW', 'DIX', 'DO', 'DT', 'EB', 'ED', 'EFFECT', 'EFFECT_GROUP', 'ELEMENT', 'ELSE', 'ELSIF', 'EN', 'END_POST_OPERATION', 'END_PRE_OPERATION', 'END_ACTION', 'END_ALTERNATIVE_BRANCH', 'END_BROWSERINFO', 'END_CASE', 'END_CAUSE', 'END_CAUSE_GROUP', 'END_CLASS', 'END_CONFIGURATION', 'END_CONST', 'END_DATA_BLOCK', 'END_EFFECT', 'END_EFFECT_GROUP', 'END_ELEMENT', 'END_FOR', 'END_FOREACH', 'END_FUNCTION', 'END_FUNCTION_BLOCK', 'END_IF', 'END_INTERFACE', 'END_INTERLOCK', 'END_INTERSECTIONS', 'END_LIBRARY', 'END_NAMESPACE', 'END_NETWORK', 'END_NAMESPACE', 'END_NETWORK', 'END_NODE', 'END_ORGANIZATION_BLOCK', 'END_PROGRAM', 'END_REGION', 'END_REPEAT', 'END_REQUIRE', 'END_RESOURCE', 'END_RUNG', 'END_SELECTION', 'END_SEQUENCE', 'END_SIMULTAN', 'EOUS_BRANCH', 'END_STEP', 'END_STRUCT', 'END_SUPERVISION', 'END_SYSTEM_FUNCTION', 'END_SYSTEM_FUNCTION_BLOCK', 'END_TRANSITION', 'END_TYPE', 'END_VAR', 'END_WHILE', 'END_WIRE', 'ENO', 'ENTRY', 'EQ', 'EW', 'EXIT', 'EXPT', 'EXTENDS', 'F_EDGE', 'FALSE', 'FAMILY', 'FB', 'FC', 'FINAL', 'FIND', 'FLOOR', 'FOR', 'FOREACH', 'FUNCTION', 'FUNCTION_BLOCK', 'GE', 'GOTO', 'GT', 'IB', 'ID', 'IF', 'IMPLEMENTATION', 'IMPLEMENTS', 'INSERT', 'INSIDE', 'INTERFACE', 'INTERLOCK', 'INTERNAL', 'INTERSECTIONS', 'INTERVAL', 'IW', 'KNOW_HOW_PROTECT', 'LABEL', 'LB', 'LD', 'LDATE', 'LE', 'LDATE_AND_TIME', 'LEFT', 'LEN', 'LIBRARY', 'LIMIT', 'LN', 'LOG', 'LOWER_BOUND', 'LT', 'LTIME_OF_DAY', 'LW', 'MAX', 'MB', 'MD', 'MDD_CHECK', 'METHOD', 'MID', 'MIN', 'MOD', 'MOVE', 'MUL', 'MUX', 'MW', 'NAME', 'NAME_OF', 'NAMESPACE', 'NE', 'NETWORK', 'NODE', 'NON_RETAIN', 'NOP', 'NOT', 'NU', 'NULL', 'OB', 'OF', 'ON', 'OR', 'ORGANIZATION_BLOCK', 'OS', 'OV', 'OVERLAP', 'OVERRIDE', 'PA', 'PAB', 'PACKED', 'PAD', 'PAW', 'PB', 'PE', 'PD', 'PEB', 'PED', 'PEW', 'PI', 'PIB', 'PID', 'PIW', 'POST_OPERATION', 'PQ', 'PQB', 'PQD', 'PQW', 'PRAGMA_BEGIN', 'PRAGMA_END', 'PRE_OPERATION', 'PRIORITY', 'PRIVATE', 'PROGRAM', 'PROTECTED', 'PUBLIC', 'PW', 'QB', 'QD', 'QW', 'R_EDGE', 'READ_ONLY', 'READ_WRITE', 'REF', 'REF_TO', 'REGION', 'RELATION', 'REPEAT', 'REPLACE', 'REQUIRE', 'RESOURCE', 'RET_VAL', 'RETAIN', 'RETURN', 'RIGHT', 'ROL', 'ROR', 'RUNG', 'S5T', 'SDB', 'SEL', 'SELECTION', 'SEQUENCE', 'SFB', 'SFC', 'SHL', 'SHR', 'SIMULTANEOUS_BRANCH', 'SIN', 'SINGLE', 'SQRT', 'STANDARD', 'STEP', 'STW', 'SUB', 'SUBSET', 'SUPER', 'SUPERVISION', 'SYSTEM_FUNCTION', 'SYSTEM_FUNCTION_BLOCK', 'TAN', 'TASK', 'THEN', 'THIS', 'TITLE', 'TO', 'TO_BOOL', 'TO_BYTE', 'TO_CHAR', 'TO_DATE', 'TO_DINT', 'TO_DT', 'TO_DWORD', 'TO_INT', 'TO_LDATE', 'TO_LDT', 'TO_LINT', 'TO_LREAL', 'TO_LTIME', 'TO_LTOD', 'TO_LWORD', 'TO_REAL', 'TO_SINT', 'TO_TIME', 'TO_TOD', 'TO_UDINT', 'TO_UINT', 'TO_ULINT', 'TO_USINT', 'TO_WCHAR', 'TO_WORD', 'TOD', 'TRANSITION', 'TRUE', 'TRUNC', 'TYPE', 'UDT', 'UNLINKED', 'UNTIL', 'UO', 'UPPER_BOUND', 'USING', 'USTRING', 'VAR', 'VAR_ACCESS', 'VAR_CONFIG', 'VAR_EXTERNAL', 'VAR_GLOBAL', 'VAR_IN_OUT', 'VAR_INPUT', 'VAR_OUTPUT', 'VAR_TEMP', 'VERSION', 'WHILE', 'WIRE', 'WITH', 'XOR']

def load_sample(path):
    data = []
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            sample['ID'] = count
            count += 1
            data.append(sample)
    return data

def load_code(dir_path):
    name2code = {}
    # 遍历目录下的所有文件
    for filename in os.listdir(dir_path):
        # 检查文件是否为.scl文件
        if filename.endswith('.scl'):
            file_path = os.path.join(dir_path, filename)
            # 确保是文件而不是目录
            if os.path.isfile(file_path):
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    # 使用文件名（不带后缀）作为键，或使用完整文件名
                    name = os.path.splitext(filename)[0]  # 去掉扩展名
                    # 如果需要保留扩展名，可以直接使用 filename
                    name2code[name] = content
                except Exception as e:
                    print(f"读取文件 {filename} 时出错: {e}")
    return name2code

def load_text_generation_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

def send2llm(final_prompt,tokenizer, model):
    inputs = tokenizer(final_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def build_prompt(data,error_message):
    PROMPT='''The following is an SCL programming question for the Siemens TIA Portal platform, including the problem statement and the answer code.
{qus_ans}
After static analysis, the following potential issues were found in the code:
{error_message}
Please modify the code to achieve the correct functionality.
'''
    variables = {
        "qus_ans": build_question_and_sample(data, is_sample=True),
        "error_message": error_message,
    }
    prompt = PROMPT.format(**variables)
    return prompt

def get_apis4(code):
    pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    all_keywords = pattern.finditer(code)
    function_names=[]
    # 遍历匹配项及其在原文中的位置
    last_name=''
    for match in all_keywords:
        next_char=code[min(match.end(),len(code)-1)]
        previous_char=code[max(match.start()-1,0)]
        # 一个name的右边是左括号而且不是关键词
        if  next_char=='(' :
            function_names.append(match.group())
        last_name=match.group()
    return function_names

def check_api_unknown(code):
    error_message=""
    flag=False

    apis=get_apis4(code)
    for api in apis:
        if api not in known_apis:
            error_message+=f'Unknown api: {api}\n'
            flag=True
    return flag,error_message

def check_var_undefined(var_definition,code):
    error_message=""
    flag=False
    # 先删除行注释
    cleaned_code = re.sub(r'//.*?\n', '', code)  
    # 然后再找到所有的# 开头变量
    def is_digit(s):  
        return bool(re.search(r'\d', s))  
    pattern = re.compile(r'#\w.*?(?=\W)')  
    matches = pattern.finditer(cleaned_code)  
    for match in matches:  
        var=match.group()
        if is_digit(var):
            continue
        var_=var.replace('#','')
        if not var_ in var_definition:
            error_message+=f'Undefined variety: {var}\n'
            flag=True
        
    # pattern = re.compile(r'\.\w.*?(?=\W)') 
    pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b') 
    
    matches = pattern.finditer(cleaned_code)  
    for match in matches:  
        start=match.start()
        if  start==0 or not code[start-1]=='.':
            continue
        var=match.group()
        if is_digit(var):
            continue
        var_=var.replace('.','')
        if not var_ in var_definition:
            error_message+=f'Undefined variety: {var}\n'
            flag=True

    return flag,error_message

def post_process(res):
    return res

def write_scl(name,code):
    f=open(res_path+name+".scl",'w')
    f.write(code)
    f.close()

if __name__=="__main__":
    #load qus ans
    data=load_sample(qus_path)
    name2code=load_code(res_path)
    for d in data:
        d["answer"]=name2code[d["name"]]

    tokenizer, model=load_text_generation_model()
    for d in data:
        ans=d["answer"]
        # 先拿到所有变量定义的块
        pattern = re.compile(r'VAR.*?\n.*?END_VAR', re.DOTALL)  # 使用 re.DOTALL 标志  
        matches = pattern.findall(ans)  
        var_definition=''
        for match in matches:  
            var_definition+=match
        
        flag1,error_message1=check_var_undefined(var_definition,ans)
        flag2,error_message2=check_api_unknown(ans)
        if flag1 or flag2:
            error_message=error_message1+error_message2
            prompt=build_prompt(d,error_message)
            res=send2llm(prompt,tokenizer, model)
            code=post_process(res)
            d["answer"]=code
    
    for d in data:
        write_scl(d["name"],d["answer"])
        



