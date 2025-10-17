import json

raw_fixed_example_path="E:/workspace/TIA/data/fixed_examples_raw.jsonl"
fixed_example_path="E:/workspace/TIA/data/fixed_examples.jsonl"

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

#代码长度适中
def line_num(data,score):
    for i in range(len(data)):
        code=data[i]["answer"]
        c=0
        lines=code.split('\n')
        for line in lines:
            if line:
                c+=1
        if c>50 and c<100:
            score[i]+=10


#结构化信息丰富
def structural_info(data,score):
    for i in range(len(data)):
        code=data[i]["answer"]
        c=0
        lines=code.split('\n')
        for line in lines:
            if "IF" in line or "ELSE" in line or "ELSIF" in line or "FOR" in line or "END_STRUCT" in line:
                score[i]+=1


#API调用丰富
api_keywords=['ASM', 'A0', 'A1', 'AB', 'ABS', 'ABSTRACT', 'ACOS', 'ACTION', 'AD', 'ADD', 'ALTERNATIVE_BRANCH', 'AND', 'Any_Array', 'Any_BCD', 'Any_Bit', 'Any_Block', 'Any_Char', 'Any_Chars', 'Any_CodeBlock', 'Any_DataBlock', 'Any_Date', 'Any_Duration', 'Any_Elementary', 'Any_Int', 'Any_Magnitude', 'Any_Num', 'Any_Ordered', 'Any_Pointer', 'Any_Real', 'Any_Reference', 'Any_Signed', 'Any_String', 'Any_Struct', 'Any_Structured', 'Type', 'Any_TypeBlock', 'Any_TypedReference', 'Any_UnOrdered', 'Any_UnSigned', 'Any_UnTypedRe', 'ference', 'AR1', 'AR2', 'ASIN', 'AT', 'ATAN', 'AUTHOR', 'AW', 'BEGIN', 'BIE', 'BR', 'BROWSERINFO', 'BR', 'BY', 'CALL', 'CASE', 'CAUSE', 'CAUSE_GROUP', 'CC0', 'CC1', 'CEIL', 'CLASS', 'CODE_VERSION1', 'COMM_BLOCK', 'CONCAT', 'CONFIGURATION', 'CONST', 'CONSTANT', 'CONTINUE', 'COS', 'DATA_BLOCK', 'DATATYPE', 'DB', 'DB_SPECIFIC', 'DBB', 'DBD', 'DBLG', 'DBNO', 'DBW', 'DBX', 'DCHAR', 'DELETE', 'DI', 'DIB', 'DID', 'DILG', 'DINO', 'DIV', 'DIW', 'DIX', 'DO', 'DT', 'EB', 'ED', 'EFFECT', 'EFFECT_GROUP', 'ELEMENT', 'ELSE', 'ELSIF', 'EN', 'END_POST_OPERATION', 'END_PRE_OPERATION', 'END_ACTION', 'END_ALTERNATIVE_BRANCH', 'END_BROWSERINFO', 'END_CASE', 'END_CAUSE', 'END_CAUSE_GROUP', 'END_CLASS', 'END_CONFIGURATION', 'END_CONST', 'END_DATA_BLOCK', 'END_EFFECT', 'END_EFFECT_GROUP', 'END_ELEMENT', 'END_FOR', 'END_FOREACH', 'END_FUNCTION', 'END_FUNCTION_BLOCK', 'END_IF', 'END_INTERFACE', 'END_INTERLOCK', 'END_INTERSECTIONS', 'END_LIBRARY', 'END_NAMESPACE', 'END_NETWORK', 'END_NAMESPACE', 'END_NETWORK', 'END_NODE', 'END_ORGANIZATION_BLOCK', 'END_PROGRAM', 'END_REGION', 'END_REPEAT', 'END_REQUIRE', 'END_RESOURCE', 'END_RUNG', 'END_SELECTION', 'END_SEQUENCE', 'END_SIMULTAN', 'EOUS_BRANCH', 'END_STEP', 'END_STRUCT', 'END_SUPERVISION', 'END_SYSTEM_FUNCTION', 'END_SYSTEM_FUNCTION_BLOCK', 'END_TRANSITION', 'END_TYPE', 'END_VAR', 'END_WHILE', 'END_WIRE', 'ENO', 'ENTRY', 'EQ', 'EW', 'EXIT', 'EXPT', 'EXTENDS', 'F_EDGE', 'FALSE', 'FAMILY', 'FB', 'FC', 'FINAL', 'FIND', 'FLOOR', 'FOR', 'FOREACH', 'FUNCTION', 'FUNCTION_BLOCK', 'GE', 'GOTO', 'GT', 'IB', 'ID', 'IF', 'IMPLEMENTATION', 'IMPLEMENTS', 'INSERT', 'INSIDE', 'INTERFACE', 'INTERLOCK', 'INTERNAL', 'INTERSECTIONS', 'INTERVAL', 'IW', 'KNOW_HOW_PROTECT', 'LABEL', 'LB', 'LD', 'LDATE', 'LE', 'LDATE_AND_TIME', 'LEFT', 'LEN', 'LIBRARY', 'LIMIT', 'LN', 'LOG', 'LOWER_BOUND', 'LT', 'LTIME_OF_DAY', 'LW', 'MAX', 'MB', 'MD', 'MDD_CHECK', 'METHOD', 'MID', 'MIN', 'MOD', 'MOVE', 'MUL', 'MUX', 'MW', 'NAME', 'NAME_OF', 'NAMESPACE', 'NE', 'NETWORK', 'NODE', 'NON_RETAIN', 'NOP', 'NOT', 'NU', 'NULL', 'OB', 'OF', 'ON', 'OR', 'ORGANIZATION_BLOCK', 'OS', 'OV', 'OVERLAP', 'OVERRIDE', 'PA', 'PAB', 'PACKED', 'PAD', 'PAW', 'PB', 'PE', 'PD', 'PEB', 'PED', 'PEW', 'PI', 'PIB', 'PID', 'PIW', 'POST_OPERATION', 'PQ', 'PQB', 'PQD', 'PQW', 'PRAGMA_BEGIN', 'PRAGMA_END', 'PRE_OPERATION', 'PRIORITY', 'PRIVATE', 'PROGRAM', 'PROTECTED', 'PUBLIC', 'PW', 'QB', 'QD', 'QW', 'R_EDGE', 'READ_ONLY', 'READ_WRITE', 'REF', 'REF_TO', 'REGION', 'RELATION', 'REPEAT', 'REPLACE', 'REQUIRE', 'RESOURCE', 'RET_VAL', 'RETAIN', 'RETURN', 'RIGHT', 'ROL', 'ROR', 'RUNG', 'S5T', 'SDB', 'SEL', 'SELECTION', 'SEQUENCE', 'SFB', 'SFC', 'SHL', 'SHR', 'SIMULTANEOUS_BRANCH', 'SIN', 'SINGLE', 'SQRT', 'STANDARD', 'STEP', 'STW', 'SUB', 'SUBSET', 'SUPER', 'SUPERVISION', 'SYSTEM_FUNCTION', 'SYSTEM_FUNCTION_BLOCK', 'TAN', 'TASK', 'THEN', 'THIS', 'TITLE', 'TO', 'TO_BOOL', 'TO_BYTE', 'TO_CHAR', 'TO_DATE', 'TO_DINT', 'TO_DT', 'TO_DWORD', 'TO_INT', 'TO_LDATE', 'TO_LDT', 'TO_LINT', 'TO_LREAL', 'TO_LTIME', 'TO_LTOD', 'TO_LWORD', 'TO_REAL', 'TO_SINT', 'TO_TIME', 'TO_TOD', 'TO_UDINT', 'TO_UINT', 'TO_ULINT', 'TO_USINT', 'TO_WCHAR', 'TO_WORD', 'TOD', 'TRANSITION', 'TRUE', 'TRUNC', 'TYPE', 'UDT', 'UNLINKED', 'UNTIL', 'UO', 'UPPER_BOUND', 'USING', 'USTRING', 'VAR', 'VAR_ACCESS', 'VAR_CONFIG', 'VAR_EXTERNAL', 'VAR_GLOBAL', 'VAR_IN_OUT', 'VAR_INPUT', 'VAR_OUTPUT', 'VAR_TEMP', 'VERSION', 'WHILE', 'WIRE', 'WITH', 'XOR']
def api_info(data,score):
    for i in range(len(data)):
        code_text=data[i]["answer"]
        keyword_count = {}
        for kw in api_keywords:
            count = code_text.count(kw)
            if count > 0:
                keyword_count[kw] = count

        score[i] += sum(keyword_count.values())


data=load_sample(raw_fixed_example_path)
score=[0]*len(data)
line_num(data,score)
print(score)
structural_info(data,score)
print(score)
api_info(data,score)
print(score)
for i in range(len(data)):
    data[i]["score"]=score[i]

with open(fixed_example_path, 'w', encoding='utf-8') as f:
    for d in data:
        json_line = json.dumps(d, ensure_ascii=False)
        f.write(json_line + '\n')


