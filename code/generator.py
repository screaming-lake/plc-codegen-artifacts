import json
import os
from numpy import sort
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import util as st_utils
from nltk.corpus import stopwords
import jieba
from typing import List
from BM25Retriever import BM25Retriever
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import nltk

model_name="llama3.1"
model_path=""
sample_path="..\data\samples_clean.jsonl"
qus_path="..\data\questions.jsonl"
embedding_model_path=""
device="cuda"
res_dir="../res/"
res_path="baseline/"
fixed_example_path="../data/fixed_examples.jsonl"
topN=15

class Question_Dataset_Instance:
    def __init__(self, ins):
        self.ins = ins

    def nl_user_query(self):
        return self.ins['description']
    
    def get_name(self):
        return self.ins['name']

    def get_json(self):
        return self.ins

#dense
class Embedding_Example_Encoder:
    def __init__(self):
        assert os.path.exists(embedding_model_path)
        self.device = device
        word_embedding_model = models.Transformer(embedding_model_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        sentence_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.encoder = sentence_encoder.to(self.device)
        self.samples = load_sample()

        for sample in self.samples:
            description = sample['description']
            sample['description_embedding'] = self.encoder.encode(description, convert_to_tensor=True, device=self.device)

    def get_semantic_similarity(self, ins_query, exp_id):
        query_embedding = self.encoder.encode(ins_query, convert_to_tensor=True, device=self.device)
        sample = self.samples[int(exp_id)]
        target_embedding = sample['description_embedding']
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, target_embedding)[0].detach().cpu().numpy().item()
        return cos_scores

#BM25
class BM25_Example_Encoder:
    def __init__(self):
        nltk.download('stopwords')
        self.samples = load_sample()
        documents = []
        for sample in self.samples:
            doc = Document(text=sample['answer'])
            documents.append(doc)

        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)

        self.retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=len(documents), tokenizer=self.en_tokenizer)
        
    def chinese_tokenizer(text: str) -> List[str]:
        tokens = jieba.lcut(text)
        return [token for token in tokens if token not in stopwords.words('chinese')]
    
    def code_tokenizer(text: str) -> List[str]:
        """
        简单的代码tokenizer，直接按空格分词，保留所有信息
        """
        tokens = text.split()
        return [token for token in tokens if token]
    
    def en_tokenizer(text: str) -> List[str]:
        tokens = text.split()
        return [token for token in tokens if token not in stopwords.words('english')]

    def get_bm25_similarity(self, ins_query):
        nodes = self.retriever.retrieve(ins_query)
        texts = []
        for node in nodes:
            texts.append(node.text)
        return texts

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

def load_text_generation_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

def get_examples_by_ins_embedding(ins, expEncoder):
    assert isinstance(ins, Question_Dataset_Instance)
    samples = load_sample(sample_path)

    semantic_similarity = lambda sample: expEncoder.get_semantic_similarity(ins.nl_user_query(), sample['ID'])

    sort_functions = [
        lambda sample: -semantic_similarity(sample) * 12e2,
    ]
    sorted_samples = sorted(samples, key=lambda exp: sum([sort_fn(exp) for sort_fn in sort_functions]), reverse= not True)
    return sorted_samples

def get_examples_by_ins_BM25(ins, expEncoder):
    assert isinstance(ins, Question_Dataset_Instance)
    samples = load_sample(sample_path)
    desription2sample={}
    for s in samples:
        desription2sample[s['description']]=s

    examples_desription = expEncoder.get_bm25_similarity(ins.nl_user_query())

    examples = []
    for description in examples_desription:
        examples.append(desription2sample[description])
    
    return examples

def get_fixed_example():
    samples = load_sample(fixed_example_path)
    samples.sort(key=lambda x: x['score'], reverse=True)
    return samples

def merge_examples(ins,topN):
    num=topN//3
    samples = load_sample(sample_path)
    name2sample={}
    for s in samples:
        name2sample[s['name']]=s
    result_names = set()
    fixed_examples=get_fixed_example()
    bm25_examples=get_examples_by_ins_BM25(ins, BM25_Example_Encoder())
    dense_examples=get_examples_by_ins_embedding(ins, Embedding_Example_Encoder())
    for i in range(num):
        result_names.add(fixed_examples[i]["name"])
        result_names.add(bm25_examples[i]["name"])
        result_names.add(dense_examples[i]["name"])

    result_samples=[]
    for name in result_names:
        result_samples.append(name2sample[name])
    print(f"Find {len(result_samples)} examples.")

    return result_samples

def build_code_template(json_obj):
    #参数信息
    json_obj["var"]=""
    if json_obj["input"]:
        json_obj["var"]+="\n    VAR_INPUT"
        for i in json_obj["input"]:
            i["description"]=i["description"].replace("\n",' ')
            if "fields" not in i.keys():
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+";"+"    //"+i["description"]
            else:
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+"    //"+i["description"]
                if type(i["fields"])==list:
                    for j in i["fields"]:
                        j["description"]=j["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+j["name"]+" : "+j["type"]+";"+"    //"+j["description"]
                else:
                    # print(type(i["fields"]))
                    if "description" in i["fields"].keys():
                        i["fields"]["description"]=i["fields"]["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+i["fields"]["name"]+" : "+i["fields"]["type"]+";"+"    //"+i["fields"]["description"]
                    else:
                        for k in i["fields"].keys():
                            i["fields"][k]["description"]=i["fields"][k]["description"].replace("\n",' ')
                            json_obj["var"]+="\n            "+i["fields"][k].get("name","name")+" : "+i["fields"][k].get("type","type")+";"+"    //"+i["fields"][k].get("description","description")
                json_obj["var"]+="\n        "+"END_STRUCT;"
        json_obj["var"]+="\n    END_VAR\n"
    if json_obj["output"]:
        json_obj["var"]+="\n    VAR_OUTPUT"
        for i in json_obj["output"]:
            i["description"]=i["description"].replace("\n",' ')
            if "fields" not in i.keys():
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+";"+"    //"+i["description"]
            else:
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+"    //"+i["description"]
                if type(i["fields"])==list:
                    for j in i["fields"]:
                        j["description"]=j["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+j["name"]+" : "+j["type"]+";"+"    //"+j["description"]
                else:
                   # print(type(i["fields"]))
                    if "description" in i["fields"].keys():
                        i["fields"]["description"]=i["fields"]["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+i["fields"]["name"]+" : "+i["fields"]["type"]+";"+"    //"+i["fields"]["description"]
                    else:
                        for k in i["fields"].keys():
                            i["fields"][k]["description"]=i["fields"][k]["description"].replace("\n",' ')
                            json_obj["var"]+="\n            "+i["fields"][k].get("name","name")+" : "+i["fields"][k].get("type","type")+";"+"    //"+i["fields"][k].get("description","description")
                json_obj["var"]+="\n        "+"END_STRUCT;"
        json_obj["var"]+="\n    END_VAR\n"
    if json_obj["in/out"]:
        json_obj["var"]+="\n    VAR_IN_OUT"
        for i in json_obj["in/out"]:
            i["description"]=i["description"].replace("\n",' ')
            if "fields" not in i.keys():
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+";"+"    //"+i["description"]
            else:
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+"    //"+i["description"]
                if type(i["fields"])==list:
                    for j in i["fields"]:
                        j["description"]=j["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+j["name"]+" : "+j["type"]+";"+"    //"+j["description"]
                else:
                    # print(type(i["fields"]))
                    if "description" in i["fields"].keys():
                        i["fields"]["description"]=i["fields"]["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+i["fields"]["name"]+" : "+i["fields"]["type"]+";"+"    //"+i["fields"]["description"]
                    else:
                        for k in i["fields"].keys():
                            i["fields"][k]["description"]=i["fields"][k]["description"].replace("\n",' ')
                            json_obj["var"]+="\n            "+i["fields"][k].get("name","name")+" : "+i["fields"][k].get("type","type")+";"+"    //"+i["fields"][k].get("description","description")
                json_obj["var"]+="\n        "+"END_STRUCT;"
        json_obj["var"]+="\n    END_VAR\n"
    json_obj["var"]+='''
    VAR 
        【】
    END_VAR
    
    VAR_TEMP
        【】
    END_VAR
    
    VAR CONSTANT
        【】
    END_VAR'''
    
    if json_obj["type"]=="FUNCTION_BLOCK":
        p_format='''{type} "{name}"
{{ S7_Optimized_Access := 'TRUE' }}
    {var}

BEGIN
    【】
END_{type}
    '''
    elif json_obj["type"]=="FUNCTION":
        #类型为函数且没有返回值，return_value置为Void
        if not json_obj["return_value"]:
            json_obj["return_value"].append({"type":"Void","description":""})
        p_format='''{type} "{name}" : {return_value[0][type]}
{{ S7_Optimized_Access := 'TRUE' }}
    {var}

BEGIN
    【】
END_{type}
''' 
    return p_format.format(**json_obj)

def build_question_and_sample(json_obj, is_sample=False):
    prompt_format = "title:{title}\n\ndescription:{description}\n\ntype:{type}\n\nname:{name}\n\n"

    if is_sample:
        answer = json_obj["answer"]
    else:
        answer = None
    
    api_info="code template:\n"
    api_info+=build_code_template(json_obj)

    if is_sample:
        #COT
        # api_info+="\n逻辑步骤：\n"+json_obj["cot"]
        api_info += "\nanswer:\n"+"```SCL\n"
        api_info += answer.strip() + "\n```\n"
    # else:
        # api_info += "\n逻辑步骤：\n"
    prompt = prompt_format.format(**json_obj) + api_info
    return prompt

def build_prompt(ins,examples):
    sys_prompt="Based on reference program examples, complete the code filling task by writing an SCL program that runs on the Siemens TIA Portal platform. You need to fill in the correct code within the 【】 to achieve the corresponding functionality."
    FINAL_PROMPT = '''examples:
<public example>
{public_example}
</public example>

The following is the code that needs to be generated. You need to fill in the correct code within the 【】 to achieve the corresponding functionality.
{complete_user_query}

'''
    public_example = ''
    for example in examples:
        public_example += build_question_and_sample(example, is_sample=True)+"\n-------------------------------------------------\n"
    complete_user_query = build_question_and_sample(ins.get_json(), is_sample=False)

    variables = {
        "complete_user_query": complete_user_query,
        "public_example": public_example,
    }
    final_prompt = FINAL_PROMPT.format(**variables)
    return sys_prompt,final_prompt

def send2llm(sys_prompt,prompt,tokenizer, model):
    final_prompt=sys_prompt+'\n'+prompt
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

def post_process(text):
    code=text
    return code

def write_scl(name,code):
    f=open(res_dir+res_path+name+".scl",'w')
    f.write(code)
    f.close()

def code_generation(data,tokenizer, model):
    qus=Question_Dataset_Instance(data)
    examples=merge_examples(qus,topN)
    sys_prompt,prompt=build_prompt(qus,examples)
    # print(prompt)
    predict_res=send2llm(sys_prompt,prompt,tokenizer, model)
    code=post_process(predict_res)
    write_scl(qus.get_name(),code)
    

if __name__ == '__main__':
    samples=load_sample(sample_path)
    questions=load_sample(qus_path)
    tokenizer, model=load_text_generation_model()
    for qus in questions:
        code_generation(qus,tokenizer, model)
