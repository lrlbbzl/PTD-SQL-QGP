import os 
import argparse
from functools import partial
from tqdm import tqdm
from datasets import load_dataset
import json
import torch
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

from torch import nn
from transformers import get_linear_schedule_with_warmup
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

query_template = """### Instruction:
You are a text-to-sql expert. Your task is to classify text-based queries. The types are defined as follows:
1. Set operations, which require complex logical connections between multiple conditions and often involve the use of intersect, except, union, and other operations; 
2. Combination operations, which require grouping of specific objects and finding the maximum value or sorting, often achieved using GROUP BY; 
3. Filtering problems, which select targets based on specific judgment conditions, generally completed using where statements; 
4. Other simple problems, including simple retrieval and sorting.

Your task is to judge the query step by step to see if it belongs to a certain category. For example, if you think the query has the characteristics of the first type, then classify it as the first type without considering the subsequent types. If you think the query does not have the characteristics of the first type but has the second type, then classify it as the second type without considering the subsequent types.

## Example 1:
What are the ids of the students who either registered or attended a course?
Type: Set operations

## Example 2:
List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.
Type: Set operations

## Example 3:
Find all the zip codes in which the max dew point have never reached 70.
Type: Set operations

## Example 4:
Find the name of customers who do not have an saving account.
Type: Set operations

## Example 5:
Which origin has most number of flights?
Type: Combination operations

## Example 6:
Which course is enrolled in by the most students? Give me the course name.
Type: Combination operations

## Example 7:
Find the name of the train whose route runs through greatest number of stations.
Type: Combination operations

## Example 8:
What are the names of musicals with nominee "Bob Fosse"?
Type: Filtering problems

## Example 9:
How many distinct kinds of camera lenses are used to take photos of mountains in the country 'Ethiopia'? 
Type: Filtering problems

## Example 10:
How many products are there?
Type: Other simple problems


### Input:
{query}

### Response:
"""

response_template = "Type: {response}"

@dataclass
class hit1:
    right: int = 0
    total: int = 0
    sentence: str = "The missing entity of query quadruplet is "
    def update(self, response, answer):
        st = response.find(self.sentence) + len(self.sentence)
        response = response[st : ]
        if response.find('.\n') != -1:
            response = response[:response.find('.\n')]
        elif response.find('.</s>') != -1:
            response = response[:response.find('.</s>')]
        else:
            response = response[:response.find('.')]
        if response == answer:
            self.right += 1
        self.total += 1
    
    def acc(self, ):
        return self.right / self.total
    
def generate_prompt(input, label=None):
    # schema1 = r"\{query\}"
    # res = re.sub(schema1, input, self.query_template)
    # if label is not None:
    #     schema2 = r"\{response\}"
    #     label = re.sub(schema2, label, self.response_template)
    #     res = f"{res}{label}"
    res = query_template.replace("{query}", input)
    if label is not None:
        label = response_template.replace("{response}", label)
        res = f"{res}{label}"
    return res

def inference(args):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_map = "auto"
    
    base_model_path = os.path.join(args.base_model_path, args.base_model)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                             torch_dtype=torch.float16, device_map=device_map)
    model = PeftModel.from_pretrained(model, args.lora_weights_path, torch_dtype=torch.float16)


    if args.add_prefix:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    model.eval()


    # Prepare things such as prompts list for test data;
    ## During fine-tuning, we select valid dataset as test set and create history in train dataset
    ## During testing here, we evaluate on test dataset and create history in train + valid dataset

    


    prompts = json.load(open(args.data_path, 'r'))
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.9,
        # top_p=0.92,
        # top_k=100,
        num_beams=4, # beam search
        # no_repeat_ngram_size=2
    )
    result = []
    prompts = prompts[args.partial_num : ]
    with torch.no_grad(), tqdm(prompts) as tbar:
        for test_sample in tbar:
            query, answer = test_sample['query'], test_sample['response']
            prompt = generate_prompt(query)
            model_inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = model_inputs.input_ids.to(device)

            generate_ids = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                max_new_tokens=50,
                # output_scores=True,
                # renormalize_logits=True,
            )

            inputs_text = tokenizer.decode(input_ids[0])
            output = tokenizer.decode(generate_ids.sequences[0]).replace(inputs_text, "")
            if args.check_example:
                print('*' * 20)
                print(output)
                print('-' * 20)
                print(answer)
            
            result.append(
                {
                    "query": query,
                    "predict": output
                }
            )
    json.dump(result, open(os.path.join(args.output_dir, 'results.json'), 'w'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TKGC inference')
    
    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument("--dataset", type=str, choices=['ICEWS14', 'YAGO', 'WIKI', 'ICEWS18', 'ICEWS05-15'], default='ICEWS14', help='select dataset')
    parser.add_argument("--save-path", type=str, default='./pretrained_emb', help='embedding save path')
    parser.add_argument("--template-path", type=str, default='./templates', help='prompt template path')
    parser.add_argument("--prompt-path", type=str, default='./prompts', help='prompt save path')
    parser.add_argument("--base-model-path", type=str, default='./models/modelscope', help='base llm')
    parser.add_argument("--base-model", type=str, default='Llama-2-7b-ms', help='base llm')
    parser.add_argument("--gpu", type=int, default=1, help='gpu id')
    parser.add_argument("--lora-weights-path", type=str, default='./outputs', help='lora save path')
    parser.add_argument("--half", type=bool, default=False, help='half precision')
    parser.add_argument("--add-prefix", action='store_true', help='whether use kge prefix')
    parser.add_argument("--inference-direction", type=str, default='right', choices=['right', 'left', 'bi'])
    parser.add_argument("--data-augment", action='store_true',)
    parser.add_argument("--partial-num", type=int, default=0, help='inference start point')
    parser.add_argument("--useid", action='store_true')

    parser.add_argument("--history-length", type=int, default=8, help='history references')
    parser.add_argument("--output-dir", type=str, default='./outputs', help='output dirs')
    parser.add_argument("--add-reciprocal", type=bool, default=False, help='whether do reverse reasoning')
    parser.add_argument("--check-example", action='store_true', help='whether print the example to check')
    args = parser.parse_args()
    inference(args)
