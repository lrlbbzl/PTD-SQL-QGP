import torch
import argparse
import warnings
from functools import partial
warnings.filterwarnings("ignore")
import logging
from copy import deepcopy
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
import pickle
import os 
import json
import sys
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler

from torch.optim import AdamW
from torch import nn
from transformers import get_linear_schedule_with_warmup
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)

from datasets import Dataset


import warnings
warnings.filterwarnings("ignore")

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

def run(args):
    local_rank = args.local_rank
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    logging.warning("Process rank: %s, device: %s, distributed training: %s",
                    args.local_rank, device, bool(args.local_rank != -1))



    if args.do_finetune:
        logging.info('*' * 20 + 'Start fine-tuning' + '*' * 20)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        llm_path = os.path.join(args.base_model_path, args.base_model)
        gradient_accumulation_steps = args.batch_size // args.sm_batch_size

        ## training setting
        device_map = "auto"
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        #     gradient_accumulation_steps = gradient_accumulation_steps // world_size



        ## Tokenize
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        def tokenize(prompt, tokenizer, length_limit, add_eos_token=False):
            # there's probably a way to do this with the tokenizer settings
            # but again, gotta move fast
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=length_limit,
                padding=False,
                return_tensors=None,
            )
            # if (
            #     result["input_ids"][-1] != tokenizer.eos_token_id
            #     and len(result["input_ids"]) < length_limit
            #     and add_eos_token
            # ):
            #     result["input_ids"].append(tokenizer.eos_token_id)
            #     result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result

        def generate_and_tokenize_prompt(data_point):
            full_prompt = generate_prompt(
                data_point["query"],
                data_point["response"],
            )

            full_tokenized = tokenize(full_prompt, tokenizer, args.truncation_length, add_eos_token=True)
            user_prompt = generate_prompt(
                data_point["query"]
            )
            user_tokenized = tokenize(user_prompt, tokenizer, args.truncation_length)
            user_length = len(user_tokenized["input_ids"])
            mask_token = [-100] * user_length
            full_tokenized["labels"] = mask_token + full_tokenized["labels"][user_length : ]
            return full_tokenized

        data = load_dataset('json', data_files=args.data_path)
        if args.partial_num != -1:
            data['train'] = Dataset.from_dict(data['train'][:args.partial_num])
        # partial_func = partial(generate_and_tokenize_prompt, prompter=prompter, tokenizer=tokenizer, length_limit=args.truncation_length, if_test=False)
        # train_data = data["train"].shuffle().map(partial_func)
        train_data = data["train"].map(generate_and_tokenize_prompt)
        train_data = train_data.remove_columns(data['train'].column_names)
        val_data = None

        
        ## create peft model and trainer
        # if not ddp and torch.cuda.device_count() > 1:
        #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        #     model.is_parallelizable = True
        #     model.model_parallel = True

        ## Prepare model
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        if args.prepare_kbit:
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, peft_config)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


        training_args = TrainingArguments(
            per_device_train_batch_size=args.sm_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=500,
            num_train_epochs=args.n_ft_epoch,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=100,
            save_strategy="steps",
            eval_steps=None,
            save_steps=5000,
            output_dir=args.output_dir,
            save_total_limit=2,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=False,
            report_to='wandb',
            run_name=args.run_name,
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=None,
            args=training_args,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        model.module.config.use_cache = False

        # old_state_dict = model.state_dict
        # model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(
        #         self, old_state_dict()
        #     )
        # ).__get__(model, type(model))

        # import sys
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)

        trainer.train()

        model.module.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM for TKGC')

    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument("--dataset", type=str, default='ICEWS14', help='select dataset', choices=['ICEWS14', 'ICEWS18', 'ICEWS05-15', 'YAGO', 'WIKI'])
    parser.add_argument("--save-path", type=str, default='./pretrained_emb', help='embedding save path')
    parser.add_argument("--template-path", type=str, default='./templates', help='prompt template path')
    parser.add_argument("--prompt-path", type=str, default='./prompts', help='prompt save path')
    parser.add_argument("--base-model-path", type=str, default='./models', help='base llm')
    parser.add_argument("--base-model", type=str, default='Llama-2-7b-ms', help='base llm')
    parser.add_argument("--gpu", type=int, default=1, help='gpu id')
    parser.add_argument("--local_rank", type=int)

    # Configure for global KGE
    parser.add_argument("--hidden-size", type=int, default=200, help='hidden size for KGE')
    parser.add_argument("--global-gnn", type=str, default='rgat', help='type of gnn in global graph')
    parser.add_argument("--global-heads", type=int, default=4, help='heads of attention during RGAT')
    parser.add_argument("--global-layers", type=int, default=1, help='numbers of propagation')
    parser.add_argument("--n-global-epoch", type=int, default=200, help='KGE epochs')
    parser.add_argument("--gcn", action='store_true', help='whether use rgcn or some other gnn models during pretraining')
    parser.add_argument("--score-func", type=str, default='RotatE', help='KGE model for optimization')
    parser.add_argument("--kge-lr", type=str, default=1e-4, help='learning rate in KGE phase')
    parser.add_argument("--weight-decay", type=float, default=1e-6, help='weight decay for optimizer')
    parser.add_argument("--kge-batch-size", type=int, default=500, help='batch size in KGE')
    parser.add_argument("--grad-norm", type=float, default=1., help='grad norm during training')
    parser.add_argument("--add-prefix", action='store_true', help='whether add prefix embedding')
    # Configure for phase
    parser.add_argument("--do-pretrain", action='store_true', help='whether pretrain KGE')
    parser.add_argument("--do-finetune", action='store_true', help='whether fine-tuning')

    # Configure for LLM fine-tune
    parser.add_argument("--batch-size", type=int, default=8, help='fine-tuning batch size')
    parser.add_argument("--sm-batch-size", type=int, default=2, help='small batch size')
    parser.add_argument("--n-ft-epoch", type=int, default=2, help='fine-tuning epoch')
    parser.add_argument("--prepare-kbit", action='store_true', help='whether prepare for kbit training')
    parser.add_argument("--inference-direction", choices=['right', 'left', 'bi'], default='right', type=str, help='type of data used')
    parser.add_argument("--lr", type=float, default=2e-5, help='learning rate during fine-tuning')
    parser.add_argument("--truncation-length", type=int, default=3000, help='truncation length limit')
    parser.add_argument("--train-on-inputs", type=bool, default=True, help='whether training on inputs data')
    parser.add_argument("--add-eos-tokens", type=bool, default=False, help='whether adding eos')
    parser.add_argument("--prompt-template", type=str, default='llama', help='prompt template')
    parser.add_argument("--data-augment", action='store_true', help='whether use other information to pad history')
    parser.add_argument("--partial-num", type=int, default=-1, help='set fine-tune number of samples')
    parser.add_argument("--useid", action='store_true')
    # Configure for lora
    parser.add_argument("--lora-rank", type=int, default=32, help='lora rank')
    parser.add_argument("--lora-alpha", type=int, default=16, help='lora alpha')
    parser.add_argument("--lora-dropout", type=float, default=0.05, help='dropout rate during ft')
    parser.add_argument("--lora-target-modules", type=List[str], default=['q_proj', 'k_proj', 'v_proj', 'o_proj'], help='lora target modules')

    # Configure for other places
    parser.add_argument("--history-length", type=int, default=8, help='history references')
    parser.add_argument("--val-size", type=int, default=0, help='vaild dataset length')
    parser.add_argument("--output-dir", type=str, default='./outputs', help='output dirs')
    parser.add_argument("--logging-dir", type=str, default='./logs', help='logs save dir')
    parser.add_argument("--add-reciprocal", type=bool, default=False, help='whether do reverse reasoning')
    parser.add_argument("--run-name", type=str, default='llama-2-7b', help='tag for checking in wandb')
    args = parser.parse_args()

    # start
    run(args)
