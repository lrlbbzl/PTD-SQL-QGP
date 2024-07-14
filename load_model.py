from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

from transformers import AutoModel

MODELSCOPE_CAHCE='/apdcephfs/private_rileyrlluo/models/LLM'

model_dir = snapshot_download("deepseek-ai/deepseek-vl-1.3b-chat", cache_dir='/apdcephfs/private_rileyrlluo/models/LLM')

# huggingface-cli download --resume-download deepseek-ai/deepseek-vl-1.3b-chat --local-dir './'
