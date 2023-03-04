import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import requests
import json

tokenizer = AutoTokenizer.from_pretrained(
'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)

model = AutoModelForCausalLM.from_pretrained(
'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
pad_token_id=tokenizer.eos_token_id,
torch_dtype='auto', low_cpu_mem_usage=True,
device_map="auto", load_in_8bit=True,
)
model.eval()

def generate(text, max_new_tokens,  num_sequences, do_sample, top_k, top_p, temperature):
    tokenized_text = tokenizer(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                tokenized_text.input_ids,
                do_sample=do_sample, #샘플링 전략 사용
                max_new_tokens= max_new_tokens, # 최대 디코딩 길이
                top_k=top_k, # 확률 순위 밖인 토큰은 샘플링에서 제외
                top_p=top_p, # 누적 확률 이내의 후보집합에서만 생성
                temperature=temperature,
                no_repeat_ngram_size = 4,
                num_return_sequences = num_sequences # 한 번에 출력할 다음 문장 선택지의 개수
            )
        except:
            generated_ids = [''] * num_sequences
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    generated_texts = [tokenizer.decode(id, skip_special_tokens=True) for id in generated_ids]
    new_sentences = [sent.split(text)[-1] for sent in generated_texts]
    return new_sentences

app = FastAPI(title='kogpt',
              description='kakaobrain-kogpt6B-fp16',
              version="1.0")


class textInput(BaseModel):
    """Input model for prediction
    """
    text: str = Field(None, description='Prompt', example="여기에 입력")
    max_new_tokens: int = Field(64, description='Maximum length of generated sequence')
    num_sequences: int = Field(1, description="Number of sequences to generate")
    do_sample: bool = Field(True, description="Use sampling for generation")
    top_k: int = Field(0, description="Top K for generation")
    top_p: float = Field(0.8, description="Top P for generation")
    temperature:float = Field(0.19, description="Temperature for generation") 
    

class textResponse(BaseModel):
    prompt: str = Field(None, description="Original prompt for generation")
    generated: List[str] = Field(None, description="generated texts")
    num_sequences: int = Field(None, description="Number of generated sequences")


@app.get("/")
def home():
    return "Refer to '/docs' for API documentation"


@app.post("/generate", description="Generation", response_model=textResponse)
def get_generation(req_body: textInput):
    """Prediction
    :param req_body:
    :return:
    """
    torch.cuda.empty_cache()
    result = generate(req_body.text,
                      req_body.max_new_tokens,
                      req_body.num_sequences,
                      req_body.do_sample,
                      req_body.top_k,
                      req_body.top_p,
                      req_body.temperature,
                    )
    return {"prompt": req_body.text, "generated":result, "num_sequences":req_body.num_sequences}
