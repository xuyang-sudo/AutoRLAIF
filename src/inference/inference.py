import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import pandas as pd
from transformers import (
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    DataCollatorWithPadding
)
from peft import PeftModel

@dataclass
class Config:
    gemma_dir = '/kaggle/input/gemma-2/transformers/gemma-2-9b-it-4bit/1/gemma-2-9b-it-4bit'
    lora_dir = '/kaggle/input/73zap2gx/checkpoint-5748'  # 替换为你的模型路径
    max_length = 2048
    batch_size = 4
    device = torch.device("cuda")
    tta = True  # Test-Time Augmentation
    spread_max_length = False

def process_text(text: str) -> str:
    return " ".join(eval(text, {"null": ""}))

def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=2048, spread_max_length=False
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    if spread_max_length:
        prompt = tokenizer(prompt, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_a = tokenizer(response_a, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_b = tokenizer(response_b, max_length=max_length//3, truncation=True, padding=False).input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        attention_mask = [[1]* len(i) for i in input_ids]
    else:
        text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
    return input_ids, attention_mask

@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, tokenizer, device, batch_size=4, max_length=2048):
    a_win, b_win, tie = [], [], []
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = DataCollatorWithPadding(tokenizer=tokenizer)(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.softmax(-1).cpu()
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie
    return df

def main():
    cfg = Config()

    test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
    test['prompt'] = test['prompt'].apply(process_text)
    test['response_a'] = test['response_a'].apply(process_text)
    test['response_b'] = test['response_b'].apply(process_text)

    tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir)
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "right"

    data = pd.DataFrame()
    data["id"] = test["id"]
    data["input_ids"], data["attention_mask"] = tokenize(
        tokenizer, test["prompt"], test["response_a"], test["response_b"],
        max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
    )
    data["length"] = data["input_ids"].apply(len)

    aug_data = pd.DataFrame()
    aug_data["id"] = test["id"]
    aug_data['input_ids'], aug_data['attention_mask'] = tokenize(
        tokenizer, test["prompt"], test["response_b"], test["response_a"],
        max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
    )
    aug_data["length"] = aug_data["input_ids"].apply(len)

    device_0 = torch.device('cuda:0')
    model_0 = Gemma2ForSequenceClassification.from_pretrained(
        cfg.gemma_dir,
        device_map=device_0,
        use_cache=False,
    )
    model_0 = PeftModel.from_pretrained(model_0, cfg.lora_dir)

    device_1 = torch.device('cuda:1')
    model_1 = Gemma2ForSequenceClassification.from_pretrained(
        cfg.gemma_dir,
        device_map=device_1,
        use_cache=False,
    )
    model_1 = PeftModel.from_pretrained(model_1, cfg.lora_dir)

    st = time.time()
    data = data.sort_values("length", ascending=False)
    sub_1 = data.iloc[0::2].copy()
    sub_2 = data.iloc[1::2].copy()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(
            inference, 
            [sub_1, sub_2], 
            [model_0, model_1], 
            [tokenizer, tokenizer], 
            [device_0, device_1], 
            [cfg.batch_size]*2, 
            [cfg.max_length]*2
        )

    result_df = pd.concat(list(results), axis=0)
    proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values
    print(f"Elapsed time: {time.time() - st}")

    if cfg.tta:
        data = aug_data.sort_values("length", ascending=False)
        sub_1 = data.iloc[0::2].copy()
        sub_2 = data.iloc[1::2].copy()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(
                inference, 
                [sub_1, sub_2], 
                [model_0, model_1], 
                [tokenizer, tokenizer], 
                [device_0, device_1], 
                [cfg.batch_size]*2, 
                [cfg.max_length]*2
            )

        tta_result_df = pd.concat(list(results), axis=0)
        tta_proba = tta_result_df[["winner_model_b", "winner_model_a", "winner_tie"]].values
        proba = (proba + tta_proba) / 2
        print(f"Elapsed time with TTA: {time.time() - st}")

    result_df.loc[:, "winner_model_a"] = proba[:, 0]
    result_df.loc[:, "winner_model_b"] = proba[:, 1]
    result_df.loc[:, "winner_tie"] = proba[:, 2]
    submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
    submission_df.to_csv('submission.csv', index=False)
    print(submission_df.head())

if __name__ == "__main__":
    main()
