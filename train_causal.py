import os
import gc
import json
import jsonlines
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from utils import (
    set_seed, 
    init_logger,
)
from dataset import CausalLMDataset


# global params
MODEL_CACHE_DIR = "./models"

# Backbone: Codegen (pretrained on data consisting of 119.2B tokens, including C, C++, Go, Java, JavaScript, and Python.)
# https://huggingface.co/Salesforce/codegen-350M-multi
# https://huggingface.co/Salesforce/codegen-2B-multi
# https://huggingface.co/Salesforce/codegen-6B-multi

def parse_args():
    parser = argparse.ArgumentParser(description="Solidity CodeGen")
    parser.add_argument("--seed", type=int, default=23333333)
    # datasets
    parser.add_argument("--trainset", type=str, required=True)
    parser.add_argument("--testset", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    # checkpoints and log saving directory
    parser.add_argument("--basedir", type=str, default="./log")
    parser.add_argument("--expname", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    # model backbone
    parser.add_argument("--backbone", type=str, required=True) # choices=["Salesforce/codegen-16B-multi", "Salesforce/codegen-2B-multi", "Salesforce/codegen-6B-multi", "Salesforce/codegen-350M-multi"]
    # fine-tuning options
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--train_bsz", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--maxlen", type=int, default=1024)
    # checkpoint
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_only", action="store_true")
    # inference
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--topp", type=float, default=0.9)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--infer_maxlen", type=int, default=512)
    args = parser.parse_args()
    return args


def train_epoch(args, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, optimizer, scheduler, dataloader, device):
    gc.collect()
    torch.cuda.empty_cache()
    total_train_loss_value = 0.0
    train_loss_value = 0.0
    accumulation_count = 0
    tqdm_vars = {
        "lr": np.nan,
        "loss": np.nan,
        "norm": np.nan,
    }
    tbar = tqdm(enumerate(dataloader, start=1), desc="train", total=len(dataloader), postfix=tqdm_vars)
    model.train()
    for _, seqs in tbar:
        # for tokenizer options, see 
        # https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#transformers.GPT2Tokenizer
        # https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
        batch_tokenized = tokenizer.batch_encode_plus(
            seqs,
            add_special_tokens=True,
            truncation=True,
            max_length=args.maxlen,
            padding="longest",
            return_offsets_mapping=False,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        # copy tensors to gpu
        input_ids = torch.tensor(batch_tokenized["input_ids"], device=device)
        attention_mask = torch.tensor(batch_tokenized["attention_mask"], device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        train_loss_value += outputs.loss.item()
        accumulation_count += 1
        total_train_loss_value += outputs.loss.item()
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        if accumulation_count % args.gradient_accumulation_steps == 0:
            norm = clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm, norm_type=2).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tqdm_vars["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
            tqdm_vars["loss"] = train_loss_value / args.gradient_accumulation_steps
            tqdm_vars["norm"] = norm
            tbar.set_postfix(tqdm_vars)
            accumulation_count = 0
            train_loss_value = 0.0
    return total_train_loss_value / len(dataloader)


@torch.no_grad()
def eval_epoch(args, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataloader, device):
    gc.collect()
    torch.cuda.empty_cache()
    valid_loss_value = 0.0
    model.eval()
    for i, seqs in tqdm(enumerate(dataloader, start=1), desc="eval", total=len(dataloader)):
        batch_tokenized = tokenizer.batch_encode_plus(
            seqs,
            add_special_tokens=True,
            truncation=True,
            max_length=args.maxlen,
            padding="longest",
            return_offsets_mapping=False,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        # copy tensors to gpu
        input_ids = torch.tensor(batch_tokenized["input_ids"], device=device)
        attention_mask = torch.tensor(batch_tokenized["attention_mask"], device=device)
        # forwarding
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        valid_loss_value += outputs.loss.item()
    return valid_loss_value / len(dataloader)


@torch.no_grad()
def inference(args, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataloader, device):
    # TODO: fix following inference warnings
    # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    # A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
    # Input length of input_ids is 512, but `max_length` is set to 512. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    generated_file = os.path.join(args.basedir, args.expname, "generated.jsonl")
    texts = []
    for i, seqs in tqdm(enumerate(dataloader, start=1), desc="inference", total=len(dataloader)):
        batch_tokenized = tokenizer.batch_encode_plus(
            seqs,
            add_special_tokens=True,
            truncation=True,
            max_length=args.infer_maxlen,
            padding="longest",
            return_offsets_mapping=False,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        # copy tensors to gpu
        input_ids = torch.tensor(batch_tokenized["input_ids"], device=device)
        attention_mask = torch.tensor(batch_tokenized["attention_mask"], device=device)
        # forward
        # see doc: https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=args.infer_maxlen,
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_k=args.topk,
            top_p=args.topp,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            early_stopping=True,
        )
        # print(generated_ids.shape)
        for text in tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            truncate_before_pattern=[
                # "\n\n^#", "^'''", '^"""', '^#', 
                # r"\n\n//", "\n\n\n", "\n\n\r\n", 
                "\n\n", "<|endoftext|>",
            ],
        ):
            # print(text)
            texts.append({"code": text})
    # save generations to file
    with jsonlines.open(generated_file, "w") as f:
        f.write_all(texts)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make logging directory
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(args.basedir, exist_ok=True)
    logging_dir = os.path.join(args.basedir, args.expname)
    if os.path.exists(logging_dir) and not args.overwrite:
        print(f"[WARN] logging directory {logging_dir} already exists. If you want to overwrite previous logs, use param `--overwrite` please.")
        exit(-1)
    os.makedirs(logging_dir, exist_ok=True)
    # build logger
    logger = init_logger(logdir=logging_dir)
    # save configs
    with open(os.path.join(logging_dir, "config.json"), "w+", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    
    # build datasets
    logger.info("Loading datasets...")
    train_valid_dataset = CausalLMDataset(args.trainset)
    train_valid_dataset_sz = len(train_valid_dataset)
    train_dataset_sz = int(args.train_ratio * train_valid_dataset_sz)
    valid_dataset_sz = train_valid_dataset_sz - train_dataset_sz
    train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [train_dataset_sz, valid_dataset_sz])
    test_dataset = CausalLMDataset(args.testset)
    inference_dataset = CausalLMDataset(args.testset, train=False)
    logger.info(f"[DATASET] train set size = {len(train_dataset)}")
    logger.info(f"[DATASET] valid set size = {len(valid_dataset)}")
    logger.info(f"[DATASET] test  set size = {len(test_dataset )}")
    logger.info(f"[DATASET] train samples: {train_valid_dataset.examples(n=2)}")
    logger.info(f"[DATASET] test  samples: { test_dataset.examples(n=2)}")
    logger.info(f"[DATASET] inference  samples: { inference_dataset.examples(n=2)}")
    # build dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers)
    inference_dataloader = DataLoader(inference_dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers)
    # build tokenizer
    config = AutoConfig.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
    config.torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR, config=config)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # save vocab
    with open(os.path.join(MODEL_CACHE_DIR, "models--" + args.backbone.replace("/", "--"), "vocab.json"), "w+", encoding="utf-8") as f:
        json.dump(tokenizer.get_vocab(), f, indent=2)
    # build model
    model = AutoModelForCausalLM.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR, config=config).to(device)
    model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    logger.info(f"[MODEL] configs: {config}")
    
    if not args.test_only:
        # build optimizer
        num_training_steps = args.epoch * len(train_dataloader)
        optimizer = torch.optim.AdamW(model_params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
        # load checkpoint if necessary
        begin_epoch = 0
        if args.checkpoint is not None:
            logger.info(f"[MODEL] Load params from checkpoint: {args.checkpoint}")
            params = torch.load(args.checkpoint)
            begin_epoch = params["epoch"]
            model.load_state_dict(params["model"], strict=True)
            optimizer.load_state_dict(params["optimizer"])
            scheduler.load_state_dict(params["scheduler"])
        # training and validating
        for i in range(begin_epoch+1, args.epoch+1):
            logger.info(f"[EPOCH] {i}")
            train_loss = train_epoch(args, model, tokenizer, optimizer, scheduler, train_dataloader, device)
            logger.info(f"[TRAIN] loss={train_loss}, ppl={np.exp(train_loss)}")
            # save checkpoints
            if i % args.save_every == 0:
                torch.save({
                    "epoch": i,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, os.path.join(logging_dir, f"{i}.pth"))
            valid_loss = eval_epoch(args, model, tokenizer, valid_dataloader, device)
            logger.info(f"[VALID] loss={valid_loss}, ppl={np.exp(valid_loss)}")
    else: # test_only
        if args.checkpoint is None:
            logger.info(f"[WARN] args.checkpoint is None! Use pretrained params of {args.backbone}")
        else:
            logger.info(f"[MODEL] Load params from checkpoint: {args.checkpoint}")
            params = torch.load(args.checkpoint)
            model.load_state_dict(params["model"], strict=True)

    # testing
    logger.info(f"testing...")
    test_loss = eval_epoch(args, model, tokenizer, test_dataloader, device)
    logger.info(f"[TEST] loss={test_loss}, ppl={np.exp(test_loss)}")
    # inference
    logger.info(f"inference...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, padding_side="left", cache_dir=MODEL_CACHE_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inference(args, model, tokenizer, inference_dataloader, device)
