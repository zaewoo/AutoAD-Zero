import os
os.environ['TRANSFORMERS_CACHE'] = "/content/drive/MyDrive/experiments/model/cache"
import sys
import ast
import json
import math
import re
import torch
import random
import argparse
import numpy as np
import transformers
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from promptloader import get_user_prompt
from cache_selector import CacheClipScorer

try:
    from nltk.stem import PorterStemmer as NltkPorterStemmer
except Exception:
    NltkPorterStemmer = None

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:
    ENGLISH_STOP_WORDS = None


FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "am", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at", "for", "from", "by", "with", "as",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "them", "his", "her",
    "their", "you", "your", "i", "we", "our", "me", "my", "mine", "ours", "yours", "do", "does",
    "did", "doing", "have", "has", "had", "having", "not", "no", "so", "than", "then", "too",
    "very", "can", "could", "should", "would", "will", "just", "into", "out", "up", "down", "over",
    "under", "again", "once", "there", "here", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "s", "t",
}


class RuleBasedPorterFallback:
    @staticmethod
    def stem(word: str) -> str:
        w = str(word)
        for suffix in ["ingly", "edly", "ing", "ed", "ies", "sses", "s"]:
            if len(w) > 4 and w.endswith(suffix):
                if suffix == "ies":
                    return w[:-3] + "y"
                if suffix == "sses":
                    return w[:-2]
                return w[: -len(suffix)]
        return w


def basic_tokenize(text: str):
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return [tok for tok in cleaned.split() if tok]


def build_unigram_prior(corpus_texts, alpha=1e-3):
    token_counts = {}
    total_tokens = 0
    for text in corpus_texts:
        tokens = basic_tokenize(text)
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
            total_tokens += 1

    vocab_size = len(token_counts)
    denominator = total_tokens + alpha * max(vocab_size, 1)
    return {
        "counts": token_counts,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "alpha": alpha,
        "denominator": denominator,
    }


def lm_prior_score(caption: str, unigram_prior):
    tokens = basic_tokenize(caption)
    if not tokens:
        return -1e9

    counts = unigram_prior["counts"]
    alpha = unigram_prior["alpha"]
    denominator = unigram_prior["denominator"]
    log_probs = []
    for token in tokens:
        numerator = counts.get(token, 0) + alpha
        log_probs.append(math.log(numerator / denominator))
    return float(sum(log_probs) / len(log_probs))


def make_content_set(caption: str, stopwords, stemmer):
    tokens = basic_tokenize(caption)
    content = [tok for tok in tokens if tok not in stopwords]
    stems = {stemmer.stem(tok) for tok in content if tok}
    return stems


def jaccard_score(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def select_candidate_with_prior(candidates, unigram_prior, score_mode="quality"):
    if not candidates:
        return 0, []

    if ENGLISH_STOP_WORDS is not None:
        stopwords = set(ENGLISH_STOP_WORDS)
    else:
        stopwords = set(FALLBACK_STOPWORDS)

    stemmer = NltkPorterStemmer() if NltkPorterStemmer is not None else RuleBasedPorterFallback()

    content_sets = [make_content_set(caption, stopwords, stemmer) for caption in candidates]
    content_lens = [len(stem_set) for stem_set in content_sets]

    consensus_scores = []
    for i, stem_set in enumerate(content_sets):
        if len(candidates) == 1:
            consensus_scores.append(0.0)
            continue
        pairwise = []
        for j, other_set in enumerate(content_sets):
            if i == j:
                continue
            pairwise.append(jaccard_score(stem_set, other_set))
        consensus_scores.append(float(sum(pairwise) / len(pairwise)) if pairwise else 0.0)

    if score_mode == "safety":
        consensus_weight = 0.5
        content_penalty = 1.5
    else:
        consensus_weight = 0.8
        content_penalty = 0.45

    details = []
    for idx, caption in enumerate(candidates):
        lm_prior = lm_prior_score(caption, unigram_prior)
        score = lm_prior + consensus_weight * consensus_scores[idx] - content_penalty * content_lens[idx]
        details.append({
            "idx": idx,
            "caption": caption,
            "score": float(score),
            "lm_prior": float(lm_prior),
            "consensus": float(consensus_scores[idx]),
            "content_len": int(content_lens[idx]),
        })

    best_detail = max(
        details,
        key=lambda d: (
            d["score"],
            d["consensus"],
            -d["content_len"],
            -len(d["caption"]),
        ),
    )
    return int(best_detail["idx"]), details


def initialise_model(access_token):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=access_token,
    )

    # Llama-family tokenizers often do not define a pad token by default.
    # Batching in HF pipelines requires a valid pad token.
    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    if getattr(pipeline.tokenizer, "pad_token", None) is None and pipeline.tokenizer.eos_token is not None:
        pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    if getattr(pipeline.model.config, "pad_token_id", None) is None:
        pipeline.model.config.pad_token_id = pipeline.tokenizer.pad_token_id

    # Decoder-only LLM batching should use left padding to avoid generation artifacts.
    if getattr(pipeline.tokenizer, "padding_side", None) != "left":
        pipeline.tokenizer.padding_side = "left"

    return pipeline


def summary_batch(
    pipeline,
    user_prompts,
    dataset,
    generation_batch_size=1,
    generation_num_workers=0,
    generation_temperature=0.85,
    generation_top_p=0.93,
    generation_top_k=50,
    generation_repetition_penalty=1.1,
):
    if dataset in ["cmdad", "madeval"]:
        dataset_text = "movie"
    elif dataset in ["tvad"]:
        dataset_text = "TV series"
    else:
        dataset_text = "video"

    sys_prompt = (
        f"You are an intelligent chatbot designed for summarizing {dataset_text} audio descriptions. "
        "Here's how you can accomplish the task:------##INSTRUCTIONS: you should convert the predicted descriptions into one sentence. "
        "You should directly start the answer with the converted results WITHOUT providing ANY more sentences at the beginning or at the end."
    )

    prompts = []
    for user_prompt in user_prompts:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    terminators = [token_id for token_id in terminators if token_id is not None]

    outputs = pipeline(
        prompts,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=float(generation_temperature),
        top_p=float(generation_top_p),
        top_k=int(generation_top_k),
        repetition_penalty=float(generation_repetition_penalty),
        pad_token_id=pipeline.tokenizer.eos_token_id,
        batch_size=max(1, int(generation_batch_size)),
        num_workers=max(0, int(generation_num_workers)),
    )

    output_texts = []
    for prompt, output in zip(prompts, outputs):
        generated = output[0]["generated_text"][len(prompt):]
        output_texts.append(generated)
    return output_texts


def initialise_embedding_model(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


def mean_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


@torch.inference_mode()
def encode_texts(texts, tokenizer, model, batch_size=64):
    embeddings = []
    device = model.device
    for idx in range(0, len(texts), batch_size):
        batch = [str(text).strip() for text in texts[idx:idx + batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)


def build_retrieval_query(text: str, cache_scorer: CacheClipScorer, row) -> str:
    raw = str(text).strip()
    if not raw:
        return ""

    flattened = re.sub(r"\s+", " ", raw)
    pattern = re.compile(
        r"(?:^|\b)2\s*\.\s*actions\s*:\s*(.*?)(?=(?:\b3\s*\.\s*character\s*-?\s*character\s*interactions\s*:)|$)",
        flags=re.IGNORECASE,
    )
    matched = pattern.search(flattened)
    if matched:
        action_text = matched.group(1).strip().strip("; ")
    else:
        fallback = re.search(r"actions\s*:\s*(.*)$", flattened, flags=re.IGNORECASE)
        action_text = fallback.group(1).strip().strip("; ") if fallback else flattened

    if not action_text:
        action_text = flattened

    masked_text = cache_scorer.preprocess_text(action_text, row)
    return masked_text if masked_text else action_text


def select_examples_by_embedding(query_text, gt_embeddings, gt_num_words, gt_texts, word_limit, tokenizer, model, num_examples=10, lambda_penalty=0.1):
    query_embedding = encode_texts([query_text], tokenizer, model, batch_size=1)[0]
    cosine_scores = torch.matmul(gt_embeddings, query_embedding)

    word_limit = max(1, int(word_limit))
    len_diff = torch.tensor([abs(length - word_limit) for length in gt_num_words], dtype=torch.float32)
    penalties = lambda_penalty * (len_diff / word_limit)
    final_scores = cosine_scores - penalties

    topk = min(num_examples, len(gt_texts))
    top_indices = torch.topk(final_scores, k=topk).indices.tolist()
    return [gt_texts[index] for index in top_indices]


def parse_single_output(text_summary):
    text_summary = str(text_summary).replace('{"summarized_AD": "', "").replace('"}', "").strip()
    if text_summary and text_summary[-1] != ".":
        text_summary = text_summary + "."
    return text_summary




def extract_ranked_candidates(candidates, candidate_scores, default_best_idx=0):
    indexed = []
    for idx, caption in enumerate(candidates):
        score = candidate_scores[idx] if idx < len(candidate_scores) else None
        indexed.append((idx, caption, score))

    scored = [entry for entry in indexed if entry[2] is not None]
    if scored:
        scored_desc = sorted(scored, key=lambda x: x[2], reverse=True)
        scored_asc = sorted(scored, key=lambda x: x[2])
        best = scored_desc[0]
        second = scored_desc[1] if len(scored_desc) > 1 else None
        worst = scored_asc[0]
    else:
        safe_idx = min(max(default_best_idx, 0), len(indexed) - 1)
        best = indexed[safe_idx]
        second = indexed[1] if len(indexed) > 1 else None
        worst = indexed[-1]

    return best, second, worst

def main(args):
    person_aliases = None
    if args.person_aliases_json:
        with open(args.person_aliases_json, "r", encoding="utf-8") as fp:
            person_aliases = json.load(fp)

    # Initialise the model
    pipeline = initialise_model(args.access_token)
    emb_tokenizer, emb_model = initialise_embedding_model(args.embedding_model)
    cache_scorer = CacheClipScorer(
        cache_root=args.cache_root,
        embedding_priority=args.cache_embedding_priority,
        text_model_name=args.clip_text_model,
        apply_name_masking=args.apply_name_masking,
        person_placeholder=args.person_placeholder,
        person_aliases=person_aliases,
    )

    # Read predicted output from Stage I
    pred_df = pd.read_csv(args.pred_path)

    # Dataset-specific information
    if args.dataset in ["cmdad"]:
        gt_df = pd.read_csv(args.exam_path)  # GT ADs in training split
        ad_speed = 0.275

    elif args.dataset in ["tvad"]:
        gt_df = pd.read_csv(args.exam_path)  # GT ADs in training split
        ad_speed = 0.2695

    elif args.dataset in ["madeval"]:
        gt_df = pd.read_csv(args.exam_path)  # GT ADs in training split
        ad_speed = 0.5102
    else:
        print("Check the dataset name")
        sys.exit()

    # Extract GT AD list (w & wo character information)
    all_gts = gt_df["text_gt"].tolist()
    all_gts_wo_char = gt_df["text_gt_wo_char"].tolist()
    all_gts_num_words = [len(str(e).strip().split(" ")) for e in all_gts_wo_char]
    gt_embeddings = encode_texts(all_gts_wo_char, emb_tokenizer, emb_model)

    unigram_prior = None
    if args.enable_prior_candidate_selection:
        unigram_prior = build_unigram_prior(all_gts_wo_char, alpha=args.prior_alpha)

    text_gen_list = []
    text_gt_list = []
    start_sec_list = []
    end_sec_list = []
    imdbid_list = []
    selected_caption_list = []
    selected_clipscore_list = []
    next_best_caption_list = []
    next_best_clipscore_list = []
    lowest_caption_list = []
    lowest_clipscore_list = []
    selected_cache_path_list = []
    selected_cache_hit_list = []

    generation_batch_size = max(1, int(args.generation_batch_size))
    pred_records = pred_df.to_dict("records")

    for chunk_start in tqdm(
        range(0, len(pred_records), generation_batch_size),
        total=(len(pred_records) + generation_batch_size - 1) // generation_batch_size,
        desc="Stage 2 inference",
        unit="batch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%",
    ):
        batch_records = pred_records[chunk_start:chunk_start + generation_batch_size]

        batch_row_dicts = []
        batch_user_prompts = []

        for row in batch_records:
            # Estimate the number of words based on training split statistics
            duration = round(row['end'] - row['start'], 2)
            rough_num_words = round(duration / ad_speed)

            text_pred = str(row['text_gen'])
            retrieval_query = build_retrieval_query(text_pred, cache_scorer, row)
            sampled_examples = select_examples_by_embedding(
                query_text=retrieval_query,
                gt_embeddings=gt_embeddings,
                gt_num_words=all_gts_num_words,
                gt_texts=all_gts_wo_char,
                word_limit=rough_num_words,
                tokenizer=emb_tokenizer,
                model=emb_model,
                num_examples=args.num_examples,
                lambda_penalty=args.lambda_penalty,
            )

            # Formulate the user prompt
            user_prompt = get_user_prompt(
                mode=args.mode,
                prompt_idx=args.prompt_idx,
                text_pred=text_pred,
                word_limit=int(duration / ad_speed) + 1,
                examples=sampled_examples,
            )

            batch_row_dicts.append(row)
            batch_user_prompts.append(user_prompt)

        # Generate candidate ADs in batch
        batch_candidates = [[] for _ in range(len(batch_user_prompts))]
        if args.mode == "single":
            for _ in range(args.num_candidates):
                raw_summaries = summary_batch(
                    pipeline,
                    batch_user_prompts,
                    args.dataset,
                    generation_batch_size=args.generation_batch_size,
                    generation_num_workers=args.generation_num_workers,
                    generation_temperature=args.generation_temperature,
                    generation_top_p=args.generation_top_p,
                    generation_top_k=args.generation_top_k,
                    generation_repetition_penalty=args.generation_repetition_penalty,
                )
                for idx, raw_summary in enumerate(raw_summaries):
                    parsed_summary = parse_single_output(raw_summary)
                    if parsed_summary:
                        batch_candidates[idx].append(parsed_summary)
        else:
            # assistant mode returns 5 candidates in one JSON response
            raw_summaries = summary_batch(
                pipeline,
                batch_user_prompts,
                args.dataset,
                generation_batch_size=args.generation_batch_size,
                generation_num_workers=args.generation_num_workers,
                generation_temperature=args.generation_temperature,
                generation_top_p=args.generation_top_p,
                generation_top_k=args.generation_top_k,
                generation_repetition_penalty=args.generation_repetition_penalty,
            )
            for idx, raw_summary in enumerate(raw_summaries):
                candidates = []
                for ad_idx in range(1, 6):
                    parsed_summary = raw_summary.split(f'"summarized_AD_{ad_idx}":')[-1].split("\n")[0].split(",")[0].replace('"', "").replace('{', "").replace('}', "").strip()
                    if parsed_summary and parsed_summary[-1] != ".":
                        parsed_summary = parsed_summary + "."
                    if parsed_summary:
                        candidates.append(parsed_summary)
                batch_candidates[idx] = candidates

        for row_offset, (row_dict, candidates) in enumerate(zip(batch_row_dicts, batch_candidates)):
            if not candidates:
                candidates = [""]

            # Cache-aware CLIP scoring
            candidate_scores, best_idx, cache_path = cache_scorer.score_candidates(row_dict, candidates)

            if args.print_candidate_scores:
                sample_index = chunk_start + row_offset
                print(
                    f"[score-debug] idx={sample_index} cache={'hit' if cache_path is not None else 'miss'} "
                    f"cache_path={str(cache_path) if cache_path is not None else ''} scores={candidate_scores}"
                )

            if best_idx is None:
                best_idx = 0

            selected_by = "clip"
            prior_details = []
            if args.enable_prior_candidate_selection and unigram_prior is not None:
                best_idx, prior_details = select_candidate_with_prior(
                    candidates,
                    unigram_prior,
                    score_mode=args.prior_selection_mode,
                )
                selected_by = f"prior_{args.prior_selection_mode}"

            output_ads = candidates[best_idx]
            selected_score = candidate_scores[best_idx] if best_idx < len(candidate_scores) else None

            # Real-time generation output for monitoring.
            sample_index = chunk_start + row_offset
            print(f"[generation] idx={sample_index} selected_by={selected_by} selected_score={selected_score} selected={output_ads}")
            if args.print_prior_scores and prior_details:
                print(f"[prior-debug] idx={sample_index} mode={args.prior_selection_mode} details={prior_details}")

            best_candidate, second_candidate, worst_candidate = extract_ranked_candidates(candidates, candidate_scores, default_best_idx=best_idx)

            text_gen_list.append(output_ads)
            text_gt_list.append(row_dict['text_gt'])
            start_sec_list.append(row_dict['start'])
            end_sec_list.append(row_dict['end'])
            imdbid_list.append(row_dict['imdbid'])
            selected_caption_list.append(best_candidate[1])
            selected_clipscore_list.append(best_candidate[2])
            next_best_caption_list.append(second_candidate[1] if second_candidate is not None else "")
            next_best_clipscore_list.append(second_candidate[2] if second_candidate is not None else None)
            lowest_caption_list.append(worst_candidate[1])
            lowest_clipscore_list.append(worst_candidate[2])
            selected_cache_path_list.append(str(cache_path) if cache_path is not None else "")
            selected_cache_hit_list.append(bool(cache_path is not None))

    output_df = pd.DataFrame.from_records({
        'imdbid': imdbid_list,
        'start': start_sec_list,
        'end': end_sec_list,
        'text_gt': text_gt_list,
        'text_gen': text_gen_list,
        'selected_caption': selected_caption_list,
        'selected_clipscore': selected_clipscore_list,
        'next_best_caption': next_best_caption_list,
        'next_best_clipscore': next_best_clipscore_list,
        'lowest_caption': lowest_caption_list,
        'lowest_clipscore': lowest_clipscore_list,
        'selected_cache_embedding_path': selected_cache_path_list,
        'selected_cache_hit': selected_cache_hit_list,
    })
    save_path = os.path.join(args.save_dir, args.dataset + "_ads", f"stage2_llama3_{args.mode}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default=None, type=str, help='input directory')
    parser.add_argument('--exam_path', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str, help='output directory')
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--mode', default="single", type=str)
    parser.add_argument('--access_token', default=None, type=str, help='HuggingFace token to access llama3')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--prompt_idx', default=0, type=int, help='optional, use to indicate you own prompt')
    parser.add_argument('--num_examples', default=10, type=int, help='number of GT ADs')
    parser.add_argument('--embedding_model', default='sentence-transformers/all-MiniLM-L6-v2', type=str, help='embedding model for retrieval')
    parser.add_argument('--lambda_penalty', default=0.1, type=float, help='length penalty coefficient for retrieval score')
    parser.add_argument('--num_candidates', default=1, type=int, help='number of candidate captions per clip')
    parser.add_argument('--cache_root', default='cache/output', type=str, help='cache root containing output/video or output/frame embeddings')
    parser.add_argument('--cache_embedding_priority', default='video,frame', type=str, help='comma-separated embedding priority for cache lookup')
    parser.add_argument('--clip_text_model', default='openai/clip-vit-base-patch32', type=str, help='CLIP text encoder used for candidate scoring')
    parser.add_argument('--apply_name_masking', action='store_true', help='apply name masking/normalisation before CLIP text embedding dot-product scoring')
    parser.add_argument('--person_placeholder', default='[PERSON]', type=str, help='placeholder token used when masking person names')
    parser.add_argument('--person_aliases_json', default=None, type=str, help='optional JSON path mapping canonical names to alias lists for name normalisation')
    parser.add_argument('--generation_batch_size', default=8, type=int, help='batch size for LLM text generation pipeline')
    parser.add_argument('--generation_num_workers', default=4, type=int, help='number of workers used by generation pipeline dataloader')
    parser.add_argument('--generation_temperature', default=0.85, type=float, help='sampling temperature for generation')
    parser.add_argument('--generation_top_p', default=0.93, type=float, help='nucleus sampling top-p for generation')
    parser.add_argument('--generation_top_k', default=50, type=int, help='top-k sampling for generation')
    parser.add_argument('--generation_repetition_penalty', default=1.1, type=float, help='repetition penalty for generation')
    parser.add_argument('--print_candidate_scores', action='store_true', help='print per-sample candidate CLIP scores with sample index and cache lookup status')
    parser.add_argument('--enable_prior_candidate_selection', action='store_true', help='enable candidate selection using LM prior + consensus + content length instead of CLIP best score')
    parser.add_argument('--prior_selection_mode', default='quality', choices=['quality', 'safety'], help='final scoring mode for prior-based candidate selection')
    parser.add_argument('--prior_alpha', default=1e-3, type=float, help='add-alpha smoothing value for unigram LM prior')
    parser.add_argument('--print_prior_scores', action='store_true', help='print per-candidate prior score details when prior-based selection is enabled')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
