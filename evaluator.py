"""
Hiver AI Challenge — Email Reply Evaluator
Measures accuracy of AI-generated replies using:
1. BERTScore (Semantic Similarity via embeddings)
2. Style Matching (action phrases + opening similarity)
3. LLM Judge (Groq/Llama evaluates tone, actionability, completeness, empathy)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# ── BULLETPROOF .env LOADER ──────────────────────────────────────────
def load_env_robust():
    paths_to_try = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    for p in paths_to_try:
        if p.exists():
            load_dotenv(dotenv_path=p)
            return str(p)
    load_dotenv()
    return None

load_env_robust()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Fallback: paste key here if .env fails
# GROQ_API_KEY = "gsk_your_actual_key_here"

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print("ERROR: GROQ_API_KEY not found. Set it in .env or hardcode above.")
    sys.exit(1)

client = Groq(api_key=GROQ_API_KEY)

# ── LAZY EMBEDDING MODEL ─────────────────────────────────────────────
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("[Evaluator] Loading embedding model (one-time setup)...")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        from sentence_transformers import SentenceTransformer, util
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[Evaluator] Embedding model ready.")
    return _embedding_model


def bertscore_similarity(generated: str, expected: str) -> float:
    """Compute semantic similarity using sentence-transformer embeddings."""
    model = get_embedding_model()
    from sentence_transformers import util
    emb1 = model.encode(generated, convert_to_tensor=True)
    emb2 = model.encode(expected, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return max(0.0, min(1.0, similarity))


def style_match_score(generated: str, expected: str) -> float:
    """
    Compute style matching score based on:
    - Opening phrase similarity (greeting patterns)
    - Closing phrase similarity (sign-off patterns)
    - Action phrase overlap ("I've processed", "I've attached", etc.)
    """
    gen_lower = generated.lower()
    exp_lower = expected.lower()

    # Extract opening phrases (first 30 chars)
    gen_open = gen_lower[:30]
    exp_open = exp_lower[:30]

    # Check for common action phrases
    action_phrases = [
        "i've processed", "i've attached", "i've updated", "i've created",
        "i've sent", "i've checked", "i've identified", "i've corrected",
        "i've initiated", "i've enabled", "i've scheduled", "i've added",
        "please let me know", "feel free to", "don't worry",
        "i apologize", "i'm sorry", "thank you for"
    ]

    gen_actions = sum(1 for phrase in action_phrases if phrase in gen_lower)
    exp_actions = sum(1 for phrase in action_phrases if phrase in exp_lower)

    # Normalize action match
    if exp_actions > 0:
        action_score = min(gen_actions / max(exp_actions, 1), 1.0)
    else:
        action_score = 0.5

    # Opening similarity (simple word overlap)
    gen_words = set(gen_open.split())
    exp_words = set(exp_open.split())
    if len(exp_words) > 0:
        opening_score = len(gen_words & exp_words) / len(exp_words)
    else:
        opening_score = 0.5

    # Combined style score (0-1)
    style_score = 0.6 * action_score + 0.4 * opening_score
    return min(1.0, max(0.0, style_score))


def llm_judge(generated: str, expected: str, customer_email: str, model: str = "llama-3.3-70b-versatile") -> dict:
    """Use an LLM as a judge to evaluate the generated reply."""
    judge_prompt = f"""You are an expert customer support quality evaluator.

Evaluate the AI-generated reply compared to the expected (human-written) reply.
Consider the original customer email for context.

**Customer Email:**
{customer_email}

**Expected Reply (human gold standard):**
{expected}

**AI-Generated Reply:**
{generated}

Score the AI reply on these dimensions (0-100 each). Be strict but fair:

1. **Tone**: Is the tone appropriate? Warm, professional, not robotic or overly casual?
2. **Actionability**: Does it provide clear next steps, timelines, or concrete actions?
3. **Completeness**: Does it address all parts of the customer's email? Nothing missed?
4. **Empathy**: Does it show understanding of the customer's situation and frustration?
5. **Overall**: How good is this reply overall as a customer support response?

Respond ONLY in this exact JSON format:
{{
  "tone": <0-100>,
  "actionability": <0-100>,
  "completeness": <0-100>,
  "empathy": <0-100>,
  "overall": <0-100>,
  "reasoning": "<brief explanation of the overall score>"
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict but fair customer support quality evaluator. Always respond in valid JSON."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.3,
        max_tokens=512,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return result


def evaluate_single(generated: str, expected: str, customer_email: str) -> dict:
    """Run full evaluation on one generated reply."""
    semantic = bertscore_similarity(generated, expected)
    style = style_match_score(generated, expected)
    llm_scores = llm_judge(generated, expected, customer_email)

    # Combined semantic score: 70% BERTScore + 30% style match
    combined_semantic = 0.7 * semantic + 0.3 * style

    # Final score: 25% semantic + 75% LLM Judge overall
    final_score = 0.25 * (combined_semantic * 100) + 0.75 * llm_scores["overall"]

    return {
        "semantic_similarity": round(semantic * 100, 2),
        "style_match": round(style * 100, 2),
        "combined_semantic": round(combined_semantic * 100, 2),
        "llm_judge": {
            "tone": llm_scores["tone"],
            "actionability": llm_scores["actionability"],
            "completeness": llm_scores["completeness"],
            "empathy": llm_scores["empathy"],
            "overall": llm_scores["overall"],
            "reasoning": llm_scores["reasoning"]
        },
        "final_score": round(final_score, 2)
    }


def evaluate_batch(results: list[dict]) -> dict:
    """Aggregate scores across a batch of evaluations."""
    semantic_scores = [r["semantic_similarity"] for r in results]
    style_scores = [r["style_match"] for r in results]
    combined_semantic_scores = [r["combined_semantic"] for r in results]
    final_scores = [r["final_score"] for r in results]

    tone_scores = [r["llm_judge"]["tone"] for r in results]
    action_scores = [r["llm_judge"]["actionability"] for r in results]
    complete_scores = [r["llm_judge"]["completeness"] for r in results]
    empathy_scores = [r["llm_judge"]["empathy"] for r in results]
    overall_scores = [r["llm_judge"]["overall"] for r in results]

    return {
        "count": len(results),
        "semantic_similarity": {
            "mean": round(np.mean(semantic_scores), 2),
            "std": round(np.std(semantic_scores), 2),
            "min": round(min(semantic_scores), 2),
            "max": round(max(semantic_scores), 2)
        },
        "style_match": {
            "mean": round(np.mean(style_scores), 2),
            "std": round(np.std(style_scores), 2)
        },
        "combined_semantic": {
            "mean": round(np.mean(combined_semantic_scores), 2),
            "std": round(np.std(combined_semantic_scores), 2)
        },
        "llm_judge": {
            "tone": {"mean": round(np.mean(tone_scores), 2), "std": round(np.std(tone_scores), 2)},
            "actionability": {"mean": round(np.mean(action_scores), 2), "std": round(np.std(action_scores), 2)},
            "completeness": {"mean": round(np.mean(complete_scores), 2), "std": round(np.std(complete_scores), 2)},
            "empathy": {"mean": round(np.mean(empathy_scores), 2), "std": round(np.std(empathy_scores), 2)},
            "overall": {"mean": round(np.mean(overall_scores), 2), "std": round(np.std(overall_scores), 2)}
        },
        "final_score": {
            "mean": round(np.mean(final_scores), 2),
            "std": round(np.std(final_scores), 2),
            "min": round(min(final_scores), 2),
            "max": round(max(final_scores), 2)
        }
    }