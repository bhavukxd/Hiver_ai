"""
Hiver AI Challenge — RAG Email Reply Generator
Retrieves similar emails for few-shot learning.
CRITICAL: Prevents hallucination of actions, amounts, dates, and order IDs.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

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

# ── RAG SYSTEM ───────────────────────────────────────────────────────
class EmailRAG:
    """Retrieval-Augmented Generation for email replies."""

    def __init__(self, dataset_path: str = "dataset.json", model_name: str = "all-MiniLM-L6-v2"):
        print("[RAG] Loading dataset...")

        possible_paths = [
            Path(dataset_path),
            Path(__file__).parent / dataset_path,
            Path.cwd() / dataset_path,
        ]

        found_path = None
        for p in possible_paths:
            if p.exists():
                found_path = p
                break

        if not found_path:
            print(f"[RAG] ERROR: Could not find {dataset_path}")
            sys.exit(1)

        print(f"[RAG] Found dataset at: {found_path}")

        with open(found_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        print(f"[RAG] Loaded {len(self.dataset)} examples. Now loading embedding model...")
        print("[RAG] (This downloads ~90MB on first run, takes 30-60 seconds...)")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        self.encoder = SentenceTransformer(model_name)

        # Embed emails for retrieval
        self.email_texts = [item["email"] for item in self.dataset]
        self.email_embeddings = self.encoder.encode(self.email_texts, convert_to_tensor=True)

        print(f"[RAG] Ready! {len(self.dataset)} examples indexed.")

    def retrieve(self, query_email: str, top_k: int = 3) -> List[dict]:
        """Find the top_k most similar emails from the dataset."""
        query_emb = self.encoder.encode(query_email, convert_to_tensor=True)
        similarities = util.cos_sim(query_emb, self.email_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:top_k].tolist()
        return [self.dataset[i] for i in top_indices]

    def extract_entities(self, email_text: str) -> dict:
        """Extract specific entities from the customer email."""
        entities = {}

        # Extract order IDs
        order_match = re.search(r'order\s*(?:id)?[:\s#]*([A-Z0-9\-]+)', email_text, re.IGNORECASE)
        if order_match:
            entities['order_id'] = order_match.group(1)

        # Extract amounts
        amount_matches = re.findall(r'\$([\d,]+\.?\d*)', email_text)
        if amount_matches:
            entities['amounts'] = amount_matches

        # Extract dates
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, email_text, re.IGNORECASE)
            if dates:
                entities['dates'] = dates
                break

        # Extract email addresses
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', email_text)
        if emails:
            entities['emails'] = emails

        # Extract card endings
        card_match = re.search(r'(?:card|ending\s+in)\s*(\*{2,4}\d{2,4})', email_text, re.IGNORECASE)
        if card_match:
            entities['card'] = card_match.group(1)

        return entities

    def build_few_shot_prompt(self, query_email: str, examples: List[dict]) -> str:
        """Build a prompt with strict anti-hallucination rules."""

        # Extract entities from customer email
        entities = self.extract_entities(query_email)
        entity_info = ""
        if entities:
            entity_info = "\nSPECIFIC DETAILS FROM CUSTOMER EMAIL (USE ONLY THESE):\n"
            for key, value in entities.items():
                if isinstance(value, list):
                    entity_info += f"- {key}: {', '.join(value)}\n"
                else:
                    entity_info += f"- {key}: {value}\n"
        else:
            entity_info = "\nNO SPECIFIC DETAILS PROVIDED IN CUSTOMER EMAIL.\n"

        prompt_parts = [
            "You are a senior customer support agent at Hiver, a shared inbox platform for teams.",
            "Your job is to write helpful, empathetic email replies to customer inquiries.",
            "\n=== CRITICAL RULES — VIOLATING THESE WILL CAUSE HARM ===",
            "1. NEVER claim you have 'processed', 'completed', 'refunded', 'cancelled', or 'fixed' anything unless the customer email explicitly states it was already done.",
            "2. NEVER state specific dollar amounts, dates, timelines, or order IDs unless they appear in the customer's email.",
            "3. If the customer asks for a refund/cancellation/fix, say you will 'look into it', 'forward to the team', or 'process it shortly' — NOT that it's already done.",
            "4. If you don't know a detail, use vague but honest language: 'we'll verify', 'our team will review', 'we'll get back to you shortly'.",
            "5. NEVER invent customer names. Use 'Hi there,' if no name is provided.",
            "6. NEVER invent order IDs, amounts, card numbers, or email addresses.",
            "7. It's BETTER to sound slightly less polished than to make a false claim.",
            entity_info,
            "\n=== STYLE RULES (from our best replies) ===",
            "- Start with 'Hi there,' or 'Hi [name],' if a name is CLEARLY provided.",
            "- Acknowledge the customer's concern with empathy.",
            "- Explain what YOU (the agent) will do next, NOT what you already did.",
            "- Provide realistic next steps without overpromising.",
            "- End with an open invitation for follow-up.",
            "\n=== EXAMPLES ==="
        ]

        for i, ex in enumerate(examples, 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Customer: {ex['email']}")
            prompt_parts.append(f"Reply: {ex['expected_reply']}")

        prompt_parts.append("\n=== NOW WRITE THE REPLY ===")
        prompt_parts.append(f"Customer: {query_email}")
        prompt_parts.append("Reply (follow ALL rules above, especially #1-7):")

        return "\n".join(prompt_parts)

    def generate_reply(self, email_text: str, model: str = "llama-3.3-70b-versatile", top_k: int = 3) -> str:
        print(f"[RAG] Retrieving similar examples...")
        examples = self.retrieve(email_text, top_k=top_k)
        print(f"[RAG] Found {len(examples)} similar examples. Extracting entities...")

        prompt = self.build_few_shot_prompt(email_text, examples)

        print(f"[RAG] Generating reply via Groq...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert customer support agent. CRITICAL: You are an AI assistant helping draft replies. You CANNOT actually process refunds, cancel orders, or access real systems. NEVER claim an action is complete. ALWAYS use tentative language like 'I'll look into this' or 'our team will review'. Factual correctness is more important than sounding polished."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512,
            top_p=0.9,
        )

        return response.choices[0].message.content.strip()


# Global RAG instance (lazy-loaded)
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        _rag = EmailRAG("dataset.json")
    return _rag


def generate_reply(email_text: str, model: str = "llama-3.3-70b-versatile") -> str:
    rag = get_rag()
    return rag.generate_reply(email_text, model=model, top_k=3)


def generate_reply_batch(emails: List[str], model: str = "llama-3.3-70b-versatile") -> List[str]:
    return [generate_reply(email, model) for email in emails]


if __name__ == "__main__":
    test_email = "can you give me refund of this food"
    reply = generate_reply(test_email)
    print("\nCustomer:", test_email)
    print("\nAI Reply:", reply)