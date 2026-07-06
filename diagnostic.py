"""
Diagnostic script — test each component independently.
Run this to find out where things are breaking.
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("HIVER AI CHALLENGE — DIAGNOSTIC")
print("=" * 60)

# Test 1: .env file
print("\n[1/5] Checking .env file...")
env_found = False
for p in [Path(".env"), Path.cwd() / ".env", Path(__file__).parent / ".env"]:
    if p.exists():
        print(f"  ✅ Found .env at: {p}")
        with open(p) as f:
            content = f.read()
            if "GROQ_API_KEY" in content and "your_groq" not in content:
                print(f"  ✅ GROQ_API_KEY appears to be set")
                env_found = True
            else:
                print(f"  ❌ GROQ_API_KEY is missing or still has placeholder value")
        break

if not env_found:
    print("  ❌ .env file not found in expected locations")
    print("  Fix: Create .env in the same folder as this script")
    print("  Content: GROQ_API_KEY=gsk_your_actual_key")

# Test 2: Python packages
print("\n[2/5] Checking Python packages...")
packages = ["groq", "sentence_transformers", "numpy", "dotenv"]
for pkg in packages:
    try:
        if pkg == "dotenv":
            __import__("dotenv")
        else:
            __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} — run: pip install -r requirements.txt")

# Test 3: Dataset
print("\n[3/5] Checking dataset.json...")
try:
    import json
    with open("dataset.json", "r") as f:
        data = json.load(f)
    print(f"  ✅ dataset.json loaded: {len(data)} examples")
except Exception as e:
    print(f"  ❌ dataset.json error: {e}")

# Test 4: Groq API key
print("\n[4/5] Checking Groq API key...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if key and "your_groq" not in key:
        print(f"  ✅ Key loaded (starts with: {key[:10]}...)")

        # Test actual API call
        print("  🔄 Testing Groq API call (this may take 5-10 seconds)...")
        from groq import Groq
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say 'Groq API is working'"}],
            max_tokens=20
        )
        print(f"  ✅ Groq API works! Response: {response.choices[0].message.content.strip()}")
    else:
        print(f"  ❌ Key not loaded or still has placeholder")
except Exception as e:
    print(f"  ❌ Groq API error: {e}")

# Test 5: Embedding model
print("\n[5/5] Checking embedding model...")
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    print("  🔄 Loading all-MiniLM-L6-v2 (this takes 30-60s on first run)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode("Hello world")
    print(f"  ✅ Embedding model loaded! Vector shape: {emb.shape}")
except Exception as e:
    print(f"  ❌ Embedding model error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)