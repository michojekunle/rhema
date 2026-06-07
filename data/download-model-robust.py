import time
import sys
import os

# Set default HF_ENDPOINT to hf-mirror.com if not already configured, 
# ensuring GHA and other environments default to the reliable mirror.
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen3-Embedding-0.6B"
FILES = [
    '.gitattributes',
    '1_Pooling/config.json',
    'README.md',
    'config.json',
    'config_sentence_transformers.json',
    'generation_config.json',
    'merges.txt',
    'model.safetensors',
    'modules.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.json'
]

def main():
    # Force stdout/stderr to use UTF-8 if they have encoding limitations
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print(f"Starting robust download of {REPO_ID} (12 files)...")
    
    # Disable XET protocol which stalls on large files
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_DISABLE"] = "1"

    # Set up endpoints list for fallback/rotation
    initial_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    endpoints = [initial_endpoint, "https://huggingface.co", "https://hf-mirror.com"]
    # De-duplicate while preserving order
    seen = set()
    endpoints = [x for x in endpoints if not (x in seen or seen.add(x))]
    
    for filename in FILES:
        print(f"\n--> Downloading {filename}...")
        success = False
        
        for attempt in range(1, 16):
            # Rotate endpoints to bypass server blockages or timeouts
            current_endpoint = endpoints[(attempt - 1) % len(endpoints)]
            os.environ["HF_ENDPOINT"] = current_endpoint
            try:
                # hf_hub_download is resumable by default and uses standard cache
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    repo_type="model"
                )
                success = True
                print(f"[OK] Completed: {filename} (using {current_endpoint})")
                break
            except Exception as e:
                print(f"[WARN] Attempt {attempt}/15 failed for {filename} using {current_endpoint}: {e}")
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
                
        if not success:
            print(f"[ERROR] Failed to download {filename} after multiple attempts.")
            sys.exit(1)
            
    print("\n[SUCCESS] All 12 files successfully downloaded!")

if __name__ == "__main__":
    main()

