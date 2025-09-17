
# src/client_demo.py
import argparse, requests, json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8080")
    p.add_argument("--prompt", default="To be, or not to be")
    p.add_argument("--max-new-tokens", type=int, default=200)
    args = p.parse_args()
    payload = {"prompt": args.prompt, "max_new_tokens": args.max_new_tokens}
    r = requests.post(f"{args.host}/generate", json=payload, timeout=60)
    print(json.dumps(r.json(), indent=2))
