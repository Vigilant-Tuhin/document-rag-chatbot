# A real tokenizer (e.g. HuggingFace AutoTokenizer for the Llama model) would be
# more accurate, but pulling gated HF tokenizers requires an auth token, which
# is awkward to provision inside Docker. This heuristic (~4 chars/token) is good
# enough for budgeting a prompt, not for anything that needs exact counts.


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
