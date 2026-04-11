import re
import math
from urllib.parse import urlparse, parse_qs

def tokenize_url(url):
    tokens = re.split(r"[./\-_]", url.lower())
    tokens = [t for t in tokens if t]
    return tokens

def entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum([p * math.log2(p) for p in prob])

def extract_features(url: str):
    parsed = urlparse(url if url.startswith("http") else "http://" + url)
    domain = parsed.netloc
    path = parsed.path

    features = {}

    features["url_length"] = len(url)
    features["has_https"] = 1 if parsed.scheme == "https" else 0
    features["dot_count"] = url.count(".")
    features["subdomain_count"] = max(0, domain.count(".") - 1)
    features["has_ip"] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
    features["special_char_count"] = sum(not c.isalnum() for c in url)
    features["digit_count"] = sum(c.isdigit() for c in url)

    features["digit_ratio"] = features["digit_count"] / len(url) if len(url) else 0
    features["param_count"] = len(parse_qs(parsed.query))

    # keyword detection
    brand_keywords = ["paypal", "google", "facebook", "amazon", "bank"]
    phish_keywords = ["login", "verify", "update", "secure", "account"]

    features["brand_keyword"] = int(any(k in url.lower() for k in brand_keywords))
    features["phish_keyword"] = int(any(k in url.lower() for k in phish_keywords))

    features["suspicious_tld"] = int(domain.endswith((
        ".tk",".ml",".ga",".cf",".gq",".xyz",".top",".click",".link"
    )))

    features["url_entropy"] = entropy(url)
    features["domain_entropy"] = entropy(domain)

    features["hyphen_count"] = url.count("-")
    features["path_depth"] = path.count("/")
    features["token_count"] = len(re.split(r'[.\-_/]', url))

    vowels = sum(c in "aeiou" for c in url.lower())
    features["vowel_ratio"] = vowels / len(url) if len(url) else 0

    features["domain_length"] = len(domain)
    features["at_symbol"] = url.count("@")
    features["double_slash"] = url.count("//")

        # ─────────────────────────────────────────
    # NLP / TOKEN FEATURES (Phase 6)
    # ─────────────────────────────────────────

    tokens = tokenize_url(url)

    features["token_count"] = len(tokens)

    # average token length
    if tokens:
        features["avg_token_length"] = sum(len(t) for t in tokens) / len(tokens)
    else:
        features["avg_token_length"] = 0

    # random-looking token ratio (heuristic)
    def is_random(token):
        return any(char.isdigit() for char in token) and len(token) > 5

    random_tokens = [t for t in tokens if is_random(t)]
    features["random_token_ratio"] = len(random_tokens) / len(tokens) if tokens else 0

    return features