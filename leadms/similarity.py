"""String-similarity primitives and a TF-IDF character n-gram index.

All of this is deliberately dependency-free. The n-gram index is the piece
that keeps deduplication tractable: it produces a short list of plausible
neighbours per record so the expensive per-pair scoring never has to run
across all ~2.1M record pairs.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .normalize import COMPANY_SUFFIX_TOKENS, company_tokens

# Suffix tokens carry little identity information because the fixture swaps
# them between duplicates; they still count, just much less.
_SUFFIX_WEIGHT = 0.15


def jaro(a, b):
    """Jaro similarity in [0, 1]."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    match_window = max(len(a), len(b)) // 2 - 1
    if match_window < 0:
        match_window = 0

    a_matched = [False] * len(a)
    b_matched = [False] * len(b)
    matches = 0

    for i, ch in enumerate(a):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len(b))
        for j in range(start, end):
            if b_matched[j] or b[j] != ch:
                continue
            a_matched[i] = True
            b_matched[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions among the matched characters.
    transpositions = 0
    k = 0
    for i, ch in enumerate(a):
        if not a_matched[i]:
            continue
        while not b_matched[k]:
            k += 1
        if ch != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (
        matches / len(a) + matches / len(b) + (matches - transpositions) / matches
    ) / 3.0


def jaro_winkler(a, b, prefix_weight=0.1, max_prefix=4):
    """Jaro-Winkler: Jaro with a bonus for a shared prefix.

    Preferred over a generic edit-distance ratio for personal names, where
    errors cluster at the end of the string (typos, truncations, initials)
    and the first few characters are usually stable.
    """
    score = jaro(a or "", b or "")
    if score <= 0.0:
        return 0.0
    prefix = 0
    for x, y in zip(a or "", b or ""):
        if x != y:
            break
        prefix += 1
        if prefix == max_prefix:
            break
    return score + prefix * prefix_weight * (1.0 - score)


def build_company_token_weights(names, floor=0.1):
    """Derive per-token weights from the corpus by inverse document frequency.

    A hand-curated suffix list cannot work here, because the same token is
    core in one company name and a throwaway suffix in another: "Analytics" is
    the identity of "Huang Analytics Consulting" but mere decoration in
    "Lotus Finance Analytics". IDF settles it from the data - "Co", "Ltd" and
    "Trading" appear in hundreds of names and score near the floor, while
    "Huang" or "Lumen" appear in a handful and score near 1.0.

    Returns a ``{token: weight}`` mapping scaled into ``[floor, 1.0]``.
    """
    document_freq = Counter()
    total = 0
    for name in names:
        tokens = set(company_tokens(name))
        if not tokens:
            continue
        total += 1
        document_freq.update(tokens)
    if not total:
        return {}

    idf = {
        token: math.log(total / df) for token, df in document_freq.items() if df
    }
    max_idf = max(idf.values()) if idf else 1.0
    if max_idf <= 0:
        return {token: 1.0 for token in idf}
    return {
        token: floor + (1.0 - floor) * (value / max_idf)
        for token, value in idf.items()
    }


def company_similarity(a, b, token_weights=None):
    """Weighted token overlap between two company names, in [0, 1].

    With ``token_weights`` from :func:`build_company_token_weights`, common
    legal/descriptor tokens contribute little and distinguishing tokens carry
    the decision. Without a corpus, we fall back to the static suffix list.

    A plain Jaccard scores "Huang Analytics Consulting" against
    "Huang Analytics Robotics" at only 0.5; this scores it ~0.9.
    """
    tokens_a, tokens_b = set(company_tokens(a)), set(company_tokens(b))
    if not tokens_a or not tokens_b:
        return 0.0

    if token_weights:
        def weight(token):
            # Unseen tokens are assumed distinctive - a token absent from the
            # corpus is by definition rare.
            return token_weights.get(token, 1.0)
    else:
        def weight(token):
            return _SUFFIX_WEIGHT if token in COMPANY_SUFFIX_TOKENS else 1.0

    intersection = sum(weight(t) for t in tokens_a & tokens_b)
    union = sum(weight(t) for t in tokens_a | tokens_b)
    return intersection / union if union else 0.0


def char_ngrams(text, n=3):
    """Character n-grams over a whitespace-padded, squashed string."""
    cleaned = re.sub(r"\s+", " ", (text or "").lower().strip())
    if not cleaned:
        return Counter()
    padded = " " + cleaned + " "
    if len(padded) < n:
        return Counter([padded])
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


class TfidfNgramIndex:
    """Inverted TF-IDF index over character n-grams with top-K retrieval.

    Why this and not an embedding model: at ~2k records the signal that
    separates duplicates here is surface-level (transposed characters, dropped
    initials, reformatted punctuation), which character n-grams capture
    directly. It also needs no API call, no model download, and no vector
    store, and it stays exact and reproducible in tests. The class is small
    enough to swap for an embedding + ANN backend behind the same
    ``neighbours()`` interface if the corpus ever outgrows it.
    """

    def __init__(self, n=3, max_df_ratio=0.25):
        self.n = n
        # Grams appearing in more than this fraction of documents are dropped
        # from the postings lists: they are near-stopwords for this corpus,
        # contribute almost nothing to cosine similarity, and would otherwise
        # dominate the cost of every query.
        self.max_df_ratio = max_df_ratio
        self.keys = []
        self._vectors = []
        self._postings = defaultdict(list)

    def build(self, documents):
        """``documents`` is an iterable of ``(key, text)`` pairs."""
        self.keys = []
        raw_counts = []
        for key, text in documents:
            self.keys.append(key)
            raw_counts.append(char_ngrams(text, self.n))

        total = len(self.keys)
        if total == 0:
            return self

        document_freq = Counter()
        for counts in raw_counts:
            document_freq.update(counts.keys())

        df_cutoff = max(1, int(self.max_df_ratio * total))
        idf = {
            gram: math.log((total + 1) / (df + 1)) + 1.0
            for gram, df in document_freq.items()
        }

        self._vectors = []
        self._postings = defaultdict(list)
        for index, counts in enumerate(raw_counts):
            vector = {}
            for gram, tf in counts.items():
                vector[gram] = (1.0 + math.log(tf)) * idf[gram]
            norm = math.sqrt(sum(w * w for w in vector.values())) or 1.0
            vector = {gram: w / norm for gram, w in vector.items()}
            self._vectors.append(vector)
            for gram, weight in vector.items():
                if document_freq[gram] <= df_cutoff:
                    self._postings[gram].append((index, weight))
        return self

    def neighbours(self, index, top_k=8, min_score=0.0):
        """Top-K cosine neighbours of document ``index``, excluding itself."""
        scores = defaultdict(float)
        for gram, weight in self._vectors[index].items():
            for other, other_weight in self._postings.get(gram, ()):
                if other != index:
                    scores[other] += weight * other_weight

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (self.keys[other], score)
            for other, score in ranked[:top_k]
            if score >= min_score
        ]

    def all_neighbour_pairs(self, top_k=8, min_score=0.0):
        """Yield ``(key_a, key_b, score)`` for every document's top-K list."""
        for index in range(len(self.keys)):
            key_a = self.keys[index]
            for key_b, score in self.neighbours(index, top_k, min_score):
                yield (key_a, key_b, score)
