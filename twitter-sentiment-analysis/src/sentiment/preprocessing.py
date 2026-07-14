"""Tweet normalisation and tokenisation.

Twitter text is noisy (slang, hashtags, elongated words, emoticons), so
tweets are normalised before feature extraction:

- lowercasing
- URLs -> ``<url>``, user mentions -> ``<user>``
- hashtags keep the word, drop the ``#``
- elongated words are shortened (``soooo`` -> ``sooo``)
- emoticons/emoji are kept as standalone tokens (strong sentiment cues)

Tokenisation uses NLTK's ``TweetTokenizer``, which handles emoticons and
Twitter-specific constructs much better than the generic ``word_tokenize``
used in early experiments.
"""

import re

from nltk.tokenize import TweetTokenizer

_URL_RE = re.compile(r"(https?://\S+)|(\bwww\.\S+)")
_MENTION_RE = re.compile(r"^@\w+")

_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)


def clean_tweet(text: str) -> list:
    """Normalise a raw tweet and return a list of tokens."""
    text = text.lower()
    text = text.replace("#", "")
    tokens = _tokenizer.tokenize(text)

    cleaned = []
    for tok in tokens:
        if _URL_RE.match(tok):
            cleaned.append("<url>")
        elif _MENTION_RE.match(tok):
            cleaned.append("<user>")
        else:
            cleaned.append(tok)
    return cleaned


def text_to_clean_str(text: str) -> str:
    """Normalised tweet as a single whitespace-joined string (for TF-IDF)."""
    return " ".join(clean_tweet(text))
