"""Download NLTK corpora required at import/runtime (Heroku has no pre-installed data)."""

_NLTK_PACKAGES: tuple[tuple[str, str], ...] = (
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
)


def ensure_nltk_data() -> None:
    import nltk

    for path, package in _NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
