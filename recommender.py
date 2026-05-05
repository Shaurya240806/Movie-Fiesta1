"""
Content-Based Recommendation Engine
Uses TF-IDF + Cosine Similarity on genre, keywords, cast, director, mood_tags

Optimised:
  - Sparse matrix dot-product instead of full cosine_similarity materialisation
  - Normalised rows cached once at build time (O(n) lookup per query)
  - Mood search uses a single vectorised pass
  - All model attribute access is null-safe (getattr with default '')
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np


class ContentBasedRecommender:
    def __init__(self, movies):
        self.movies    = list(movies)
        self.movie_map = {m.id: i for i, m in enumerate(self.movies)}
        self._build_matrix()

    def _build_matrix(self):
        """Build normalised TF-IDF matrix — rows pre-normalised for fast dot-product."""
        corpus = []
        for m in self.movies:
            parts = [
                getattr(m, 'genre',     '') or '',
                getattr(m, 'genre',     '') or '',   # weight genre ×2
                getattr(m, 'keywords',  '') or '',
                getattr(m, 'cast',      '') or '',
                getattr(m, 'director',  '') or '',
                getattr(m, 'mood_tags', '') or '',
            ]
            corpus.append(' '.join(filter(None, parts)).lower())

        if len(corpus) < 2:
            self._tfidf_norm = None
            return

        try:
            vectorizer       = TfidfVectorizer(
                stop_words='english',
                min_df=1,
                max_features=5000,
                sublinear_tf=True,   # log(1+tf) — improves quality
            )
            tfidf            = vectorizer.fit_transform(corpus)  # sparse (n, vocab)
            # Pre-normalise rows once → similarity = dot product (no division at query time)
            self._tfidf_norm = normalize(tfidf, norm='l2')
        except Exception as e:
            print(f"⚠️  Recommender build error: {e}")
            self._tfidf_norm = None

    def get_similar(self, movie_id: int, top_n: int = 8) -> list:
        """Return top_n movies most similar to movie_id. O(n·vocab) sparse dot product."""
        if self._tfidf_norm is None or movie_id not in self.movie_map:
            return self._fallback(top_n)

        idx    = self.movie_map[movie_id]
        # Sparse row × full matrix → dense 1-D score array; no full n×n matrix needed
        scores = (self._tfidf_norm[idx] @ self._tfidf_norm.T).toarray().ravel()
        scores[idx] = -1   # exclude self
        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self.movies[i] for i in top_indices]

    def get_by_mood(self, mood: str, top_n: int = 12) -> list:
        """Return movies matching a mood string, sorted by rating."""
        mood_lower = mood.lower()

        matched = [
            m for m in self.movies
            if mood_lower in (getattr(m, 'mood_tags', '') or '').lower()
        ]

        if not matched:
            matched = [
                m for m in self.movies
                if mood_lower in (getattr(m, 'genre', '') or '').lower()
            ]

        matched.sort(key=lambda m: getattr(m, 'rating', 0) or 0, reverse=True)
        return matched[:top_n]

    def get_top_rated(self, top_n: int = 20) -> list:
        """Return top-rated movies (useful as a fallback)."""
        return sorted(
            self.movies,
            key=lambda m: getattr(m, 'rating', 0) or 0,
            reverse=True
        )[:top_n]

    def _fallback(self, top_n: int) -> list:
        return self.get_top_rated(top_n)
