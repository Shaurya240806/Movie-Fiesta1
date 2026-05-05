"""
Movie Fiesta — Main Flask Application
Fully integrated with TMDB API, Auth, Watchlist, Ratings & Recommender
Optimised: concurrent fetching via ThreadPoolExecutor + retry logic
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import requests
import time
import certifi
from werkzeug.security import generate_password_hash, check_password_hash
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
app.secret_key = 'moviefiesta_secret_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///moviefiesta.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── TMDB CONFIG ──────────────────────────────────────────────
TMDB_API_KEY  = "4aadc90f1f936271c62b128ecb44f9b4"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE    = "https://image.tmdb.org/t/p/w500"

# Auto-generate mood tags from genres (no extra API call needed)
GENRE_MOOD_MAP = {
    'Action':          'action',
    'Adventure':       'action adventure',
    'Comedy':          'happy funny',
    'Drama':           'sad emotional',
    'Romance':         'romantic happy',
    'Horror':          'thriller scary',
    'Thriller':        'thriller suspense',
    'Mystery':         'thriller mystery',
    'Science Fiction': 'sci-fi futuristic',
    'Animation':       'happy family fun',
    'Fantasy':         'adventure fantasy',
    'Crime':           'thriller crime',
    'Family':          'happy family',
    'War':             'sad action',
    'History':         'inspiring drama',
    'Music':           'happy inspiring',
    'Biography':       'inspiring drama',
    'Documentary':     'inspiring educational',
    'Western':         'action adventure',
}

def mood_from_genres(genre_str: str) -> str:
    moods = set()
    for g in genre_str.split(','):
        for tag in GENRE_MOOD_MAP.get(g.strip(), '').split():
            moods.add(tag)
    return ' '.join(moods)


# ── MODELS ──────────────────────────────────────────────────

class Movie(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    tmdb_id     = db.Column(db.Integer, unique=True, nullable=True)
    title       = db.Column(db.String(200))
    genre       = db.Column(db.String(200))
    year        = db.Column(db.Integer)
    rating      = db.Column(db.Float, default=0.0)
    description = db.Column(db.Text)
    cast        = db.Column(db.String(500), default='')
    director    = db.Column(db.String(200), default='')
    language    = db.Column(db.String(50))
    duration    = db.Column(db.Integer, default=120)
    poster_url  = db.Column(db.String(500), default='')
    keywords    = db.Column(db.String(500), default='')
    mood_tags   = db.Column(db.String(400), default='')
    popularity  = db.Column(db.Float, default=0.0)


class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw: str):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw: str) -> bool:
        return check_password_hash(self.password_hash, pw)


class Watchlist(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    user_id  = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)


class UserRating(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id   = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    rating     = db.Column(db.Integer)
    review     = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ── SERIALIZER ──────────────────────────────────────────────

def movie_to_dict(m: Movie) -> dict:
    return {
        'id':          m.id,
        'title':       m.title or '',
        'genre':       m.genre or '',
        'year':        m.year,
        'rating':      round(m.rating or 0, 1),
        'description': m.description or '',
        'cast':        m.cast or '',
        'director':    m.director or '',
        'language':    m.language or '',
        'duration':    m.duration or 120,
        'poster_url':  m.poster_url or '',
        'mood_tags':   m.mood_tags or '',
        'keywords':    m.keywords or '',
    }


# ── RECOMMENDER (lazy singleton) ────────────────────────────

_recommender = None

def get_recommender():
    global _recommender
    if _recommender is None:
        from recommender import ContentBasedRecommender
        _recommender = ContentBasedRecommender(Movie.query.all())
    return _recommender

def reset_recommender():
    """Call after any DB change that affects recommendations."""
    global _recommender
    _recommender = None


# ── TMDB HELPERS ─────────────────────────────────────────────

# Shared session for connection pooling (reuses TCP connections)
_tmdb_session = requests.Session()
_tmdb_session.verify = certifi.where()

def safe_request(url: str, params: dict, retries: int = 4) -> dict | None:
    """GET with exponential backoff retry. Returns parsed JSON or None."""
    for attempt in range(retries):
        try:
            r = _tmdb_session.get(url, params=params, timeout=10)
            if r.status_code == 429:
                wait = int(r.headers.get('Retry-After', 2))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5 * (2 ** attempt))   # 0.5s, 1s, 2s
            else:
                print(f"  ⚠️  TMDB request failed after {retries} attempts: {e}")
    return None


def get_genre_map() -> dict:
    res = safe_request(f"{TMDB_BASE_URL}/genre/movie/list", {"api_key": TMDB_API_KEY})
    if not res:
        return {}
    return {g["id"]: g["name"] for g in res.get("genres", [])}


def fetch_discover_page(lang_code: str, genre_map: dict, page: int, lang_label: str) -> list:
    """Fetch one page of TMDB discover results — called concurrently."""
    res = safe_request(f"{TMDB_BASE_URL}/discover/movie", {
        "api_key":                TMDB_API_KEY,
        "with_original_language": lang_code,
        "sort_by":                "popularity.desc",
        "vote_count.gte":         50,
        "page":                   page,
    })
    if not res:
        return []

    results = []
    for m in res.get("results", []):
        genre_names = [genre_map.get(gid, '') for gid in m.get("genre_ids", [])]
        genre_str   = ', '.join(filter(None, genre_names)) or 'Unknown'
        year        = None
        if m.get("release_date"):
            try:
                year = int(m["release_date"][:4])
            except ValueError:
                pass

        results.append({
            'tmdb_id':     m.get('id'),
            'title':       m.get('title', '').strip(),
            'genre':       genre_str,
            'year':        year,
            'rating':      m.get('vote_average', 0),
            'description': m.get('overview', ''),
            'language':    lang_label,
            'duration':    120,
            'poster_url':  f"{IMAGE_BASE}{m['poster_path']}" if m.get('poster_path') else '',
            'mood_tags':   mood_from_genres(genre_str),
            'popularity':  m.get('popularity', 0),
            'cast':        '',
            'director':    '',
            'keywords':    '',
        })
    return results


def enrich_movie(m: dict) -> dict:
    """
    Single-movie enrichment using 3 parallel sub-requests:
      /movie/{id}          → runtime
      /movie/{id}/credits  → cast + director
      /movie/{id}/keywords → keywords
    All fired concurrently so latency ≈ max(3 requests) not sum.
    """
    tid = m['tmdb_id']
    base_params = {"api_key": TMDB_API_KEY}

    def _details():
        return safe_request(f"{TMDB_BASE_URL}/movie/{tid}", base_params)

    def _credits():
        return safe_request(f"{TMDB_BASE_URL}/movie/{tid}/credits", base_params)

    def _keywords():
        return safe_request(f"{TMDB_BASE_URL}/movie/{tid}/keywords", base_params)

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_det  = ex.submit(_details)
        f_cred = ex.submit(_credits)
        f_kw   = ex.submit(_keywords)

        det  = f_det.result()
        cred = f_cred.result()
        kw   = f_kw.result()

    if det:
        m['duration'] = det.get('runtime') or 120

    if cred:
        actors = [p['name'] for p in cred.get('cast', [])[:5]]
        m['cast'] = ', '.join(actors)
        crew = cred.get('crew', [])
        directors = [p['name'] for p in crew if p.get('job') == 'Director']
        m['director'] = directors[0] if directors else ''

    if kw:
        kws = [k['name'] for k in kw.get('keywords', [])[:10]]
        m['keywords'] = ' '.join(kws)

    return m


def seed_data():
    """
    Seed the database with movies from TMDB if empty.
    Uses concurrent page fetching for speed.
    """
    if Movie.query.count() > 0:
        print(f"✅ DB already has {Movie.query.count()} movies — skipping seed.")
        return

    print("🌱 Seeding database from TMDB …")
    t0 = time.time()

    genre_map = get_genre_map()
    LANGUAGES = [
        ('en', 'English'),
        ('ko', 'Korean'),
        ('hi', 'Hindi'),
        ('ja', 'Japanese'),
        ('fr', 'French'),
        ('es', 'Spanish'),
    ]
    PAGES_PER_LANG = 3

    # Concurrent page fetches across all languages
    raw_movies = []
    tasks = [
        (lang_code, genre_map, page, lang_label)
        for lang_code, lang_label in LANGUAGES
        for page in range(1, PAGES_PER_LANG + 1)
    ]

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_discover_page, *t): t for t in tasks}
        for f in as_completed(futures):
            raw_movies.extend(f.result())

    # De-duplicate by tmdb_id
    seen   = set()
    unique = []
    for m in raw_movies:
        if m['tmdb_id'] and m['tmdb_id'] not in seen and m['title']:
            seen.add(m['tmdb_id'])
            unique.append(m)

    print(f"  → {len(unique)} unique movies discovered — enriching …")

    # Concurrent enrichment (cast/director/runtime/keywords)
    enriched = []
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = [ex.submit(enrich_movie, m) for m in unique]
        for f in as_completed(futures):
            enriched.append(f.result())

    # Bulk insert
    objs = [
        Movie(
            tmdb_id     = m['tmdb_id'],
            title       = m['title'],
            genre       = m['genre'],
            year        = m['year'],
            rating      = m['rating'],
            description = m['description'],
            cast        = m['cast'],
            director    = m['director'],
            language    = m['language'],
            duration    = m['duration'],
            poster_url  = m['poster_url'],
            keywords    = m['keywords'],
            mood_tags   = m['mood_tags'],
            popularity  = m['popularity'],
        )
        for m in enriched
    ]
    db.session.bulk_save_objects(objs)
    db.session.commit()

    elapsed = time.time() - t0
    print(f"✅ Seeded {len(objs)} movies in {elapsed:.1f}s\n")


# ── ROUTES ───────────────────────────────────────────────────

@app.route('/')
def index():
    """Show login page first. Redirect to /app if already logged in."""
    if session.get('user_id'):
        return redirect(url_for('main_app'))
    return render_template('login.html')


@app.route('/app')
def main_app():
    """Main movie recommender app — requires login."""
    if not session.get('user_id'):
        return redirect(url_for('index'))
    trending  = Movie.query.order_by(Movie.popularity.desc()).limit(10).all()
    top_rated = Movie.query.order_by(Movie.rating.desc()).limit(150).all()
    return render_template('index.html', trending=trending, top_rated=top_rated)


# ── MOVIES API ────────────────────────────────────────────────

@app.route('/movies')
def movies():
    q     = request.args.get('q', '').strip()
    genre = request.args.get('genre', '').strip()
    lang  = request.args.get('language', '').strip()

    query = Movie.query
    if q:
        query = query.filter(Movie.title.ilike(f'%{q}%'))
    if genre:
        query = query.filter(Movie.genre.ilike(f'%{genre}%'))
    if lang:
        query = query.filter(Movie.language == lang)

    results = query.order_by(Movie.rating.desc()).limit(60).all()
    return jsonify([movie_to_dict(m) for m in results])


@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    m = Movie.query.get_or_404(movie_id)
    return jsonify(movie_to_dict(m))


# ── RECOMMENDATIONS ───────────────────────────────────────────

@app.route('/recommend')
def recommend():
    movie_id = request.args.get('movie_id', type=int)
    mood     = request.args.get('mood', '').strip().lower()
    duration = request.args.get('duration', '').strip().lower()

    rec = get_recommender()

    if movie_id:
        results = rec.get_similar(movie_id, top_n=8)
    elif mood:
        results = rec.get_by_mood(mood, top_n=50)
    elif duration == 'short':
        results = Movie.query.filter(Movie.duration < 100) \
                      .order_by(Movie.rating.desc()).limit(50).all()
    elif duration == 'long':
        results = Movie.query.filter(Movie.duration >= 150) \
                      .order_by(Movie.rating.desc()).limit(50).all()
    else:
        results = Movie.query.order_by(Movie.rating.desc()).limit(50).all()

    return jsonify([movie_to_dict(m) for m in results])


# ── AUTH ──────────────────────────────────────────────────────

@app.route('/register', methods=['POST'])
def register():
    data  = request.get_json() or {}
    name  = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    pw    = data.get('password', '')

    if not name or not email:
        return jsonify({'error': 'Name and email are required'}), 400
    if len(pw) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400

    user = User(name=name, email=email)
    user.set_password(pw)
    db.session.add(user)
    db.session.commit()

    session['user_id']   = user.id
    session['user_name'] = user.name
    return jsonify({'ok': True, 'name': user.name})


@app.route('/login', methods=['POST'])
def login():
    data  = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    pw    = data.get('password', '')

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(pw):
        return jsonify({'error': 'Invalid email or password'}), 401

    session['user_id']   = user.id
    session['user_name'] = user.name
    return jsonify({'ok': True, 'name': user.name})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/session-info')
def session_info():
    uid = session.get('user_id')
    if uid:
        return jsonify({'logged_in': True, 'name': session.get('user_name', '')})
    return jsonify({'logged_in': False})


# ── WATCHLIST ─────────────────────────────────────────────────

@app.route('/watchlist/add', methods=['POST'])
def watchlist_add():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Login required'}), 401

    data     = request.get_json() or {}
    movie_id = data.get('movie_id')
    if not movie_id:
        return jsonify({'error': 'movie_id required'}), 400

    if Watchlist.query.filter_by(user_id=uid, movie_id=movie_id).first():
        return jsonify({'ok': True, 'message': 'Already in watchlist'})

    db.session.add(Watchlist(user_id=uid, movie_id=movie_id))
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/watchlist/remove', methods=['POST'])
def watchlist_remove():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Login required'}), 401

    data     = request.get_json() or {}
    movie_id = data.get('movie_id')
    entry    = Watchlist.query.filter_by(user_id=uid, movie_id=movie_id).first()
    if entry:
        db.session.delete(entry)
        db.session.commit()
    return jsonify({'ok': True})


@app.route('/watchlist')
def watchlist():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Login required'}), 401

    entries = (Watchlist.query
               .filter_by(user_id=uid)
               .order_by(Watchlist.added_at.desc())
               .all())
    movies = [Movie.query.get(e.movie_id) for e in entries]
    return jsonify([movie_to_dict(m) for m in movies if m])


# ── USER RATINGS ──────────────────────────────────────────────

@app.route('/rate', methods=['POST'])
def rate():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Login required'}), 401

    data     = request.get_json() or {}
    movie_id = data.get('movie_id')
    stars    = data.get('rating')
    review   = data.get('review', '').strip()

    if not movie_id or not stars:
        return jsonify({'error': 'movie_id and rating required'}), 400

    existing = UserRating.query.filter_by(user_id=uid, movie_id=movie_id).first()
    if existing:
        existing.rating = stars
        existing.review = review
    else:
        db.session.add(UserRating(
            user_id=uid, movie_id=movie_id, rating=stars, review=review
        ))
    db.session.commit()
    return jsonify({'ok': True})


# ── ADMIN: re-seed endpoint (dev only) ────────────────────────

@app.route('/admin/reseed', methods=['POST'])
def reseed():
    """Drop all movies and re-fetch from TMDB. Dev-only!"""
    Movie.query.delete()
    db.session.commit()
    reset_recommender()
    with app.app_context():
        seed_data()
    return jsonify({'ok': True, 'count': Movie.query.count()})


# ── MAIN ──────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_data()
    app.run(debug=True)
