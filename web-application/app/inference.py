import joblib, librosa, numpy as np, os, json
from catboost import CatBoostClassifier

HYBRID_PATH = os.path.join(os.path.dirname(__file__), "hybrid_model.joblib")
GENRE_PATH  = os.path.join(os.path.dirname(__file__), "by_genre_catboost_recommendation_model.cbm")

hybrid_model  = joblib.load(HYBRID_PATH)
genre_model   = CatBoostClassifier()
genre_model.load_model(GENRE_PATH)

def preprocess_profile(form_data):
    vec = [
        hash(form_data.get("full_name")) & 0xFFFFFFFF,
        hash(form_data.get("city")) & 0xFFFF,
        int(form_data.get("age")),
        1 if form_data.get("gender")=="female" else 0,
    ]
    return np.asarray(vec, dtype=np.float32)

def recommend_tracks(vec, top_n=20):
    scores = hybrid_model.predict_proba(vec.reshape(1,-1))[0,1]
    top_ids = np.argsort(scores)[::-1][:top_n]
    return top_ids.tolist()

def extract_audio_features(mp3_bytes):
    y, sr = librosa.load(mp3_bytes, sr=None)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def by_genre_reco(mp3_bytes, all_tracks_feats):
    feats = extract_audio_features(mp3_bytes)
    genre = genre_model.predict(feats.reshape(1,-1))[0]
    same  = [t for t,g in all_tracks_feats if g==genre]
    nearest = min(same, key=lambda t: np.linalg.norm(t[0]-feats))
    return nearest[1]  # track_id
