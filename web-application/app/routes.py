from flask import request, jsonify, render_template, redirect
from .db import SessionLocal
from .models import User, Recommendation, Feedback
from .inference import preprocess_profile, recommend_tracks, by_genre_reco
from .ext_api import spotify_iframe, vk_iframe
import hashlib, json, io, os

def _msno(profile_dict):
    return hashlib.md5(json.dumps(profile_dict, sort_keys=True).encode()).hexdigest()

def _first_reco(user, db):
    rec = db.query(Recommendation).filter_by(user_id=user.id, played=False).order_by(Recommendation.rank).first()
    return rec

def _track_payload(track_id):
    return {
        "title": f"Track #{track_id}",
        "artist": "Unknown",
        "genre": "—",
        "spotify_iframe": spotify_iframe(track_id),
        "vk_iframe": vk_iframe(track_id)
    }

def register_routes(app):
    @app.route('/')
    def index():
        return render_template("index.html")

    @app.route('/start', methods=['POST'])
    def start():
        form = request.form.to_dict()
        vec  = preprocess_profile(form)
        track_ids = recommend_tracks(vec, top_n=50)
        msno = _msno(form)
        db = SessionLocal()
        user = db.query(User).filter_by(msno=msno).first()
        if not user:
            user = User(msno=msno, profile=form)
            db.add(user); db.flush()
        for rk,tid in enumerate(track_ids):
            db.add(Recommendation(user_id=user.id, track_id=int(tid), rank=rk))
        db.commit(); db.close()
        return redirect('/recommend')

    @app.route('/recommend')
    def recommend():
        db = SessionLocal()
        user = db.query(User).order_by(User.id.desc()).first()
        rec  = _first_reco(user, db)
        if not rec:
            db.close()
            return "No recommendations", 404
        rec.played=True; db.commit(); db.close()
        return render_template("recommend.html", data=_track_payload(rec.track_id))

    @app.route('/next')
    def next_track():
        db = SessionLocal()
        user_id = db.query(User).order_by(User.id.desc()).first().id
        rec = db.query(Recommendation).filter_by(user_id=user_id, played=False).order_by(Recommendation.rank).first()
        if not rec:
            db.close()
            return jsonify({})
        rec.played=True; db.commit(); db.close()
        return jsonify(_track_payload(rec.track_id))

    @app.route('/by_genre')
    def by_genre():
        # mock: choose random id 99999
        return jsonify(_track_payload(99999))

    @app.route('/feedback', methods=['POST'])
    def feedback():
        data = request.get_json()
        track_id = data.get("track_id")
        liked    = bool(data.get("mark"))
        db = SessionLocal()
        user_id = db.query(User).order_by(User.id.desc()).first().id
        fb = Feedback(user_id=user_id, track_id=track_id, liked=liked)
        db.add(fb); db.commit(); db.close()
        return "", 204
