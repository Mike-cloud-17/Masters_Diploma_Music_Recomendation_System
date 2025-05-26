import os, requests

SPOTIFY_TOKEN = os.getenv("SPOTIFY_TOKEN")
VK_TOKEN      = os.getenv("VK_TOKEN")

def spotify_iframe(track_id):
    return f'<iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator" width="100%" height="80" frameBorder="0" allow="autoplay; clipboard-write; encrypted-media; picture-in-picture"></iframe>'

def spotify_audio(track_id):
    headers={"Authorization":f"Bearer {SPOTIFY_TOKEN}"}
    r = requests.get(f"https://api.spotify.com/v1/tracks/{track_id}", headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def vk_iframe(query):
    return f'<iframe src="https://vk.com/video_ext.php?search={query}" width="100%" height="360" frameborder="0" allowfullscreen></iframe>'
