#!/bin/bash
cd /code
python - <<'PY'
from app.db import SessionLocal
from app.models import Feedback, Recommendation
from joblib import load, dump
import numpy as np, os
hyb_path="app/hybrid_model.joblib"
model=load(hyb_path)
db=SessionLocal()
likes=db.query(Feedback).filter_by(liked=True).all()
if likes:
    X=[ [fb.user_id%1000, fb.track_id%1000] for fb in likes ]
    y=[1]*len(likes)
    model.partial_fit(np.asarray(X,dtype=float), y, classes=[0,1])
    dump(model, hyb_path)
db.close()
PY
