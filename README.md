# Masters Diploma: Music Recommendation System

## Annotation
The thesis is devoted to developing a hybrid neural music-recommendation system that merges collaborative and content-based signals to precisely predict the probability of a track being replayed—a robust indicator of a user’s genuine liking.
On the public KKBOX dataset (7 056 972 training events) an extensive data-analysis pipeline was built: ISRC decoding, statistical popularity counters, latent SVD embeddings, temporal windows and probabilistic preference profiles.

Thirteen classical algorithms were benchmarked; the best tabular model was LightGBM (ROC-AUC = 0.824).
The proposed ensemble—Joint Embedding Neural Network (JENN) + Functional ELU-Net + LightGBM—combines deep collaborative representations, a robust content filter and gradient boosting on engineered features. The combined model reached ROC-AUC = 0.8449, outperforming the best single neural net by 0.5 pp and consistently improving F1-score, accuracy, recall and precision.

The system is deployed in a Flask application: the user enters basic attributes that are pre-processed and fed to the Hybrid Model to generate a personalised track pool; audio is fetched via the Spotify API, video via the VK Video API.
The “Next” flow relies on the hybrid ensemble, while “By genre” calls a CatBoost model with Librosa spectral features to pick tracks of the closest genre colour. Every like/dislike is stored in PostgreSQL and used in nightly fine-tuning, closing the loop of on-line personalisation.

The architecture transfers easily to related recommendation tasks within the T-Bank ecosystem—from automatic soundtrack selection for stories to personalised shelves in the “Shopping” and “Travel” services.
The results confirm the practical viability of the ensemble and show it to be competitive with industrial systems such as Spotify, Apple Music and Yandex Music.

![telegram-cloud-photo-size-2-5285239669463839883-y](https://github.com/user-attachments/assets/03985732-d624-44cb-bc8a-738d0b91e6d6)
