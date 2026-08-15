# /models

Moodify loads its on-device face-detection and expression-recognition
weights (`face-api.js`, built on TensorFlow.js) from a public CDN by
default, so the app works immediately with **zero setup**:

```
https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights
```

This is still 100% client-side and free — no API key, no account, no
backend — the browser just fetches static model files directly, the
same way it fetches this page's CSS or fonts.

## Self-hosting the models (optional)

If you'd rather ship every file yourself (e.g. for a fully offline
deployment, or to avoid depending on a third-party CDN), download the
weight files into this folder:

1. Get the files from the face-api.js weights folder:
   `https://github.com/justadudewhohacks/face-api.js/tree/master/weights`
2. Copy at least these into `/models`:
   - `tiny_face_detector_model-weights_manifest.json`
   - `tiny_face_detector_model-shard1`
   - `face_expression_model-weights_manifest.json`
   - `face_expression_model-shard1`
3. In `script.js`, move `'./models'` to the front of the `MODEL_URLS`
   array so it's tried first.

That's it — no build step, no server code. The whole app (including
the models) can then be hosted as static files with no network
dependency at runtime.
