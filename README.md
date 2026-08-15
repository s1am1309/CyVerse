# Moodify

A real-time **Facial Expression Estimate** app that runs entirely on-device
in the browser. No paid API, no API key, no backend, no database.

Moodify uses your camera (via `getUserMedia`) and a free, open-source,
client-side computer-vision model (`face-api.js`, built on TensorFlow.js)
to estimate visible facial-expression patterns — happy, sad, angry,
surprised, fearful, disgusted, or neutral — directly in your browser.
No image or video frame is ever uploaded anywhere.

> Moodify estimates **visible facial-expression patterns** only. It does
> not know how you actually feel, and is not a medical, psychological,
> or emotional-diagnosis tool.

## Project structure

```
index.html      Markup for the landing screen, camera/analysis screen,
                 results panel, history panel, and error/toast UI
style.css        Dark, glassmorphic "AI lab" visual design system
script.js        Camera lifecycle, on-device detection loop, UI state,
                 session history, and error handling
/models          (Optional) place to self-host model weight files —
                 see models/README.md
/assets          Static assets (icons, etc.)
```

## Running it

Because it's a static site, you can simply open `index.html` in a
browser — though most browsers require a real HTTP origin (not
`file://`) to grant camera permissions, so a tiny static server is
recommended for local testing:

```bash
npx serve .
# or
python3 -m http.server 8080
```

Then visit the printed local URL and click **Start Camera**.

## Deploying

Moodify is a static site with no build step. Deploy the folder as-is to:

- **GitHub Pages** — push this folder to a repo and enable Pages on
  the branch/folder.
- **Vercel** — `vercel deploy` from this folder, or connect the repo
  (framework preset: "Other" / static).
- **Netlify** — drag-and-drop this folder onto Netlify, or connect the
  repo with build command empty and publish directory `.`.

Camera access requires HTTPS (or `localhost`) — all three platforms
serve over HTTPS by default.

## How detection works

1. `face-api.js` is loaded from a CDN (`<script>` tag in `index.html`) —
   free and open-source, no key required.
2. On "Start Camera", two small on-device models are loaded:
   a fast face detector (`tinyFaceDetector`) and an expression
   classifier (`faceExpressionNet`).
3. Every ~280ms, a frame from the live `<video>` element is run through
   both models **in the browser**, producing a bounding box and a
   probability for each of the 7 expression classes.
4. The highest-probability class is shown as the current estimate, with
   a confidence percentage, animated bars for every class, and a radial
   "mood core" visualization.
5. Readings are periodically appended to a session-only history, stored
   in `localStorage` — never sent anywhere.

See `models/README.md` for how to fully self-host the model weight
files instead of using the default CDN.

## Browser support

Requires a browser with `getUserMedia` and WebAssembly/WebGL support for
TensorFlow.js — recent Chrome, Edge, Firefox, and Safari (including iOS
Safari) all work. Moodify detects unsupported browsers, denied camera
permission, missing cameras, and model-loading failures, and shows a
friendly in-app message for each instead of crashing.
