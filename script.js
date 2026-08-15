/* =====================================================================
   MOODIFY — script.js
   All computer vision runs on-device via face-api.js (TensorFlow.js
   under the hood). Nothing here ever sends a frame, image, or reading
   to a server. No API key. No backend.
   ===================================================================== */

(() => {
  'use strict';

  /* -------------------------------------------------------------------
     0. Config
     ------------------------------------------------------------------- */

  // Where to load the (free, open-source) face-api.js model weights from.
  // Tries each URL in order until one works, so this app runs out of the
  // box from a CDN, but can be fully self-hosted by downloading the
  // weight files into /models (see /models/README.md) and moving that
  // path first in this list.
  const MODEL_URLS = [
    'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights',
    './models'
  ];

  const DETECTION_INTERVAL_MS = 280;   // how often we run inference
  const HISTORY_LOG_INTERVAL_MS = 4000; // min gap between history entries
  const HISTORY_MAX_ENTRIES = 40;
  const STABLE_FACE_GRACE_MS = 700;    // frames-lost tolerance before "no face"

  const EXPRESSIONS = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral'];

  const EXPR_META = {
    happy:     { symbol: '☺', color: '#FFC857', label: 'Happy',
      explain: 'Raised cheeks and a widened mouth suggest a happy expression.' },
    sad:       { symbol: '⌄', color: '#4C7EF3', label: 'Sad',
      explain: 'Lowered mouth corners and relaxed brows suggest a sad expression.' },
    angry:     { symbol: '▲', color: '#FF4757', label: 'Angry',
      explain: 'Lowered, drawn-together brows suggest an angry expression.' },
    surprised: { symbol: '✦', color: '#FF61D8', label: 'Surprised',
      explain: 'Raised eyebrows and widened eyes suggest a surprised expression.' },
    fearful:   { symbol: '◇', color: '#2EE6D6', label: 'Fearful',
      explain: 'Tensed brows and widened eyes suggest a fearful expression.' },
    disgusted: { symbol: '✕', color: '#7ED957', label: 'Disgusted',
      explain: 'A wrinkled nose and raised upper lip suggest a disgusted expression.' },
    neutral:   { symbol: '◎', color: '#8C93B0', label: 'Neutral',
      explain: 'No strong expression pattern is currently detected.' },
  };

  const CORE_CIRCUMFERENCE = 2 * Math.PI * 92; // matches SVG r=92

  /* -------------------------------------------------------------------
     1. DOM references
     ------------------------------------------------------------------- */

  const $ = (id) => document.getElementById(id);

  const screenLanding   = $('screen-landing');
  const screenCamera    = $('screen-camera');
  const btnStart        = $('btn-start');
  const btnStop         = $('btn-stop');
  const btnClearHistory = $('btn-clear-history');

  const video       = $('video');
  const overlay     = $('overlay');
  const videoFrame  = $('video-frame');
  const stageMsg    = $('stage-msg');
  const stageMsgText = $('stage-msg-text');
  const stageCaption = $('stage-caption');

  const brandStatus     = $('brand-status');
  const brandStatusText = $('brand-status-text');

  const coreProgress = $('core-progress');
  const moodEmoji     = $('mood-emoji');
  const moodConfidence = $('mood-confidence');
  const resultLabel   = $('result-label');
  const resultExplainer = $('result-explainer');
  const barsContainer = $('bars');

  const historyList  = $('history-list');
  const historyEmpty = $('history-empty');

  const toastRegion = $('toast-region');
  const errorModal  = $('error-modal');
  const errorTitle  = $('error-title');
  const errorBody   = $('error-body');
  const errorRetry  = $('error-retry');
  const errorDismiss = $('error-dismiss');

  const overlayCtx = overlay.getContext('2d');

  /* -------------------------------------------------------------------
     2. Global state
     ------------------------------------------------------------------- */

  const state = {
    modelsReady: false,
    stream: null,
    detectionTimer: null,
    lastFaceSeenAt: 0,
    lastHistoryLogAt: 0,
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    retryAction: null,
  };

  if (state.reducedMotion) document.body.classList.add('reduced-motion');

  /* -------------------------------------------------------------------
     3. Small UI helpers
     ------------------------------------------------------------------- */

  function setBrandStatus(mode, text) {
    brandStatus.dataset.state = mode; // 'idle' | 'live' | 'error'
    brandStatusText.textContent = text;
  }

  function toast(message, kind = 'info', duration = 3600) {
    const el = document.createElement('div');
    el.className = `toast toast--${kind}`;
    el.setAttribute('role', 'status');
    el.textContent = message;
    toastRegion.appendChild(el);
    setTimeout(() => {
      el.style.transition = 'opacity .3s ease';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 320);
    }, duration);
  }

  function showErrorModal({ title, body, retry = null }) {
    errorTitle.textContent = title;
    errorBody.textContent = body;
    state.retryAction = retry;
    errorRetry.hidden = !retry;
    errorModal.hidden = false;
    setBrandStatus('error', 'Error');
  }

  function hideErrorModal() {
    errorModal.hidden = true;
  }

  errorDismiss.addEventListener('click', hideErrorModal);
  errorRetry.addEventListener('click', () => {
    hideErrorModal();
    if (typeof state.retryAction === 'function') state.retryAction();
  });

  function showStageMsg(text) {
    stageMsgText.textContent = text;
    stageMsg.hidden = false;
  }
  function hideStageMsg() {
    stageMsg.hidden = true;
  }

  function switchScreen(name) {
    screenLanding.hidden = name !== 'landing';
    screenCamera.hidden = name !== 'camera';
  }

  /* -------------------------------------------------------------------
     4. Ambient particle background (subtle; skipped under reduced motion)
     ------------------------------------------------------------------- */

  (function initBackgroundCanvas() {
    const canvas = $('bg-canvas');
    const ctx = canvas.getContext('2d');
    let w, h, particles = [];

    function resize() {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const COUNT = Math.min(60, Math.floor((window.innerWidth * window.innerHeight) / 26000));
    for (let i = 0; i < COUNT; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.4 + 0.4,
        vx: (Math.random() - 0.5) * 0.12,
        vy: (Math.random() - 0.5) * 0.12,
        a: Math.random() * 0.35 + 0.08,
      });
    }

    function draw() {
      ctx.clearRect(0, 0, w, h);
      particles.forEach((p) => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = w; if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h; if (p.y > h) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(140, 176, 255, ${p.a})`;
        ctx.fill();
      });
      if (!state.reducedMotion) requestAnimationFrame(draw);
    }

    if (state.reducedMotion) {
      draw(); // draw one static frame only
    } else {
      requestAnimationFrame(draw);
    }
  })();

  /* -------------------------------------------------------------------
     5. Browser support check
     ------------------------------------------------------------------- */

  function browserIsSupported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.faceapi);
  }

  /* -------------------------------------------------------------------
     6. Model loading (on-device, static files — no key, no backend)
     ------------------------------------------------------------------- */

  async function loadModels() {
    if (state.modelsReady) return true;
    if (typeof faceapi === 'undefined') {
      throw new Error('face-api.js failed to load from CDN.');
    }

    let lastErr = null;
    for (const url of MODEL_URLS) {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(url),
          faceapi.nets.faceExpressionNet.loadFromUri(url),
        ]);
        state.modelsReady = true;
        return true;
      } catch (err) {
        lastErr = err;
        // try the next source
      }
    }
    throw lastErr || new Error('Model files could not be loaded from any source.');
  }

  /* -------------------------------------------------------------------
     7. Camera lifecycle
     ------------------------------------------------------------------- */

  async function requestCamera() {
    const constraints = {
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 960 },
        height: { ideal: 720 },
      },
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.stream = stream;
    video.srcObject = stream;
    await new Promise((resolve) => {
      if (video.readyState >= 2) return resolve();
      video.onloadedmetadata = () => resolve();
    });
    await video.play();
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    video.srcObject = null;
  }

  function friendlyCameraError(err) {
    const name = err && err.name;
    switch (name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return {
          title: 'Camera access denied',
          body: 'Moodify needs camera permission to estimate a facial expression. Enable camera access for this site in your browser settings, then try again.',
          retry: startExperience,
        };
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return {
          title: 'No camera found',
          body: 'We couldn\u2019t find a camera on this device. Connect a camera or try on a device that has one.',
          retry: startExperience,
        };
      case 'NotReadableError':
      case 'TrackStartError':
        return {
          title: 'Camera unavailable',
          body: 'Your camera seems to be in use by another application. Close other apps or tabs using the camera and try again.',
          retry: startExperience,
        };
      case 'OverconstrainedError':
        return {
          title: 'Camera constraints not supported',
          body: 'Your camera doesn\u2019t support the requested resolution. Try again \u2014 we\u2019ll fall back to a lower resolution.',
          retry: startExperience,
        };
      case 'SecurityError':
        return {
          title: 'Camera blocked',
          body: 'Camera access requires a secure (https) connection. Please load Moodify over HTTPS.',
          retry: null,
        };
      default:
        return {
          title: 'Camera error',
          body: 'Something went wrong while starting the camera. Please try again.',
          retry: startExperience,
        };
    }
  }

  /* -------------------------------------------------------------------
     8. Main start / stop flow
     ------------------------------------------------------------------- */

  async function startExperience() {
    if (!browserIsSupported()) {
      showErrorModal({
        title: 'Unsupported browser',
        body: 'Your browser doesn\u2019t support the camera or on-device vision features Moodify needs. Try the latest Chrome, Edge, Firefox, or Safari.',
        retry: null,
      });
      return;
    }

    btnStart.disabled = true;
    switchScreen('camera');
    showStageMsg('Calibrating on-device model…');
    setBrandStatus('idle', 'Loading model');
    videoFrame.classList.remove('is-scanning');

    try {
      await loadModels();
    } catch (err) {
      console.error('Model load failed:', err);
      switchScreen('landing');
      btnStart.disabled = false;
      showErrorModal({
        title: 'Model failed to load',
        body: 'The on-device expression model couldn\u2019t be downloaded, likely due to a network issue. Check your connection and try again.',
        retry: startExperience,
      });
      return;
    }

    showStageMsg('Requesting camera access…');
    try {
      await requestCamera();
    } catch (err) {
      console.error('Camera error:', err);
      switchScreen('landing');
      btnStart.disabled = false;
      showErrorModal(friendlyCameraError(err));
      return;
    }

    hideStageMsg();
    btnStart.disabled = false;
    videoFrame.classList.add('is-scanning');
    setBrandStatus('live', 'Live · on-device');
    stageCaption.textContent = 'Position your face in the frame.';
    resizeOverlay();
    beginDetectionLoop();
  }

  function stopExperience() {
    clearInterval(state.detectionTimer);
    state.detectionTimer = null;
    stopCamera();
    videoFrame.classList.remove('is-scanning');
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    setBrandStatus('idle', 'Idle');
    switchScreen('landing');
    resetResultsUI();
  }

  btnStart.addEventListener('click', startExperience);
  btnStop.addEventListener('click', stopExperience);

  window.addEventListener('beforeunload', stopCamera);
  document.addEventListener('visibilitychange', () => {
    // Pause inference (not the stream) when the tab is hidden, to save battery.
    if (document.hidden) {
      clearInterval(state.detectionTimer);
      state.detectionTimer = null;
    } else if (state.stream && !state.detectionTimer) {
      beginDetectionLoop();
    }
  });

  /* -------------------------------------------------------------------
     9. Overlay canvas sizing
     ------------------------------------------------------------------- */

  function resizeOverlay() {
    const rect = videoFrame.getBoundingClientRect();
    overlay.width = rect.width;
    overlay.height = rect.height;
  }
  window.addEventListener('resize', () => {
    if (state.stream) resizeOverlay();
  });

  /* -------------------------------------------------------------------
     10. Detection loop
     ------------------------------------------------------------------- */

  function beginDetectionLoop() {
    clearInterval(state.detectionTimer);
    state.detectionTimer = setInterval(runDetection, DETECTION_INTERVAL_MS);
  }

  let detecting = false;

  async function runDetection() {
    if (detecting || !state.stream || video.readyState < 2) return;
    detecting = true;
    try {
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
        .withFaceExpressions();

      handleDetections(detections);
    } catch (err) {
      // Non-fatal per-frame errors (e.g. a dropped frame) shouldn't crash the loop.
      console.warn('Detection frame error:', err);
    } finally {
      detecting = false;
    }
  }

  function handleDetections(detections) {
    const now = performance.now();
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    if (!detections || detections.length === 0) {
      if (now - state.lastFaceSeenAt > STABLE_FACE_GRACE_MS) {
        setNoFaceState();
      }
      return;
    }

    state.lastFaceSeenAt = now;

    let primary = detections[0];
    if (detections.length > 1) {
      // Multiple faces: analyze the largest (closest) face, and let the user know.
      primary = detections.reduce((a, b) =>
        (a.detection.box.area > b.detection.box.area ? a : b)
      );
      stageCaption.textContent = `${detections.length} faces detected — showing the primary face.`;
    } else {
      stageCaption.textContent = 'Face locked. Analyzing expression…';
    }

    drawFaceBoxes(detections, primary);
    updateExpressionUI(primary.expressions);
  }

  function setNoFaceState() {
    stageCaption.textContent = 'No face detected — center your face in the frame.';
    resultLabel.textContent = 'No face detected';
    resultExplainer.textContent = 'Moodify needs a clear, front-facing view of your face to estimate an expression.';
    moodConfidence.textContent = '--%';
    moodEmoji.textContent = '◎';
    setCoreProgress(0, EXPR_META.neutral.color);
    barsContainer.querySelectorAll('.bar-fill').forEach((el) => (el.style.width = '0%'));
    barsContainer.querySelectorAll('.bar-value').forEach((el) => (el.textContent = '0%'));
  }

  function resetResultsUI() {
    setNoFaceState();
    barsContainer.innerHTML = '';
    renderBarSkeleton();
  }

  /* -------------------------------------------------------------------
     11. Drawing the face box + scan HUD on canvas
     ------------------------------------------------------------------- */

  function drawFaceBoxes(all, primary) {
    const scaleX = overlay.width / video.videoWidth;
    const scaleY = overlay.height / video.videoHeight;

    all.forEach((d) => {
      const isPrimary = d === primary;
      const { x, y, width, height } = d.detection.box;
      const bx = x * scaleX, by = y * scaleY, bw = width * scaleX, bh = height * scaleY;
      const color = isPrimary ? currentMoodColor() : 'rgba(148,176,255,0.5)';

      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = isPrimary ? 2 : 1.2;
      overlayCtx.shadowColor = color;
      overlayCtx.shadowBlur = isPrimary ? 14 : 0;

      // rounded rect
      const r = 14;
      overlayCtx.beginPath();
      overlayCtx.moveTo(bx + r, by);
      overlayCtx.arcTo(bx + bw, by, bx + bw, by + bh, r);
      overlayCtx.arcTo(bx + bw, by + bh, bx, by + bh, r);
      overlayCtx.arcTo(bx, by + bh, bx, by, r);
      overlayCtx.arcTo(bx, by, bx + bw, by, r);
      overlayCtx.closePath();
      overlayCtx.stroke();

      if (isPrimary) {
        // small corner ticks for a HUD feel
        overlayCtx.shadowBlur = 0;
        overlayCtx.fillStyle = color;
        const tick = 4;
        [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]].forEach(([tx, ty]) => {
          overlayCtx.beginPath();
          overlayCtx.arc(tx, ty, tick, 0, Math.PI * 2);
          overlayCtx.fill();
        });
      }
    });
  }

  let lastMoodColor = EXPR_META.neutral.color;
  function currentMoodColor() { return lastMoodColor; }

  /* -------------------------------------------------------------------
     12. Expression UI: label, core ring, bars, explainer
     ------------------------------------------------------------------- */

  function renderBarSkeleton() {
    barsContainer.innerHTML = EXPRESSIONS.map((key, i) => {
      const meta = EXPR_META[key];
      return `
        <div class="bar-row" id="bar-row-${key}" style="animation-delay:${i * 40}ms">
          <span class="bar-name">${meta.label}</span>
          <span class="bar-track"><span class="bar-fill" id="bar-fill-${key}" style="--bar-color:${meta.color}"></span></span>
          <span class="bar-value" id="bar-value-${key}">0%</span>
        </div>`;
    }).join('');
  }
  renderBarSkeleton();

  function setCoreProgress(pct, color) {
    const offset = CORE_CIRCUMFERENCE * (1 - pct / 100);
    coreProgress.style.stroke = color;
    coreProgress.style.strokeDashoffset = String(offset);
    document.documentElement.style.setProperty('--mood-color', color);
    document.documentElement.style.setProperty('--mood-color-soft', hexToSoft(color));
  }

  function hexToSoft(hex, alpha = 0.18) {
    const c = hex.replace('#', '');
    const r = parseInt(c.substring(0, 2), 16);
    const g = parseInt(c.substring(2, 4), 16);
    const b = parseInt(c.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function updateExpressionUI(expressions) {
    // expressions: { neutral, happy, sad, angry, fearful, disgusted, surprised } (0..1)
    const entries = EXPRESSIONS.map((key) => [key, expressions[key] || 0]);
    entries.sort((a, b) => b[1] - a[1]);
    const [topKey, topScore] = entries[0];
    const meta = EXPR_META[topKey];
    const pct = Math.round(topScore * 100);

    lastMoodColor = meta.color;

    resultLabel.textContent = meta.label;
    resultLabel.style.color = meta.color;
    resultExplainer.textContent = `${meta.explain} (Facial Expression Estimate — not a measure of true emotion.)`;
    moodEmoji.textContent = meta.symbol;
    moodConfidence.textContent = `${pct}%`;
    setCoreProgress(pct, meta.color);

    EXPRESSIONS.forEach((key) => {
      const score = Math.round((expressions[key] || 0) * 100);
      const fill = $(`bar-fill-${key}`);
      const val = $(`bar-value-${key}`);
      const row = $(`bar-row-${key}`);
      if (fill) fill.style.width = `${score}%`;
      if (val) val.textContent = `${score}%`;
      if (row) row.classList.toggle('is-top', key === topKey);
    });

    maybeLogHistory(meta.label, pct);
  }

  /* -------------------------------------------------------------------
     13. Session history (localStorage — never leaves the browser)
     ------------------------------------------------------------------- */

  const HISTORY_KEY = 'moodify_session_history';

  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function saveHistory(entries) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
    } catch {
      // localStorage unavailable (private mode, quota, etc.) — fail silently,
      // history simply won't persist across reloads this session.
    }
  }

  function maybeLogHistory(label, pct) {
    const now = Date.now();
    if (now - state.lastHistoryLogAt < HISTORY_LOG_INTERVAL_MS) return;
    state.lastHistoryLogAt = now;

    const entry = { t: now, label, pct };
    const entries = loadHistory();
    entries.unshift(entry);
    if (entries.length > HISTORY_MAX_ENTRIES) entries.length = HISTORY_MAX_ENTRIES;
    saveHistory(entries);
    renderHistory(entries);
  }

  function formatTime(ts) {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function renderHistory(entries) {
    entries = entries || loadHistory();
    if (!entries.length) {
      historyList.innerHTML = '<li class="history-empty" id="history-empty">Your expression readings will appear here.</li>';
      return;
    }
    historyList.innerHTML = entries.map((e) => {
      const key = Object.keys(EXPR_META).find((k) => EXPR_META[k].label === e.label) || 'neutral';
      const color = EXPR_META[key].color;
      return `
        <li class="history-row">
          <span class="history-time">${formatTime(e.t)}</span>
          <span class="history-dot" style="--dot-color:${color}"></span>
          <span class="history-expr">${e.label}</span>
          <span class="history-conf">${e.pct}%</span>
        </li>`;
    }).join('');
  }

  btnClearHistory.addEventListener('click', () => {
    saveHistory([]);
    renderHistory([]);
    toast('History cleared.', 'info', 2200);
  });

  /* -------------------------------------------------------------------
     14. Init
     ------------------------------------------------------------------- */

  function init() {
    resetResultsUI();
    renderHistory(loadHistory());
    setBrandStatus('idle', 'Idle');

    if (!browserIsSupported()) {
      // Don't block the landing page — just warn, and catch it properly on Start.
      toast('This browser may not fully support on-device camera detection.', 'warn', 5000);
    }
  }

  init();
})();
