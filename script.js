let paragraphs = [];
let selectedParagraphId = 1;
let mediaStream = null;
let rawAudioData = [];
let audioContext = null;
let sourceNode = null;
let isRecording = false;
let recordingBlob = null;
let stopTimer = null;
let workletNode = null;
// Hard stop for a single capture. Keeps accidental long recordings from
// producing very large uploads and long server-side processing times.
const MAX_SECONDS = 120;

const paragraphOptions = document.getElementById('paragraphOptions');
const paragraphText = document.getElementById('paragraphText');
const recordBtn = document.getElementById('recordBtn');
const submitBtn = document.getElementById('submitBtn');
const recordStatus = document.getElementById('recordStatus');
const resultsSection = document.getElementById('resultsSection');
const renderedParagraph = document.getElementById('renderedParagraph');
const targetsTableBody = document.querySelector('#targetsTable tbody');
const scoreSummary = document.getElementById('scoreSummary');
const timingSummary = document.getElementById('timingSummary');
const recordingPlayback = document.getElementById('recordingPlayback');
const nativeExemplar = document.getElementById('nativeExemplar');
const docsRequestIdEl = document.getElementById('docsRequestId');
const analysisOverlay = document.getElementById('analysisOverlay');
const deepgramApiKeyInput = document.getElementById('deepgramApiKeyInput');
const saveDeepgramKeyBtn = document.getElementById('saveDeepgramKeyBtn');
const clearDeepgramKeyBtn = document.getElementById('clearDeepgramKeyBtn');
const deepgramKeyStatus = document.getElementById('deepgramKeyStatus');
const resultsError = document.getElementById('resultsError');


function getCookie(name) {
  const key = `${encodeURIComponent(name)}=`;
  const cookies = document.cookie ? document.cookie.split('; ') : [];
  for (const cookie of cookies) {
    if (cookie.startsWith(key)) return decodeURIComponent(cookie.slice(key.length));
  }
  return '';
}

function setDeepgramCookie(value) {
  const expires = new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toUTCString();
  document.cookie = `deepgram_api_key=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax`;
}

function clearDeepgramCookie() {
  document.cookie = 'deepgram_api_key=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax';
}

function refreshDeepgramStatus() {
  const hasKey = Boolean(getCookie('deepgram_api_key').trim());
  if (deepgramKeyStatus) {
    deepgramKeyStatus.textContent = hasKey ? 'Using your API key' : 'Using shared API key';
  }
}

function saveDeepgramKey() {
  if (!deepgramApiKeyInput) return;
  setDeepgramCookie(deepgramApiKeyInput.value || '');
  refreshDeepgramStatus();
}

function clearDeepgramKey() {
  if (deepgramApiKeyInput) deepgramApiKeyInput.value = '';
  clearDeepgramCookie();
  refreshDeepgramStatus();
}

// Convert normalized WebAudio samples (Float32 in [-1, 1]) to signed 16-bit PCM,
// which matches the backend WAV expectations (16 kHz, mono, 16-bit).
function to16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

// Build a minimal RIFF/WAV container around PCM samples so the blob can be
// uploaded directly to /api/analyze without additional transcoding.
function encodeWav(samples, sampleRate = 16000, channels = 1) {
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    view.setInt16(offset, samples[i], true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

function setRequestId(requestId) {
  if (docsRequestIdEl) docsRequestIdEl.textContent = requestId;
}

function renderParagraphOptions() {
  paragraphOptions.innerHTML = '';
  for (const p of paragraphs) {
    const label = document.createElement('label');
    label.className = 'paragraph-option';

    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'paragraph_id';
    radio.value = String(p.id);
    radio.checked = p.id === Number(selectedParagraphId);

    const text = document.createElement('span');
    text.textContent = String(p.id);

    label.appendChild(radio);
    label.appendChild(text);
    paragraphOptions.appendChild(label);
  }
}

async function loadParagraphs() {
  const resp = await fetch('/api/paragraphs');
  const data = await resp.json();
  setRequestId(data.request_id);
  paragraphs = data.paragraphs;
  renderParagraphOptions();
  renderParagraphText();
}

function getSelectedParagraph() {
  return paragraphs.find((p) => p.id === Number(selectedParagraphId));
}

function renderParagraphText() {
  const p = getSelectedParagraph();
  paragraphText.textContent = p ? p.display_text : '';
}

async function setupAudioWorklet() {
  // Define worklet inline so this app can remain a single-page static bundle
  // without an extra recorder processor file.
  const workletCode = `
    class RecorderWorkletProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const input = inputs[0];
        if (input && input[0]) {
          this.port.postMessage(input[0]);
        }
        return true;
      }
    }
    registerProcessor('recorder-worklet-processor', RecorderWorkletProcessor);
  `;
  const blob = new Blob([workletCode], { type: 'application/javascript' });
  const url = URL.createObjectURL(blob);
  try {
    await audioContext.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }

  workletNode = new AudioWorkletNode(audioContext, 'recorder-worklet-processor');
  // Each message contains one frame of mono float samples from the mic stream.
  workletNode.port.onmessage = (event) => {
    if (!isRecording) return;
    rawAudioData.push(new Float32Array(event.data));
  };

  sourceNode.connect(workletNode);
  // Connect to destination to keep some browser engines fully active while recording.
  workletNode.connect(audioContext.destination);
}

async function startRecording() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 16000,
      sampleSize: 16,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  rawAudioData = [];

  await setupAudioWorklet();

  isRecording = true;
  recordBtn.textContent = 'Stop Recording';
  recordBtn.classList.add('recording');
  recordStatus.textContent = 'Recording in progress...';
  submitBtn.disabled = true;

  // Auto-stop acts as a safety net if the user never clicks stop.
  stopTimer = setTimeout(() => {
    if (isRecording) stopRecording();
  }, MAX_SECONDS * 1000);
}

function stopRecording() {
  isRecording = false;
  clearTimeout(stopTimer);
  recordBtn.textContent = 'Start Recording';
  recordBtn.classList.remove('recording');
  recordStatus.textContent = 'Recording stopped.';

  if (workletNode) workletNode.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (audioContext) audioContext.close();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());

  workletNode = null;
  sourceNode = null;
  audioContext = null;
  mediaStream = null;

  // Concatenate captured worklet chunks into one contiguous buffer for WAV encoding.
  const mergedLength = rawAudioData.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(mergedLength);
  let offset = 0;
  for (const chunk of rawAudioData) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  const pcm16 = to16BitPCM(merged);
  recordingBlob = encodeWav(pcm16, 16000, 1);
  submitBtn.disabled = false;
  recordingPlayback.src = URL.createObjectURL(recordingBlob);
  recordingPlayback.hidden = false;
}

function setAnalysisOverlayVisible(isVisible) {
  if (!analysisOverlay) return;
  analysisOverlay.classList.toggle('is-visible', isVisible);
  analysisOverlay.setAttribute('aria-hidden', String(!isVisible));
}

function formatDuration(value) {
  if (value == null) return '-';
  return Number(value).toFixed(2);
}

function formatSeconds(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '0.000';
  return numeric.toFixed(3);
}

function formatCoreDurationsWithThreshold(target) {
  const s1 = target?.core_durations?.syll1;
  const s2 = target?.core_durations?.syll2;
  const durationText = `s1=${formatDuration(s1)} / s2=${formatDuration(s2)}`;

  if (target?.decision_method === 'learned_threshold' && target.learned_threshold != null) {
    const thresholdText = Number(target.learned_threshold).toFixed(3);
    const thresholdKey = target.threshold_key ? ` (${target.threshold_key})` : '';
    return `${durationText} | threshold=adaptive native_exemplar log(s1/s2)>=${thresholdText}${thresholdKey}`;
  }

  if (target?.decision_method === 'naive_duration' && s1 != null && s2 != null) {
    const fallbackSyllable = Number(s1) >= Number(s2) ? 's1' : 's2';
    return `${durationText} | threshold=fallback longer core vowel (${fallbackSyllable})`;
  }

  return `${durationText} | threshold=unavailable`;
}

function renderResults(data) {
  setRequestId(data.request_id);
  resultsSection.hidden = false;
  if (resultsError) {
    resultsError.hidden = true;
    resultsError.textContent = '';
  }
  const s = data.score_summary;
  scoreSummary.textContent = `Correct ${s.percent_correct}% (${s.scored_targets}/${s.total_targets} scored, missing ${s.missing_targets}, unaligned ${s.unaligned_targets})`;

  renderedParagraph.innerHTML = '';
  for (const w of data.render_words) {
    if (w.is_space) {
      renderedParagraph.append(document.createTextNode(w.text));
      continue;
    }
    const span = document.createElement('span');
    span.className = 'word';
    span.textContent = w.text;
    if (w.bg_norm == null) {
      span.style.backgroundColor = 'hsl(220, 25%, 88%)';
    } else {
      // Use confidence^3 (`bg_norm`) to drive hue: low=red, mid=orange/yellow, high=green.
      const confidence = Math.max(0, Math.min(1, Number(w.bg_norm)));
      const hue = confidence * 120;
      const lightness = 88 - confidence * 20;
      span.style.backgroundColor = `hsl(${hue.toFixed(1)}, 85%, ${lightness.toFixed(1)}%)`;
    }

    if (w.is_target) {
      if (w.target_status === 'missing' || w.target_status === 'unaligned') {
        span.classList.add('target-missing');
      } else if (w.target_correct === false) {
        span.classList.add('target-wrong');
      } else if (w.target_correct === true) {
        span.classList.add('target-correct');
      }
    }
    renderedParagraph.appendChild(span);
  }

  targetsTableBody.innerHTML = '';
  for (const t of data.targets) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${t.word}</td>
      <td>${t.label}</td>
      <td>${t.expected_stress ?? '-'}</td>
      <td>${t.inferred_stress ?? '-'}</td>
      <td>${formatCoreDurationsWithThreshold(t)}</td>
      <td>${t.status}</td>
      <td>${t.feedback}</td>
    `;
    targetsTableBody.appendChild(tr);
  }

  // Client-side total mirrors server timing fields to make perf bottlenecks visible
  // during manual QA without opening network traces.
  if (timingSummary) {
    const timing = data.timing || {};
    const totalProcessingTimeSec =
      Number(timing.bucket_json_read_process_sec || 0) +
      Number(timing.deepgram_api_sec || 0) +
      Number(timing.pocketsphinx_alignment_sec || 0) +
      Number(timing.persist_output_files_sec || 0);
    timingSummary.textContent = `Elapsed time: recording ${formatSeconds(timing.recording_duration_sec)}, read and process exemplar bucket files ${formatSeconds(timing.bucket_json_read_process_sec)}, Deepgram STT ${formatSeconds(timing.deepgram_api_sec)}, PocketSphinx alignment ${formatSeconds(timing.pocketsphinx_alignment_sec)}, writing bucket files ${formatSeconds(timing.persist_output_files_sec)}; total processing time: ${formatSeconds(totalProcessingTimeSec)} seconds`;
  }
}

function renderResultsError(data) {
  setRequestId(data.request_id);
  resultsSection.hidden = false;
  renderedParagraph.innerHTML = '';
  targetsTableBody.innerHTML = '';
  if (scoreSummary) scoreSummary.textContent = '';
  if (timingSummary) timingSummary.textContent = '';
  if (resultsError) {
    resultsError.textContent = data.error || 'Analysis failed.';
    resultsError.hidden = false;
  }
}

async function submitAnalysis() {
  if (!recordingBlob) return;
  const form = new FormData();
  form.append('paragraph_id', String(selectedParagraphId));
  form.append('audio_wav', recordingBlob, 'recording.wav');
  form.append('native_exemplar', nativeExemplar && nativeExemplar.checked ? 'true' : 'false');

  setAnalysisOverlayVisible(true);
  submitBtn.disabled = true;
  try {
    const resp = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) {
      renderResultsError(data);
      return;
    }
    renderResults(data);
  } finally {
    setAnalysisOverlayVisible(false);
    submitBtn.disabled = false;
  }
}

paragraphOptions.addEventListener('change', (e) => {
  if (!(e.target instanceof HTMLInputElement) || e.target.name !== 'paragraph_id') return;
  selectedParagraphId = Number(e.target.value);
  renderParagraphText();
});

recordBtn.addEventListener('click', async () => {
  if (!isRecording) {
    try {
      await startRecording();
    } catch (e) {
      alert('Microphone access failed.');
    }
  } else {
    stopRecording();
  }
});

submitBtn.addEventListener('click', submitAnalysis);
if (saveDeepgramKeyBtn) saveDeepgramKeyBtn.addEventListener('click', saveDeepgramKey);
if (clearDeepgramKeyBtn) clearDeepgramKeyBtn.addEventListener('click', clearDeepgramKey);

refreshDeepgramStatus();
loadParagraphs();

function addCopyButtonsToExamples() {
  const fallbackCopy = (text) => {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.setAttribute('readonly', '');
    textArea.style.position = 'absolute';
    textArea.style.left = '-9999px';
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
  };

  document.querySelectorAll('.docs pre:not(.no-copy)').forEach((pre) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'pre-block';
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-pre-btn';
    button.setAttribute('aria-label', 'Copy example to clipboard');
    button.title = 'Copy';
    button.textContent = '📋';

    button.addEventListener('click', async () => {
      const text = pre.textContent;
      try {
        await navigator.clipboard.writeText(text);
      } catch (error) {
        fallbackCopy(text);
      }
      button.textContent = '✓';
      window.setTimeout(() => {
        button.textContent = '📋';
      }, 1200);
    });

    wrapper.appendChild(button);
  });
}

addCopyButtonsToExamples();
