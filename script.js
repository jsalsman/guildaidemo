let paragraphs = [];
let selectedParagraphId = 1;
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let rawAudioData = [];
let audioContext = null;
let processor = null;
let sourceNode = null;
let isRecording = false;
let recordingBlob = null;
let stopTimer = null;
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
const recordingPlayback = document.getElementById('recordingPlayback');
const docsRequestIdEl = document.getElementById('docsRequestId');

function to16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

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
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  rawAudioData = [];
  sourceNode.connect(processor);
  processor.connect(audioContext.destination);

  processor.onaudioprocess = (e) => {
    if (!isRecording) return;
    const input = e.inputBuffer.getChannelData(0);
    rawAudioData.push(new Float32Array(input));
  };

  isRecording = true;
  recordBtn.textContent = 'Stop Recording';
  recordBtn.classList.add('recording');
  recordStatus.textContent = 'Recording in progress...';
  submitBtn.disabled = true;

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

  if (processor) processor.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (audioContext) audioContext.close();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());

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


function formatDuration(value) {
  if (value == null) return '-';
  return Number(value).toFixed(2);
}

function renderResults(data) {
  setRequestId(data.request_id);
  resultsSection.hidden = false;
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
      <td>s1=${formatDuration(t.core_durations.syll1)} / s2=${formatDuration(t.core_durations.syll2)}</td>
      <td>${t.status}</td>
      <td>${t.feedback}</td>
    `;
    targetsTableBody.appendChild(tr);
  }
}

async function submitAnalysis() {
  if (!recordingBlob) return;
  const form = new FormData();
  form.append('paragraph_id', String(selectedParagraphId));
  form.append('audio_wav', recordingBlob, 'recording.wav');
  const resp = await fetch('/api/analyze', { method: 'POST', body: form });
  const data = await resp.json();
  if (!resp.ok) {
    alert(data.error || 'Analysis failed');
    return;
  }
  renderResults(data);
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

loadParagraphs();
