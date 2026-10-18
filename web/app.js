import {
  Room,
  RoomEvent,
  Track,
} from "https://cdn.jsdelivr.net/npm/livekit-client@2.22.3/dist/livekit-client.esm.mjs";

// Must match agent.py.
const UI_TOPIC = "stt-fusion";
const PIPELINE_ATTRIBUTE = "stt_fusion.pipeline";
// Published by livekit-agents itself.
const TRANSCRIPTION_TOPIC = "lk.transcription";
const STATE_ATTRIBUTE = "lk.agent.state";

const STATE_LABELS = {
  initializing: "Agent starting…",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};
const PROVIDERS = { deepgram: "Deepgram", openai: "OpenAI", groq: "Groq" };

const $ = (id) => document.getElementById(id);
const timeline = $("timeline");

let room = null;
let agentTimer = null;
let liveCard = null; // card for the turn in progress
let lastTurnCard = null; // most recent finished turn, which the next latency report belongs to
let liveSegments = []; // race results received since the last finished turn
const replies = new Map(); // transcription segment id -> reply text element
const scores = new Map(); // engine -> { turns, matched, latency }

$("toggle").addEventListener("click", () => (room ? stop() : start()));
if (new URLSearchParams(location.search).has("autostart")) start();

// ---------------------------------------------------------------- connection

async function start() {
  hideError();
  setStatus("connecting", "Connecting…");
  $("toggle").disabled = true;
  try {
    const params = new URLSearchParams(location.search);
    const response = await fetch(`/token${params.has("room") ? `?room=${params.get("room")}` : ""}`);
    if (!response.ok) throw new Error(`token request failed (${response.status})`);
    const { url, token } = await response.json();

    room = new Room();
    wire(room);
    await room.connect(url, token);
    await room.localParticipant.setMicrophoneEnabled(true);
    await room.startAudio();

    $("toggle").textContent = "End conversation";
    $("you").hidden = false;
    $("empty-title").textContent = "Connected. Start talking once the agent is listening.";
    setStatus("waiting", "Waiting for the agent…");
    room.remoteParticipants.forEach(onAgentUpdate); // the agent may already be in the room
    agentTimer = setTimeout(() => {
      if (!agent()) showError("No agent joined the room. Is `uv run agent.py dev` running?");
    }, 15000);
  } catch (err) {
    showError(
      `Couldn't connect (${err.message}). Is \`livekit-server --dev\` running, and \`uv run web/server.py\`?`,
    );
    await stop();
  } finally {
    $("toggle").disabled = false;
  }
}

async function stop() {
  clearTimeout(agentTimer);
  const current = room;
  room = null;
  if (current) await current.disconnect();
  $("toggle").textContent = "Start conversation";
  $("you").hidden = true;
  $("empty-title").innerHTML = "Click <strong>Start conversation</strong>, allow the microphone, and talk.";
  setStatus("idle", "Not connected");
}

function wire(room) {
  room
    .on(RoomEvent.TrackSubscribed, (track) => {
      if (track.kind === Track.Kind.Audio) $("audio").appendChild(track.attach());
    })
    .on(RoomEvent.TrackUnsubscribed, (track) => track.detach().forEach((el) => el.remove()))
    .on(RoomEvent.ParticipantConnected, onAgentUpdate)
    .on(RoomEvent.ParticipantAttributesChanged, (_changed, participant) => onAgentUpdate(participant))
    .on(RoomEvent.ParticipantDisconnected, (participant) => {
      if (participant.isAgent) setStatus("waiting", "Agent left the room");
    })
    .on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
      $("you").classList.toggle("speaking", speakers.some((p) => p.isLocal));
    })
    .on(RoomEvent.Disconnected, () => {
      if (room) stop();
    });

  room.registerTextStreamHandler(UI_TOPIC, async (reader) => {
    onFusionEvent(JSON.parse(await reader.readAll()));
  });
  room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, onTranscription);
}

function agent() {
  return [...(room?.remoteParticipants.values() ?? [])].find((p) => p.isAgent);
}

function onAgentUpdate(participant) {
  if (!participant.isAgent) return;
  clearTimeout(agentTimer);
  hideError();
  const state = participant.attributes[STATE_ATTRIBUTE];
  if (state) setStatus(state, STATE_LABELS[state] ?? state);
  const pipeline = participant.attributes[PIPELINE_ATTRIBUTE];
  if (pipeline) renderPipeline(JSON.parse(pipeline));
}

// ---------------------------------------------------------------- agent events

function onFusionEvent(event) {
  $("empty").hidden = true;
  if (event.type === "segment") {
    liveSegments.push(event.segment);
    renderLiveTurn();
  } else if (event.type === "turn") {
    const done = new Set(event.segments.map((s) => s.index));
    liveSegments = liveSegments.filter((s) => !done.has(s.index));
    const card = liveCard ?? appendToTimeline(document.createElement("article"));
    liveCard = null;
    lastTurnCard = card;
    renderFinishedTurn(card, event);
    updateScores(event);
    if (liveSegments.length) renderLiveTurn();
  } else if (event.type === "latency" && lastTurnCard) {
    renderLatency(lastTurnCard, event);
  }
}

async function onTranscription(reader, participant) {
  // The user's own words arrive through UI_TOPIC with all the race details.
  if (participant.identity === room?.localParticipant.identity) return;
  $("empty").hidden = true;
  const attributes = reader.info.attributes ?? {};
  const id = attributes["lk.segment_id"] ?? reader.info.id;
  let text = replies.get(id);
  if (!text) {
    text = el("p", "reply-text");
    appendToTimeline(el("div", "reply", [el("span", "who", "Agent"), text]));
    replies.set(id, text);
  }
  if (attributes["lk.transcription_final"] === "true") {
    text.textContent = await reader.readAll(); // a whole-segment update
  } else {
    for await (const chunk of reader) {
      text.textContent += chunk; // a delta stream, word by word as it's spoken
      followTimeline();
    }
  }
  followTimeline();
}

// ---------------------------------------------------------------- rendering

function renderLiveTurn() {
  if (!liveCard) liveCard = appendToTimeline(document.createElement("article"));
  const audio = liveSegments.reduce((sum, s) => sum + s.audio_s, 0);
  liveCard.className = "turn live";
  liveCard.replaceChildren(
    el("header", "turn-head", [
      el("span", "turn-title", "You"),
      el("span", "meta", `${plural(liveSegments.length, "segment")} · ${audio.toFixed(1)} s`),
    ]),
    engineRows(mergeSegments(liveSegments), null),
    el("div", "fusing", [el("span", "spinner"), "Waiting for the end of your turn, then fusing…"]),
  );
  followTimeline();
}

function renderFinishedTurn(card, turn) {
  const { fusion } = turn;
  const audio = turn.segments.reduce((sum, s) => sum + s.audio_s, 0);
  const tags = [];
  if (fusion.source === "fallback") tags.push(el("span", "tag fallback", "fallback"));
  else if (!fusion.text.trim()) tags.push(el("span", "tag noise", "noise"));
  else if (fusion.source === "skipped") tags.push(el("span", "tag agreed", "all agreed"));
  else if (turn.changed) tags.push(el("span", "tag changed", "corrected"));

  const fused = fusion.text.trim()
    ? el("span", "text", fusion.text)
    : el("span", "text muted", "(no speech, no reply)");
  const notes = [];
  if (fusion.rationale) notes.push(el("p", "why", fusion.rationale));
  if (fusion.error) notes.push(el("p", "why warn", `Fallback reason: ${fusion.error}`));

  card.className = "turn";
  card.replaceChildren(
    el("header", "turn-head", [
      el("span", "turn-title", `Turn ${turn.turn}`),
      el("span", "meta", `${plural(turn.segments.length, "segment")} · ${audio.toFixed(1)} s`),
      ...tags,
    ]),
    engineRows(turn.candidates, fusion.text),
    el("div", "row consensus", [
      el("span", "label", "Consensus"),
      el("span", "latency"),
      el("span", "text", turn.heard),
    ]),
    el("div", "row fused", [
      el("span", "label", "Fused"),
      el("span", "latency", fusion.source === "skipped" ? "skipped" : ms(fusion.latency_s)),
      fused,
    ]),
    ...notes,
  );
  followTimeline();
}

/** LiveKit's timings for the turn: how long after the user stopped talking the agent spoke. */
function renderLatency(card, latency) {
  const stages = [
    ["transcription_delay", "transcript"],
    ["on_user_turn_completed_delay", "fusion"],
    ["llm_node_ttft", "reply first token"],
    ["tts_node_ttfb", "first audio"],
  ]
    .filter(([key]) => latency[key] != null)
    .map(([key, label]) => `${label} ${seconds(latency[key])}`);
  card.querySelector(".timing")?.remove();
  card.append(
    el("p", "timing", [
      el("strong", null, `Agent replied ${seconds(latency.e2e_latency)} after you stopped`),
      stages.length ? ` · ${stages.join(" · ")}` : "",
    ]),
  );
}

function engineRows(candidates, reference) {
  return el(
    "div",
    "engines",
    candidates.map((c) => {
      const [provider, model] = splitEngine(c.engine);
      let text;
      if (c.status === "ok") {
        text = el("span", "text");
        text.append(reference ? highlightDifferences(c.text, reference) : c.text);
      } else if (c.status === "empty") {
        text = el("span", "text muted", "heard nothing");
      } else {
        text = el("span", "text failed", `${c.status}: ${c.error ?? ""}`);
      }
      return el("div", "row", [
        el("span", "label", [el("strong", null, provider), el("small", null, model)]),
        el("span", "latency", ms(c.latency_s)),
        text,
      ]);
    }),
  );
}

function renderPipeline({ engines, fusion_llm, reply_llm, tts }) {
  const chip = (title, detail) => el("span", "chip", [el("strong", null, title), " ", detail]);
  const arrow = () => el("span", "arrow", "→");
  const nav = $("pipeline");
  nav.replaceChildren(
    el(
      "span",
      "race",
      engines.map((engine) => chip(...splitEngine(engine))),
    ),
    arrow(),
    chip("Fusion", fusion_llm),
    arrow(),
    chip("Reply", reply_llm),
    arrow(),
    chip("Voice", tts),
  );
  nav.hidden = false;
}

function updateScores(turn) {
  const target = normalize(turn.fusion.text);
  if (!target) return; // noise turns say nothing about accuracy
  for (const c of turn.candidates) {
    const score = scores.get(c.engine) ?? { turns: 0, matched: 0, answered: 0, latency: 0 };
    score.turns += 1;
    if (c.status === "ok" || c.status === "empty") {
      score.answered += 1;
      score.latency += c.latency_s; // failures would skew the average towards the deadline
    }
    if (c.status === "ok" && normalize(c.text) === target) score.matched += 1;
    scores.set(c.engine, score);
  }
  $("scoreboard").replaceChildren(
    ...[...scores].map(([engine, s]) => {
      const [provider, model] = splitEngine(engine);
      return el("tr", null, [
        el("td", null, [el("strong", null, provider), " ", el("small", null, model)]),
        el("td", null, `${s.matched}/${s.turns}`),
        el("td", null, [
          s.answered ? ms(s.latency / s.answered) : "–",
          s.answered < s.turns ? el("small", "failures", `${s.turns - s.answered} failed`) : "",
        ]),
      ]);
    }),
  );
}

// ---------------------------------------------------------------- helpers

/** Marks the words of `text` that are not part of its longest common run with `reference`. */
function highlightDifferences(text, reference) {
  const words = text.split(/\s+/).filter(Boolean);
  const a = words.map(normalizeWord);
  const b = reference.split(/\s+/).filter(Boolean).map(normalizeWord);
  const lcs = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const kept = new Array(a.length).fill(false);
  for (let i = 0, j = 0; i < a.length && j < b.length; ) {
    if (a[i] === b[j]) {
      kept[i++] = true;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) i++;
    else j++;
  }
  const fragment = document.createDocumentFragment();
  words.forEach((word, i) => {
    if (i) fragment.append(" ");
    fragment.append(kept[i] || !a[i] ? word : el("mark", "diff", word));
  });
  return fragment;
}

function mergeSegments(segments) {
  const byEngine = new Map();
  for (const segment of segments) {
    for (const c of segment.candidates) {
      const merged = byEngine.get(c.engine);
      if (!merged) {
        byEngine.set(c.engine, { ...c });
      } else {
        if (c.status === "ok") {
          merged.text = [merged.text, c.text].filter(Boolean).join(" ");
          merged.status = "ok";
        }
        merged.latency_s = Math.max(merged.latency_s, c.latency_s);
      }
    }
  }
  return [...byEngine.values()];
}

function splitEngine(engine) {
  const [provider, ...model] = engine.split("/");
  return [PROVIDERS[provider] ?? provider, model.join("/")];
}

function normalizeWord(word) {
  return word.toLowerCase().replace(/[^\p{L}\p{N}']/gu, "");
}

function normalize(text) {
  return text.split(/\s+/).map(normalizeWord).filter(Boolean).join(" ");
}

function ms(seconds) {
  return seconds == null ? "" : `${Math.round(seconds * 1000)} ms`;
}

function seconds(value) {
  return `${value.toFixed(1)} s`;
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function el(tag, className, children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (children != null) node.append(...(Array.isArray(children) ? children : [children]));
  return node;
}

function appendToTimeline(node) {
  timeline.append(node);
  followTimeline();
  return node;
}

function followTimeline() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function setStatus(state, text) {
  $("status").dataset.state = state;
  $("status-text").textContent = text;
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function hideError() {
  $("error").hidden = true;
}
