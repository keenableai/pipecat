// Minimal, dependency-free WebRTC client for the Pipecat SmallWebRTC transport.
//
// It performs the offer/answer handshake against /api/offer, plays the bot's
// audio, opens a data channel, and keeps the connection "alive" by sending a
// "ping" string every second (the server treats a peer with no recent ping as
// disconnected and stops delivering messages). Incoming RTVI "server-message"
// payloads of type "fact-check" / "fact-check-stats" drive the UI.

const connectBtn = document.getElementById("connect");
const statusEl = document.getElementById("status");
const claimsEl = document.getElementById("claims");
const emptyEl = document.getElementById("empty");
const audioEl = document.getElementById("bot-audio");

let pc = null;
let dc = null;
let pingTimer = null;
let pcId = null;

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status " + cls;
}

async function connect() {
  connectBtn.disabled = true;
  setStatus("connecting", "connecting");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    // Send our mic; the first transceiver is the bot's audio (sendrecv).
    stream.getAudioTracks().forEach((t) => pc.addTrack(t, stream));

    pc.ontrack = (e) => {
      audioEl.srcObject = e.streams[0];
    };

    // Data channel carries RTVI / server messages both ways.
    dc = pc.createDataChannel("fact-check");
    dc.onopen = () => {
      setStatus("live", "live");
      connectBtn.textContent = "Disconnect";
      connectBtn.disabled = false;
      // Keepalive: the server drops a peer with no ping in the last 3s.
      pingTimer = setInterval(() => {
        if (dc && dc.readyState === "open") dc.send("ping");
      }, 1000);
    };
    dc.onmessage = onMessage;

    pc.onconnectionstatechange = () => {
      if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
        teardown();
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIceGathering(pc);

    const res = await fetch("/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
        pc_id: pcId,
      }),
    });
    const answer = await res.json();
    pcId = answer.pc_id;
    await pc.setRemoteDescription(answer);
  } catch (err) {
    console.error(err);
    setStatus("error", "error");
    connectBtn.disabled = false;
    teardown();
  }
}

function waitForIceGathering(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    // Safety net in case the event never fires.
    setTimeout(resolve, 2000);
  });
}

function onMessage(event) {
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return; // ignore "ping"/non-JSON
  }
  if (msg.type !== "server-message" || !msg.data) return;

  const payload = msg.data;
  if (payload.type === "fact-check") renderClaim(payload);
  else if (payload.type === "fact-check-stats") renderStats(payload);
}

function renderClaim(c) {
  if (emptyEl) emptyEl.style.display = "none";

  const li = document.createElement("li");
  li.className = "card " + (c.color || "amber");

  const top = document.createElement("div");
  top.className = "card-top";
  const badge = document.createElement("span");
  badge.className = "badge " + (c.color || "amber");
  badge.textContent = c.verdict || "unverified";
  top.appendChild(badge);

  const claim = document.createElement("span");
  claim.className = "claim";
  claim.textContent = c.claim || "";

  const expl = document.createElement("p");
  expl.className = "explanation";
  expl.textContent = c.explanation || "";

  li.appendChild(top);
  li.appendChild(claim);
  if (c.explanation) li.appendChild(expl);

  if (Array.isArray(c.sources) && c.sources.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    for (const url of c.sources.slice(0, 4)) {
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener";
      try {
        a.textContent = new URL(url).hostname.replace(/^www\./, "");
      } catch {
        a.textContent = url;
      }
      sources.appendChild(a);
    }
    li.appendChild(sources);
  }

  // Newest claim on top.
  claimsEl.prepend(li);
}

function renderStats(s) {
  document.getElementById("m-checks").textContent = s.checks ?? 0;
  document.getElementById("m-searches").textContent = s.searches ?? 0;
  document.getElementById("m-llm").textContent = s.llm_calls ?? 0;
  document.getElementById("m-skipped").textContent = s.skipped ?? 0;
  document.getElementById("m-cost").textContent = "$" + (s.est_cost_usd ?? 0).toFixed(4);
}

function teardown() {
  if (pingTimer) clearInterval(pingTimer);
  pingTimer = null;
  if (dc) dc.close();
  if (pc) pc.close();
  dc = null;
  pc = null;
  pcId = null;
  setStatus("disconnected", "idle");
  connectBtn.textContent = "Connect";
  connectBtn.disabled = false;
}

connectBtn.addEventListener("click", () => {
  if (pc) teardown();
  else connect();
});
