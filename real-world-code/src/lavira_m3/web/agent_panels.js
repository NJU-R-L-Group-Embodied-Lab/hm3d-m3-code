import {
  byId,
  escapeHtml,
  imageSource,
  palette,
  seconds,
  stamp,
} from "./dashboard_api.js";

export class AgentPanels {
  constructor(config) {
    this.config = config;
    this.names = Object.keys(config.agents);
    this.frame = null;
    this.rawAgent = null;
    byId("robotStrip").style.setProperty("--agent-count", this.names.length);
    byId("fleetStrip").style.setProperty(
      "--fleet-count",
      this.names.length + 3,
    );
    byId("fleetStrip").innerHTML =
      this.names
        .map(
          (name, index) =>
            `<div class="fleet-card" data-agent="${name}" style="--robot-color:${palette[index % palette.length]}"><span>${name}</span><b id="pose-${name}">—</b></div>`,
        )
        .join("") +
      '<div class="fleet-card camera-health"><span>Cameras</span><b id="cameraHealth">Waiting</b></div><div class="fleet-card metric"><span>Occupied</span><b id="occ">0</b></div><div class="fleet-card metric"><span>Paths</span><b id="npath">0</b></div>';
    byId("robotStrip").innerHTML = this.names
      .map(
        (
          name,
          index,
        ) => `<article class="robot-card" data-agent="${name}" style="--robot-color:${palette[index % palette.length]}">
      <div class="readonly-badge">${name} · observation & decision</div>
      <div class="robot-card-head"><strong>${name}</strong><span id="phase-${name}">Waiting for data</span></div>
      <div class="robot-visuals"><div class="robot-camera" data-label="VA image"><canvas id="va-${name}" hidden></canvas><span id="vaEmpty-${name}">Waiting for VA image</span></div><div class="robot-camera" data-label="Front camera"><img id="front-${name}" hidden alt="${name} front camera"><span id="frontEmpty-${name}">Waiting for camera</span><small class="frame-stamp" id="stamp-${name}"></small></div></div>
      <div class="raw-actions"><button data-json="${name}">▸ Raw JSON</button><button data-images="${name}">▸ Raw images</button></div>
      <div class="robot-card-foot" id="nav-${name}">Waiting for nav status</div></article>`,
      )
      .join("");
    byId("robotStrip")
      .querySelectorAll("[data-json]")
      .forEach((button) => {
        button.onclick = () => this.json(button.dataset.json);
      });
    byId("robotStrip")
      .querySelectorAll("[data-images]")
      .forEach((button) => {
        button.onclick = () => this.images(button.dataset.images);
      });
    this.vaSources = {};
    this.vaTokens = {};
    this.frontTokens = {};
    this.rawStamp = 0;
  }

  cameraUrl(agent, camera) {
    if (this.replay) return imageSource(this.frame.images?.[agent]?.[camera]);
    return this.frame.cameras?.[agent]?.[camera]?.online
      ? `/v1/image/${agent}/${camera}?t=${this.frame.t}`
      : "";
  }

  async drawVA(agent, trace) {
    const source = imageSource(trace.va_image || trace.front_image);
    const signature =
      source + JSON.stringify([trace.va_bbox, trace.va_pixel, trace.va]);
    if (this.vaSources[agent] === signature) return;
    this.vaSources[agent] = signature;
    const canvas = byId(`va-${agent}`),
      empty = byId(`vaEmpty-${agent}`);
    canvas.hidden = true;
    empty.hidden = false;
    const token = Symbol();
    this.vaTokens[agent] = token;
    if (!source) return;
    const image = new Image();
    image.onload = () => {
      if (this.vaTokens[agent] !== token) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0);
      const bbox = trace.va_bbox || trace.va?.bbox_2d;
      if (
        Array.isArray(bbox) &&
        bbox.length === 4 &&
        bbox.every(Number.isFinite)
      ) {
        // VA boxes use the model's normalized 0..999 coordinate contract.
        const scaleX = canvas.width / 999,
          scaleY = canvas.height / 999;
        ctx.strokeStyle = "#27f549";
        ctx.lineWidth = Math.max(2, canvas.width / 200);
        ctx.strokeRect(
          bbox[0] * scaleX,
          bbox[1] * scaleY,
          (bbox[2] - bbox[0]) * scaleX,
          (bbox[3] - bbox[1]) * scaleY,
        );
      }
      const point = trace.va_pixel;
      if (Array.isArray(point)) {
        const original = trace.image_size || [canvas.width, canvas.height];
        ctx.fillStyle = "#facc15";
        ctx.beginPath();
        ctx.arc(
          (point[0] * canvas.width) / original[0],
          (point[1] * canvas.height) / original[1],
          5,
          0,
          Math.PI * 2,
        );
        ctx.fill();
      }
      canvas.hidden = false;
      empty.hidden = true;
    };
    image.onerror = () => {
      empty.textContent = "Failed to load image";
    };
    image.src = source;
  }

  render(frame, replay = false) {
    this.frame = frame;
    this.replay = replay;
    let fresh = 0,
      total = 0;
    this.names.forEach((agent) => {
      const trace = frame.board.traces?.[agent] || {};
      const pose = frame.status.sources?.[agent]?.pose;
      const status = byId(`pose-${agent}`);
      status.textContent = pose?.position
        ? `${pose.live ? "live" : "stale"} ${pose.position
            .slice(0, 2)
            .map((v) => v.toFixed(2))
            .join(", ")}`
        : "Waiting for pose";
      status.className = pose?.live ? "ok" : "stale";
      const decision = trace.la?.action || trace.nav?.reason || "Waiting for decision";
      byId(`phase-${agent}`).textContent =
        ` ${decision} · step ${trace.step ?? "—"} · LA ${seconds(trace.la_time_ms)} · VA ${seconds(trace.va_time_ms)} · RA ${seconds(trace.ra_time_ms)}`;
      byId(`nav-${agent}`).textContent =
        `Motion: ${trace.nav?.state || trace.nav?.status || trace.nav?.reason || "waiting for data"}`;
      this.drawVA(agent, trace);
      const camera = this.config.agents[agent].directions.includes("front")
        ? "front"
        : this.config.agents[agent].directions[0];
      const image = byId(`front-${agent}`),
        empty = byId(`frontEmpty-${agent}`);
      const source = this.cameraUrl(agent, camera);
      const token = Symbol();
      this.frontTokens[agent] = token;
      if (source) {
        image.onload = () => {
          if (this.frontTokens[agent] === token) {
            image.hidden = false;
            empty.hidden = true;
          }
        };
        image.onerror = () => {
          if (this.frontTokens[agent] === token) {
            image.hidden = true;
            empty.hidden = false;
            empty.textContent = "Waiting for a fresh camera frame";
          }
        };
        image.src = source;
      } else {
        image.hidden = true;
        empty.hidden = false;
      }
      byId(`stamp-${agent}`).textContent = source ? stamp(frame.t) : "";
      for (const item of Object.values(frame.cameras?.[agent] || {})) {
        total++;
        if (item.online) fresh++;
      }
    });
    byId("cameraHealth").textContent = `${fresh} / ${total} live`;
    byId("cameraHealth").className = fresh === total && total ? "" : "bad";
    byId("occ").textContent = String(frame.status.occupied || 0);
    byId("npath").textContent = this.names
      .map(
        (name) =>
          (
            frame.scene.plans?.[name]?.path ||
            frame.board.traces?.[name]?.va_path ||
            []
          ).length,
      )
      .join("/");
    if (byId("rawJsonDialog").open) this.json(this.rawAgent, false);
    if (byId("rawImageDialog").open && frame.t !== this.rawStamp)
      this.images(this.rawAgent, false);
  }

  json(agent, open = true) {
    this.rawAgent = agent;
    byId("rawJsonTitle").textContent = `${agent} · raw JSON`;
    byId("rawJsonContent").textContent = JSON.stringify(
      {
        trace: this.frame?.board.traces?.[agent],
        navigation: this.frame?.telemetry?.robots?.[agent],
      },
      null,
      2,
    );
    if (open) byId("rawJsonDialog").showModal();
  }

  images(agent, open = true) {
    this.rawAgent = agent;
    this.rawStamp = this.frame.t;
    byId("rawImageTitle").textContent = `${agent} · all cameras`;
    byId("rawImageContent").innerHTML = this.config.agents[agent].directions
      .map((camera) => {
        const source = this.cameraUrl(agent, camera);
        return `<figure><figcaption>${escapeHtml(camera)} · ${stamp(this.frame.t)}</figcaption>${source ? `<img src="${escapeHtml(source)}" alt="${agent} ${escapeHtml(camera)}">` : "<p>Waiting for a fresh image</p>"}</figure>`;
      })
      .join("");
    if (open) byId("rawImageDialog").showModal();
  }
}
