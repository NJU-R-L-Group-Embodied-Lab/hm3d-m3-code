import { byId, palette, request } from "./dashboard_api.js";

export function yawOf(pose) {
  const [x, y, z, w] = pose?.orientation || [0, 0, 0, 1];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

/** Canvas geometry and interaction; navigation commands stay in the ROS workflow. */
export class MapView {
  constructor(config) {
    this.config = config;
    this.names = Object.keys(config.agents);
    this.canvas = byId("c");
    this.ctx = this.canvas.getContext("2d");
    this.view = { scale: 25, ox: 0, oy: 0, ready: false };
    this.layers = new Set(this.names);
    this.planning = null;
    this.planAt = 0;
    this.planBusy = false;
    this.drag = null;
    this.canvas.addEventListener("pointerdown", (event) => {
      this.drag = this.local(event);
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!this.drag) return;
      const next = this.local(event);
      this.view.ox += next[0] - this.drag[0];
      this.view.oy += next[1] - this.drag[1];
      this.drag = next;
      this.draw();
    });
    const release = () => {
      this.drag = null;
    };
    this.canvas.addEventListener("pointerup", release);
    this.canvas.addEventListener("pointercancel", release);
    this.canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        this.zoom(Math.exp(-event.deltaY * 0.002), this.local(event));
      },
      { passive: false },
    );
    byId("zoomIn").onclick = () => this.zoom(1.3);
    byId("zoomOut").onclick = () => this.zoom(1 / 1.3);
    byId("zoomFit").onclick = () => this.fit();
    byId("mapView").onchange = () => {
      this.planning = null;
      this.planAt = 0;
      this.update(this.frame, this.replay);
    };
    for (const id of [
      "showFused",
      "showPlan",
      "showActual",
      "showLaviraVa",
      "showPrediction",
    ])
      byId(id).onchange = () => this.draw();
    new ResizeObserver(() => this.resize()).observe(this.canvas);
  }

  local(event) {
    const rect = this.canvas.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) * this.canvas.clientWidth) / rect.width,
      ((event.clientY - rect.top) * this.canvas.clientHeight) / rect.height,
    ];
  }

  point(x, y) {
    return [
      this.view.ox + x * this.view.scale,
      this.view.oy - y * this.view.scale,
    ];
  }

  resize() {
    const ratio = devicePixelRatio || 1;
    const width = this.canvas.clientWidth,
      height = this.canvas.clientHeight;
    if (this.size && this.view.ready) {
      this.view.ox += (width - this.size[0]) / 2;
      this.view.oy += (height - this.size[1]) / 2;
    }
    this.size = [width, height];
    this.canvas.width = width * ratio;
    this.canvas.height = height * ratio;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (!this.view.ready) this.fit();
    this.draw();
  }

  fit() {
    const box = this.frame?.scene.aabb || [-5, -5, 5, 5];
    this.view.scale = Math.max(
      4,
      Math.min(
        400,
        Math.min(
          this.canvas.clientWidth / Math.max(1, box[2] - box[0]),
          this.canvas.clientHeight / Math.max(1, box[3] - box[1]),
        ) * 0.85,
      ),
    );
    this.view.ox =
      this.canvas.clientWidth / 2 - ((box[0] + box[2]) / 2) * this.view.scale;
    this.view.oy =
      this.canvas.clientHeight / 2 + ((box[1] + box[3]) / 2) * this.view.scale;
    this.view.ready = Boolean(this.frame?.scene.aabb);
    this.draw();
  }

  zoom(
    factor,
    center = [this.canvas.clientWidth / 2, this.canvas.clientHeight / 2],
  ) {
    const before = this.view.scale;
    this.view.scale = Math.max(4, Math.min(400, before * factor));
    const scale = this.view.scale / before;
    this.view.ox = center[0] - (center[0] - this.view.ox) * scale;
    this.view.oy = center[1] - (center[1] - this.view.oy) * scale;
    this.view.ready = true;
    this.draw();
  }

  async update(frame, replay = false) {
    if (!frame) return;
    this.frame = frame;
    this.replay = replay;
    if (!this.view.ready) this.fit();
    const selected = byId("mapView").value;
    if (selected.startsWith("plan:")) {
      const agent = selected.slice(5);
      if (replay) this.planning = frame.planning?.[agent] || null;
      else if (!this.planBusy && Date.now() - this.planAt > 2000) {
        this.planBusy = true;
        this.planAt = Date.now();
        try {
          const result = await request(
            `/v1/planning-grid?robot=${encodeURIComponent(agent)}`,
          );
          if (!this.replay && byId("mapView").value === selected)
            this.planning = result;
        } catch (error) {
          byId("hint").textContent = error.message;
        } finally {
          this.planBusy = false;
        }
      }
    }
    this.draw();
  }

  cells(cells, color, resolution, origin = [0, 0]) {
    this.ctx.fillStyle = color;
    const size = Math.max(1, resolution * this.view.scale);
    for (const [x, y] of cells || []) {
      const point = this.point(
        origin[0] + x * resolution,
        origin[1] + y * resolution,
      );
      this.ctx.fillRect(point[0], point[1] - size, size, size);
    }
  }

  path(points, color, dashed = false) {
    if (!points?.length) return;
    const ctx = this.ctx;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      const point = this.point(x, y);
      if (index) ctx.lineTo(...point);
      else ctx.moveTo(...point);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
    byId("zoomLvl").textContent = `${Math.round(this.view.scale)} px/m`;
    if (!this.frame) return;
    const { scene, status, board } = this.frame;
    const res = scene.resolution || 0.1;
    const view = byId("mapView").value;
    if (view.startsWith("plan:") && this.planning) {
      this.cells(this.planning.hard || this.planning.inflate, "#ccd7b7", res);
      this.cells(this.planning.soft, "#e6edcf", res);
      this.cells(this.planning.occupied, "#64765a", res);
    } else if (view === "fused")
      this.cells(scene.fused, "#a9be73", res, scene.origin);
    else {
      this.names.forEach((agent, index) => {
        if (this.layers.has(agent))
          this.cells(
            scene.sources?.[agent],
            palette[index % palette.length],
            res,
            scene.origin,
          );
      });
      if (byId("showFused").checked)
        this.cells(scene.overlap, "#c6bfd2", res, scene.origin);
    }
    this.names.forEach((agent, index) => {
      const color = palette[index % palette.length];
      const telemetry = this.frame.telemetry?.robots?.[agent] || {};
      const trace = board.traces?.[agent] || {};
      if (byId("showActual").checked)
        this.path(scene.trails?.[agent], "#be7ad0", true);
      if (byId("showPlan").checked)
        this.path(
          telemetry.live && telemetry.path?.length
            ? telemetry.path
            : scene.plans?.[agent]?.path || trace.va_path,
          color,
        );
      if (byId("showPrediction").checked && telemetry.live) {
        this.path(telemetry.reference, "#539faf", true);
        this.path(telemetry.prediction, "#e09b35");
      }
      if (byId("showLaviraVa").checked && Array.isArray(trace.va_map)) {
        const point = this.point(...trace.va_map);
        ctx.fillStyle = "#facc15";
        ctx.beginPath();
        ctx.arc(...point, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      const pose = status.sources?.[agent]?.pose;
      if (!pose?.position) return;
      const point = this.point(...pose.position);
      const [length, width] = this.config.agents[agent].footprint;
      ctx.save();
      ctx.translate(...point);
      ctx.rotate(-yawOf(pose));
      ctx.strokeStyle = color;
      ctx.lineWidth = byId("selectedAgent").value === agent ? 3 : 1.5;
      ctx.setLineDash(pose.live ? [] : [3, 3]);
      ctx.strokeRect(
        (-length * this.view.scale) / 2,
        (-width * this.view.scale) / 2,
        length * this.view.scale,
        width * this.view.scale,
      );
      ctx.setLineDash([]);
      ctx.strokeStyle = "#d55b55";
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(16, 0);
      ctx.stroke();
      ctx.strokeStyle = "#498f63";
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, -16);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = "#4f6058";
      ctx.font = "10px sans-serif";
      ctx.fillText(agent, point[0] + 8, point[1] - 8);
    });
    (board.found_xy || []).forEach((point, index) => {
      if (!point) return;
      const pixel = this.point(...point);
      ctx.fillStyle = "#db4137";
      ctx.beginPath();
      ctx.arc(...pixel, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(`Goal ${index + 1}`, pixel[0] + 8, pixel[1]);
    });
    byId("mapStatus").textContent = this.replay
      ? "Replay"
      : status.ready
        ? ""
        : "Waiting for shared map";
  }
}
