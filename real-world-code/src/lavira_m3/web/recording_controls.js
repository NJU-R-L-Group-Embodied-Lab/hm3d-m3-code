import { byId, request, post, download, stamp } from "./dashboard_api.js";
import { zipStore } from "./zip_store.js";

export class RecordingControls {
  constructor(render, refresh) {
    this.render = render;
    this.refresh = refresh;
    this.replay = null;
    this.current = null;
    this.recording = {};
    this.items = [];
    this.busy = false;
    this.generation = 0;
    this.statusAt = 0;
    byId("recordStart").onclick = () =>
      this.action(async () => {
        this.recording = await post("/v1/record/start");
        this.statusAt = 0;
      });
    byId("recordStop").onclick = () =>
      this.action(async () => {
        this.recording = await post("/v1/record/stop");
        await this.list();
        byId("recordingSelect").value = this.recording.id;
        this.controls();
      });
    byId("replayStart").onclick = () =>
      this.action(async () => {
        const item = this.items.find(
          (entry) => entry.id === byId("recordingSelect").value,
        );
        if (!item?.total) throw new Error("Select a recording that contains frames");
        this.generation++;
        this.replay = { item, index: 0 };
        await this.seek(0);
      });
    byId("replayStop").onclick = () => {
      this.generation++;
      this.replay = null;
      this.controls();
      this.refresh();
    };
    byId("replaySeek").oninput = (event) =>
      this.action(() => this.seek(Number(event.target.value)));
    byId("recordingSelect").onchange = () => this.controls();
    byId("replayDownloadAll").onclick = () => {
      const id = this.replay?.item.id || byId("recordingSelect").value;
      if (id) {
        const link = document.createElement("a");
        link.href = `/v1/record/download?id=${encodeURIComponent(id)}`;
        link.download = `${id}.zip`;
        link.click();
      }
    };
    byId("replayDownloadFrame").onclick = () =>
      this.action(() => this.downloadFrame());
  }

  async action(fn) {
    try {
      await fn();
      this.controls();
    } catch (error) {
      byId("recordStatus").textContent = error.message;
    }
  }

  async list() {
    const data = await request("/v1/recordings");
    this.items = data.recordings;
    const select = byId("recordingSelect"),
      value = select.value;
    select.replaceChildren(new Option("Select a recording", ""));
    for (const item of this.items)
      select.add(new Option(`${item.id} · ${item.total} frames`, item.id));
    if (this.items.some((item) => item.id === value)) select.value = value;
  }

  async seek(index) {
    if (!this.replay) return;
    const generation = ++this.generation,
      replay = this.replay;
    const safeIndex = Math.max(0, Math.min(replay.item.total - 1, index));
    const frame = await request(
      `/v1/record/frame?id=${encodeURIComponent(replay.item.id)}&index=${safeIndex}`,
    );
    if (generation !== this.generation || this.replay !== replay) return;
    replay.index = safeIndex;
    this.current = frame;
    this.render(frame, true);
    this.controls();
  }

  async tick() {
    if (this.busy) return;
    this.busy = true;
    try {
      if (Date.now() - this.statusAt > 2000) {
        this.recording = await request("/v1/record/status");
        this.statusAt = Date.now();
        if (!this.items.length) await this.list();
      }
      if (this.replay && this.replay.index < this.replay.item.total - 1)
        await this.seek(this.replay.index + 1);
      this.controls();
    } catch (error) {
      byId("recordStatus").textContent = error.message;
    } finally {
      this.busy = false;
    }
  }

  controls() {
    const recording = this.recording.active,
      replay = Boolean(this.replay);
    byId("recordStart").disabled = recording || replay;
    byId("recordStop").disabled = !recording;
    byId("replayStart").disabled = recording || replay;
    byId("replayStop").disabled = !replay;
    byId("replayDownloadAll").disabled = !(
      this.replay?.item.id || byId("recordingSelect").value
    );
    byId("replayProgress").hidden = !replay;
    byId("replayProgress").classList.toggle("on", replay);
    byId("replaySeek").disabled = !replay;
    byId("replaySeek").max = String((this.replay?.item.total || 1) - 1);
    byId("replaySeek").value = String(this.replay?.index || 0);
    byId("replayTime").textContent = replay
      ? `${this.replay.index + 1} / ${this.replay.item.total}`
      : "0 / 0";
    const status = byId("recordStatus");
    status.className = replay ? "replaying" : recording ? "recording" : "";
    status.textContent = replay
      ? `Replay ${this.replay.index + 1}/${this.replay.item.total}`
      : recording
        ? `Recording · ${this.recording.samples} frames`
        : this.recording.error || "Live";
    if (this.current) byId("replayStamp").textContent = stamp(this.current.t);
  }

  async downloadFrame() {
    if (!this.current) throw new Error("Waiting for the first frame");
    const frame = this.current;
    const files = [
      {
        name: "snapshot.json",
        data: new TextEncoder().encode(JSON.stringify(frame, null, 2)),
      },
    ];
    for (const [agent, cameras] of Object.entries(frame.cameras || {})) {
      for (const [camera, state] of Object.entries(cameras)) {
        const source = this.replay
          ? frame.images?.[agent]?.[camera]
          : state.online
            ? `/v1/image/${agent}/${camera}`
            : "";
        if (!source) continue;
        const response = await fetch(source, {
          signal: AbortSignal.timeout(4000),
        });
        if (response.ok)
          files.push({
            name: `${agent}-${camera}.jpg`,
            data: new Uint8Array(await response.arrayBuffer()),
          });
      }
    }
    download(
      zipStore(files),
      `lavira_m3-frame-${Math.floor(frame.t * 1000)}.zip`,
    );
  }
}
