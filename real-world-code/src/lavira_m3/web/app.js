/** Compose the original field-board interactions using configured agent profiles. */
import { byId, palette, request, post } from "./dashboard_api.js";
import { MapView } from "./map_view.js";
import { GoalBoard } from "./goal_board.js";
import { AgentPanels } from "./agent_panels.js";
import { RecordingControls } from "./recording_controls.js";
import { NetworkPanel } from "./network_panel.js";

window.__m3Errors = [];
window.addEventListener("error", (event) =>
  window.__m3Errors.push(String(event.error || event.message)),
);
window.addEventListener("unhandledrejection", (event) =>
  window.__m3Errors.push(String(event.reason)),
);

async function start() {
  const config = await request("/v1/dashboard/config");
  const names = Object.keys(config.agents);
  byId("portLabel").textContent =
    `PORT ${location.port || (location.protocol === "https:" ? "443" : "80")}`;
  for (const agent of names) {
    byId("selectedAgent").add(new Option(agent, agent));
    byId("mapView").add(new Option(`${agent} planning map`, `plan:${agent}`));
  }
  byId("sourceLayers").innerHTML = names
    .map(
      (name) =>
        `<label><input type="checkbox" data-layer="${name}" checked>${name} occupancy</label>`,
    )
    .join("");
  byId("legend").innerHTML =
    names
      .map(
        (name, index) =>
          `<span><i style="background:${palette[index % palette.length]}"></i>${name}</span>`,
      )
      .join("") +
    [
      ["#facc15", "VA projection"],
      ["#be7ad0", "Traveled path"],
      ["#c6bfd2", "Shared observations"],
      ["#64765a", "Obstacle inflation"],
    ]
      .map(
        ([color, label]) =>
          `<span><i style="background:${color}"></i>${label}</span>`,
      )
      .join("");
  const map = new MapView(config),
    panels = new AgentPanels(config),
    network = new NetworkPanel();
  let busy = false;
  const render = (frame, replay = false) => {
    map.update(frame, replay);
    panels.render(frame, replay);
    board.render(frame, replay);
    recording.current = frame;
    recording.controls();
    byId("rebuildMap").disabled = replay;
  };
  const refresh = async () => {
    if (busy || recording.replay) return;
    busy = true;
    try {
      const frame = await request(
        `/v1/dashboard?robot=${encodeURIComponent(byId("selectedAgent").value)}`,
      );
      if (!recording.replay) render(frame);
    } catch (error) {
      byId("mapStatus").textContent = error.message;
    } finally {
      busy = false;
    }
  };
  const board = new GoalBoard(config, refresh);
  const recording = new RecordingControls(render, refresh);
  byId("sourceLayers")
    .querySelectorAll("input")
    .forEach((input) => {
      input.onchange = () => {
        if (input.checked) map.layers.add(input.dataset.layer);
        else map.layers.delete(input.dataset.layer);
        map.draw();
      };
    });
  byId("selectedAgent").onchange = () => {
    map.draw();
    refresh();
  };
  byId("rebuildMap").onclick = async () => {
    if (recording.replay || !confirm("Clear map history and rebuild the fused map from new observations?"))
      return;
    try {
      await post("/v1/clear");
      map.view.ready = false;
      await refresh();
    } catch (error) {
      byId("hint").textContent = error.message;
    }
  };
  byId("showFilterRanges").onclick = () => {
    byId("frameLabel").textContent = config.frame;
    byId("filterRangesContent").textContent = JSON.stringify(
      { agents: config.agents, safety: config.safety },
      null,
      2,
    );
    byId("filterRangesDialog").showModal();
  };
  await refresh();
  await recording.list();
  await network.refresh();
  setInterval(refresh, 500);
  setInterval(() => recording.tick(), 500);
  setInterval(() => network.refresh(), 1000);
  document.title = `lavira_m3 · ${names.length}-agent console`;
}

start().catch((error) => {
  byId("mapStatus").textContent = `Startup failed: ${error.message}`;
  window.__m3Errors.push(String(error));
});
