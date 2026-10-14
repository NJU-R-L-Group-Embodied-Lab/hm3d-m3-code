import { byId, escapeHtml, imageSource, post } from "./dashboard_api.js";

async function referenceImage(file) {
  if (!file || file.size > 2_000_000 || !file.type.startsWith("image/"))
    throw new Error("Choose an image of at most 2 MB");
  const image = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 1024 / Math.max(image.width, image.height));
  canvas.width = image.width * scale;
  canvas.height = image.height * scale;
  canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
  image.close();
  return canvas.toDataURL("image/jpeg", 0.85);
}

export class GoalBoard {
  constructor(config, refresh) {
    this.names = Object.keys(config.agents);
    this.refresh = refresh;
    this.replay = false;
    byId("teamLabel").textContent =
      `TEAM TASK BOARD · SHARED BY ${this.names.length} AGENTS`;
    ["ON", "IIN", "TN"].forEach((modality) => {
      for (let i = 0; i < 5; i++) this.add({ modality });
    });
    byId("addGoal").onclick = () => this.add({ modality: "ON" });
    byId("goalImport").onchange = async (event) => {
      try {
        const file = event.target.files[0];
        if (!file || file.size > 24_000_000)
          throw new Error("Goal JSON must be at most 24 MB");
        const parsed = JSON.parse(await file.text());
        const goals = Array.isArray(parsed) ? parsed : parsed.goals;
        if (!Array.isArray(goals) || !goals.length || goals.length > 100)
          throw new Error("Provide 1–100 goals");
        if (
          goals.some(
            (goal) =>
              !goal ||
              typeof goal !== "object" ||
              !["ON", "TN", "IIN"].includes(goal.modality || "ON"),
          )
        )
          throw new Error("Goal type must be ON, IIN or TN");
        byId("laviraGoals").replaceChildren();
        goals.forEach((goal) => this.add(goal));
        this.message(`Loaded ${goals.length} goal(s); complete them, then send.`);
      } catch (error) {
        this.message(error.message);
      }
    };
    byId("laviraSubmit").onclick = () => this.submit();
    byId("laviraClearRuntime").onclick = async () => {
      if (this.replay) return;
      try {
        await post("/api/board/clear-runtime");
        this.message("Cleared previous-run overlays; goals kept.");
        await this.refresh();
      } catch (error) {
        this.message(error.message);
      }
    };
  }

  message(text) {
    byId("laviraStatus").textContent = text;
  }

  add(goal) {
    const row = document.createElement("div");
    row.className = "goal-row";
    row.innerHTML = `<div class="goal-heading"><b></b><select aria-label="Goal type"><option>ON</option><option>IIN</option><option>TN</option></select><button type="button" aria-label="Remove goal">×</button></div>
      <label>Object<input class="category" type="text" placeholder="Target object"></label>
      <label class="text-label">Description<textarea class="description" rows="3" placeholder="Describe the target's appearance and location"></textarea></label>
      <label class="image-label">Reference image<input type="file" accept="image/*"></label><img class="reference-preview" hidden alt="Goal reference image">`;
    row.querySelector("select").value = goal.modality || "ON";
    row.querySelector(".category").value =
      goal.object_category || goal.name || "";
    row.querySelector(".description").value =
      goal.tn_text || goal.description || "";
    row.reference = imageSource(
      goal.reference_image_b64 || goal.reference_image,
    );
    const update = () => {
      const type = row.querySelector("select").value;
      row.querySelector(".text-label").hidden = type !== "TN";
      row.querySelector(".image-label").hidden = type !== "IIN";
      const image = row.querySelector(".reference-preview");
      image.hidden = type !== "IIN" || !row.reference;
      if (row.reference) image.src = row.reference;
    };
    row.querySelector("select").onchange = update;
    row.querySelector("button").onclick = () => {
      row.remove();
      this.numberRows();
    };
    row.querySelector("input[type=file]").onchange = async (event) => {
      try {
        row.reference = await referenceImage(event.target.files[0]);
        update();
      } catch (error) {
        this.message(error.message);
      }
    };
    byId("laviraGoals").append(row);
    update();
    this.numberRows();
  }

  numberRows() {
    [...byId("laviraGoals").children].forEach((row, index) => {
      row.querySelector("b").textContent = `Goal ${index + 1}`;
    });
  }

  async submit() {
    if (this.replay) return;
    try {
      const goals = [];
      for (const row of byId("laviraGoals").children) {
        const modality = row.querySelector("select").value;
        const category = row.querySelector(".category").value.trim();
        const description = row.querySelector(".description").value.trim();
        if (!category && !description && !row.reference) continue;
        if (modality === "IIN" && !row.reference)
          throw new Error("Upload a reference image for the image goal");
        if (modality === "TN" && !description)
          throw new Error("Enter a description for the text goal");
        if (modality === "ON" && !category) throw new Error("Enter an object category");
        const goal = {
          goal_index: goals.length,
          modality,
          object_category:
            category || description.slice(0, 60) || "image target",
        };
        if (modality === "TN") goal.tn_text = description;
        if (modality === "IIN")
          goal[
            row.reference.startsWith("data:")
              ? "reference_image_b64"
              : "reference_image"
          ] = row.reference;
        goals.push(goal);
      }
      if (!goals.length) throw new Error("Add at least one goal");
      await post("/api/board/init", { robot: "web", goals });
      this.message(`Sent ${goals.length} goal(s); search started.`);
      await this.refresh();
    } catch (error) {
      this.message(error.message);
    }
  }

  render(frame, replay) {
    this.replay = replay;
    for (const id of [
      "laviraSubmit",
      "laviraClearRuntime",
      "addGoal",
      "goalImport",
    ])
      byId(id).disabled = replay;
    const board = frame.board;
    const goals = board.goals || [];
    byId("taskBoardSummary").innerHTML = goals.length
      ? goals
          .map((goal, index) => {
            const finder = board.found_by?.[index];
            const found = Number.isInteger(finder) && finder >= 0;
            const source = imageSource(goal.reference_image);
            return `<li><strong>${escapeHtml(goal.object_category)}</strong><small>${escapeHtml(goal.modality)} · ${found ? "FOUND · " + escapeHtml(this.names[finder] || finder) : "searching"}</small>${source ? `<img src="${escapeHtml(source)}" alt="Goal reference image" width="64">` : ""}</li>`;
          })
          .join("")
      : '<li class="task-board-empty-item">Add goals below to start a cooperative search.</li>';
    const done = (board.found_by || []).filter((value) => value >= 0).length;
    byId("taskBoardCount").textContent =
      `${done} / ${goals.length} done · ${replay ? "Replay" : "Shared task board"}`;
  }
}
