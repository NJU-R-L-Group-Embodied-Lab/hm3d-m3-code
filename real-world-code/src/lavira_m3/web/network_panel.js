import { byId, escapeHtml, request, stamp } from "./dashboard_api.js";

const decimal = (value) => Number(value || 0).toFixed(2);
export class NetworkPanel {
  constructor() {
    this.last = null;
    this.busy = false;
    byId("netCopy").onclick = async () => {
      try {
        await navigator.clipboard.writeText(JSON.stringify(this.last, null, 2));
        byId("netAlert").textContent = "Network log copied";
      } catch (error) {
        byId("netAlert").textContent = error.message;
      }
    };
  }

  async refresh() {
    if (this.busy) return;
    this.busy = true;
    try {
      const data = await request("/v1/net");
      this.last = data;
      byId("netIface").textContent = `NIC ${data.iface}`;
      byId("netAlert").textContent =
        `HTTP ${decimal(data.http.rps)} req/s · ↓ ${decimal(data.http.tx_mbps)} Mbps · ↑ ${decimal(data.http.rx_mbps)} Mbps`;
      byId("netAlert").classList.remove("hot");
      const nic = data.nic[data.iface] || {};
      byId("netStrip").innerHTML = [
        ["NIC rx", nic.rx_mbps],
        ["NIC tx", nic.tx_mbps],
        ["HTTP rx", data.http.rx_mbps],
        ["HTTP tx", data.http.tx_mbps],
      ]
        .map(
          ([name, value]) =>
            `<div class="net-card"><span>${name}</span><b>${decimal(value)} Mbps</b></div>`,
        )
        .join("");
      byId("netChannelRows").innerHTML = Object.entries(data.channels)
        .map(
          ([name, row]) =>
            `<tr><td>${escapeHtml(name)}</td><td>${decimal(row.rx_mbps)}</td><td>${decimal(row.tx_mbps)}</td><td>${decimal(row.rps)}</td><td>${decimal(row.avg_ms)} / ${decimal(row.max_ms)}</td></tr>`,
        )
        .join("");
      byId("netRobotRows").innerHTML = Object.entries(data.robots)
        .map(
          ([name, row]) =>
            `<tr><td>${escapeHtml(name)}</td><td>${decimal(row.up_mbps)}</td><td>${decimal(row.down_mbps)}</td><td>${row.last_ago_s == null ? "—" : decimal(row.last_ago_s)} s</td></tr>`,
        )
        .join("");
      byId("netHistoryRows").innerHTML = data.history
        .slice(-60)
        .reverse()
        .map(
          (row) =>
            `<tr><td>${stamp(row.t)}</td>${["nic_rx_mbps", "nic_tx_mbps", "http_rx_mbps", "http_tx_mbps"].map((key) => `<td>${decimal(row[key])}</td>`).join("")}</tr>`,
        )
        .join("");
      byId("netSummary").textContent = `Last ${data.history.length} sample windows`;
      const canvas = byId("netSpark"),
        ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const peak = Math.max(
        0.01,
        ...data.history.map((row) => row.http_tx_mbps || 0),
      );
      ctx.beginPath();
      ctx.strokeStyle = "#b9f04b";
      ctx.lineWidth = 2;
      data.history.forEach((row, index) => {
        const x = (index / Math.max(1, data.history.length - 1)) * canvas.width;
        const y =
          canvas.height -
          ((row.http_tx_mbps || 0) / peak) * (canvas.height - 4);
        if (index) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      });
      ctx.stroke();
    } catch (error) {
      byId("netAlert").textContent = error.message;
      byId("netAlert").classList.add("hot");
    } finally {
      this.busy = false;
    }
  }
}
