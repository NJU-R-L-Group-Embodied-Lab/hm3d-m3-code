/** Same-origin dashboard requests and safe presentation primitives. */
export const byId = (id) => document.getElementById(id);
export const palette = ["#b9f04b", "#526eff", "#ff7959", "#a78bfa", "#27b6a0"];
export const escapeHtml = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[ch],
  );
export const seconds = (value) =>
  Number.isFinite(value) ? `${(value / 1000).toFixed(1)}s` : "—";
export const stamp = (value) =>
  new Date(value * 1000).toLocaleString("zh-CN", { hour12: false });

export async function request(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    signal: AbortSignal.timeout(6000),
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false)
    throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

export const post = (path, body = {}) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export function imageSource(value) {
  const text = String(value || "");
  if (/^\/9j\/[A-Za-z0-9+/=\s]+$/.test(text))
    return "data:image/jpeg;base64," + text;
  if (/^data:image\/(?:jpeg|png|webp);base64,/.test(text)) return text;
  if (
    /^\/(?:v1\/image|api\/board\/(?:ref|history))\/[a-zA-Z0-9_/-]+(?:\?[^#]*)?$/.test(
      text,
    )
  )
    return text;
  return "";
}

export function download(blob, name) {
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
