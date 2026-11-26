// Small helpers for talking to the FastAPI backend (proxied by Vite).

async function errorText(res) {
  try {
    const j = await res.json();
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return j.detail.map((d) => d.msg).join("; ");
    return res.statusText;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

async function handle(res) {
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export const getJSON = (url) => fetch(url).then(handle);

export const postJSON = (url, body) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle);

export const postForm = (url, formData) => fetch(url, { method: "POST", body: formData }).then(handle);

export const fmt = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
