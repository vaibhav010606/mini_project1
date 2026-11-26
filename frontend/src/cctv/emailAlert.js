// Send a camera alert email through Formspree (https://formspree.io).
// The form ID comes from frontend/.env.local:  VITE_FORMSPREE_ID=xxxxxxxx
// That file is not uploaded to GitHub, so the ID stays private.
import { FLOOR_NAMES } from "../mall/cameras.js";

export const FORMSPREE_ID = (import.meta.env.VITE_FORMSPREE_ID || "").trim();
export const emailReady = () => FORMSPREE_ID.length > 0;

const STATUS_WORDS = {
  crowded: "is CROWDED",
  busy: "is BUSY",
  normal: "is at a normal level",
};

export function alertText(cam, status, people, operator, when = new Date()) {
  const floor = `${FLOOR_NAMES[cam.floor]} floor`;
  const level = (status || "normal").toUpperCase();
  const time = when.toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "medium" });
  const sentence = `Location: on the ${floor}, ${cam.area} ${STATUS_WORDS[status] || STATUS_WORDS.normal}.`;
  const subject = `${status === "crowded" ? "⚠ " : ""}${level}: ${cam.name}, ${floor} (${cam.id})`;
  const action = status === "crowded"
    ? "Suggested action: send staff to this area and guide people towards other exits and walkways."
    : status === "busy"
      ? "Suggested action: keep watching this camera; the area may become crowded."
      : "No action needed right now.";
  const message = [
    sentence,
    "",
    `Camera: ${cam.id} (${cam.name})`,
    `Floor: ${floor}`,
    `Section: ${cam.area}`,
    `Crowd level: ${level}`,
    `People in view: ${people ?? "unknown"}`,
    `Time: ${time}`,
    `Sent by: ${operator || "control room operator"}`,
    "",
    action,
    "",
    "Crowd Safety AI control room · simulated demo alert",
  ].join("\n");
  return { subject, message, sentence, level, floor, time };
}

export async function sendCameraAlert(cam, status, people, operator) {
  if (!emailReady()) throw new Error("Email is not set up yet: add VITE_FORMSPREE_ID to frontend/.env.local");
  const t = alertText(cam, status, people, operator);
  const res = await fetch(`https://formspree.io/f/${encodeURIComponent(FORMSPREE_ID)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      _subject: t.subject,
      message: t.message,
      location: t.sentence,
      camera: `${cam.id} - ${cam.name}`,
      floor: t.floor,
      section: cam.area,
      crowd_level: t.level,
      people_in_view: people ?? "unknown",
      time: t.time,
      operator: operator || "control room operator",
    }),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.errors?.length) detail = j.errors.map((e) => e.message).join("; ");
      else if (j.error) detail = j.error;
    } catch {
      /* keep status text */
    }
    throw new Error(`Formspree refused the email: ${detail}`);
  }
  return t;
}
