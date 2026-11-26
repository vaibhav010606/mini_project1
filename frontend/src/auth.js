// Demo login for presentations. This is NOT real security: there are no user accounts,
// nothing is sent to a server, and the demo details below are shown on the login page.
// The login lasts for the browser tab's session (sessionStorage).

export const DEMO_USER = { username: "operator", password: "crowd2026", name: "Control room operator" };
const KEY = "crowd-safety-demo-login";

export function currentUser() {
  try {
    return JSON.parse(sessionStorage.getItem(KEY));
  } catch {
    return null;
  }
}

export function login(username, password) {
  if (username.trim().toLowerCase() === DEMO_USER.username && password === DEMO_USER.password) {
    const user = { username: DEMO_USER.username, name: DEMO_USER.name, since: new Date().toISOString() };
    sessionStorage.setItem(KEY, JSON.stringify(user));
    return user;
  }
  return null;
}

export function logout() {
  sessionStorage.removeItem(KEY);
}
