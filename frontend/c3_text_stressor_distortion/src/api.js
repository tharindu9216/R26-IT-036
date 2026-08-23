const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8005";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      // Keep the generic message when the response is not JSON.
    }
    throw new Error(message);
  }

  if (response.status === 204) return null;
  return response.json();
}

export function listDiaryEntries(limit = 50, offset = 0) {
  return request(`/api/diary?limit=${limit}&offset=${offset}`);
}

export function createDiaryEntry(content, title = "") {
  return request("/api/diary", {
    method: "POST",
    body: JSON.stringify({ content, title }),
  });
}

export function deleteDiaryEntry(id) {
  return request(`/api/diary/${id}`, { method: "DELETE" });
}

export function getDiarySummary() {
  return request("/api/diary/summary");
}

export function explainDiaryEntry(id) {
  return request(`/api/diary/${id}/explain`, { method: "POST" });
}
