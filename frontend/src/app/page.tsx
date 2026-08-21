"use client";

import {
  FileText,
  LogOut,
  MessageCircle,
  Send,
  Sparkles,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

// The FastAPI backend's address. Hardcoded here rather than pulled from an
// env var — that's a real MVP shortcut, not an oversight: per CLAUDE.md's
// "configuration layer" principle, a real deployment would read this from
// something like NEXT_PUBLIC_API_URL so the frontend can point at a
// different backend address per environment (dev/staging/prod) without
// editing code. One hardcoded constant is the right amount of machinery
// for a single developer testing locally right now.
const API_URL = "http://127.0.0.1:8000";

// "use client" (top of file) matters here: by default, Next.js App Router
// renders components on the SERVER first. Server components can't use
// useState/useEffect or respond to clicks — they only know how to render
// once, ahead of time. This page needs to hold state (which document is
// selected, the chat history) and react to button clicks, so it has to
// opt OUT of server rendering and run as a normal, interactive, in-browser
// React component instead.

type Document = {
  document_id: string;
  filename: string;
  status: string;
  created_at: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  sources?: { chunk_index: number; page_number: number | null; text: string }[];
};

// Where the JWT lives in the browser between page loads. localStorage
// persists across refreshes and even closing the tab (unlike plain React
// state, which resets on every reload) — this is what lets you log in
// once and stay logged in, rather than re-authenticating on every visit.
// A real production app would weigh localStorage against httpOnly cookies
// (cookies are safer against a specific attack — malicious JavaScript
// reading the token — but need more server-side setup); localStorage is
// the simpler, more transparent option for an MVP you're building to
// understand, not the objectively "correct" choice for every app.
const TOKEN_STORAGE_KEY = "doc-chat-token";

// Cycled through while waiting for the first token — purely cosmetic, but
// a single static word ("Thinking...") starts to feel stale/frozen on a
// longer wait (retrieval + a cold-loading model can take several
// seconds). Deliberately playful rather than literal pipeline-stage
// names — nobody needs to believe the model is actually "marinating" —
// it just needs to feel alive while something genuinely takes a few
// seconds to happen.
const THINKING_PHRASES = [
  "Thinking...",
  "Marinating...",
  "Wrangling context...",
  "Percolating...",
  "Deciphering the document...",
  "Manifesting an answer...",
  "Baking a response...",
];

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "signup">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [questionInput, setQuestionInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [thinkingPhraseIndex, setThinkingPhraseIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // A ref to an empty div placed after the last message (see the JSX
  // below) — scrollIntoView() on that div is what actually moves the
  // scroll position, rather than trying to compute scroll offsets by
  // hand. A ref, not state, because scrolling is a direct DOM action with
  // no visual output of its own — it doesn't belong in React's render
  // cycle the way `messages` or `asking` do.
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Read any previously-saved token ONCE, right after the page mounts in
  // the browser. This can't happen during the initial render itself:
  // localStorage doesn't exist on the server, and Next.js renders this
  // component once on the server first (even "use client" components) —
  // reaching into localStorage before that render would crash. Doing it
  // inside useEffect guarantees this only runs once we're truly running
  // in the browser.
  useEffect(() => {
    setToken(localStorage.getItem(TOKEN_STORAGE_KEY));
    setAuthChecked(true);
  }, []);

  function saveToken(newToken: string) {
    localStorage.setItem(TOKEN_STORAGE_KEY, newToken);
    setToken(newToken);
  }

  function logout() {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setToken(null);
    setDocuments([]);
    setSelectedDocumentId(null);
    setMessages([]);
  }

  async function handleAuth() {
    setAuthLoading(true);
    setAuthError(null);
    try {
      const response = await fetch(`${API_URL}/auth/${authMode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: authEmail, password: authPassword }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Authentication failed");
      }
      saveToken(data.access_token);
      setAuthPassword("");
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setAuthLoading(false);
    }
  }

  async function fetchDocuments() {
    const response = await fetch(`${API_URL}/documents`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (response.status === 401) {
      // The token we have is expired or otherwise no longer valid — same
      // "server rejected our proof of login" case get_current_user_id
      // (dependencies.py) raises on the backend. Rather than showing a
      // confusing error, just send the user back to the login screen.
      logout();
      return;
    }
    const data: Document[] = await response.json();
    setDocuments(data);
  }

  // Runs once a token actually exists (either just logged in, or restored
  // from localStorage above) — this is what actually fetches the "which
  // documents can I see" list, now correctly scoped to whoever is logged
  // in, per the ownership work on the backend.
  useEffect(() => {
    if (token) {
      fetchDocuments();
    }
  }, [token]);

  // POLLING — the actual frontend consequence of ingestion now running in
  // a background Celery worker instead of inline during the upload
  // request: the server no longer tells us the moment processing
  // finishes, because nothing is still connected waiting for it by then.
  // The only way to find out is to keep asking. This effect re-runs every
  // time `documents` changes; if anything is still "pending", it schedules
  // ONE more fetch 2 seconds later. Once a document flips to "ready" or
  // "failed", fetchDocuments() updates `documents`, this effect re-runs,
  // finds nothing pending anymore, and the polling naturally stops on its
  // own — no separate "stop polling" call needed anywhere.
  useEffect(() => {
    const hasPending = documents.some((doc) => doc.status === "pending");
    if (!hasPending || !token) return;

    const timeoutId = setTimeout(() => {
      fetchDocuments();
    }, 2000);

    return () => clearTimeout(timeoutId);
  }, [documents, token]);

  // Advances THINKING_PHRASES every 1.5s while waiting on a response, and
  // resets back to the first phrase the moment asking becomes false —
  // this reset matters for the NEXT question: without it, a second
  // question would start mid-way through the phrase list instead of at
  // "Thinking..." again, which would look like a leftover glitch rather
  // than a fresh wait.
  useEffect(() => {
    if (!asking) {
      setThinkingPhraseIndex(0);
      return;
    }
    const intervalId = setInterval(() => {
      setThinkingPhraseIndex((i) => (i + 1) % THINKING_PHRASES.length);
    }, 1500);
    return () => clearInterval(intervalId);
  }, [asking]);

  // Auto-scroll to the newest message every time the list changes —
  // including every single streamed token, since each one produces a new
  // `messages` array (see handleAsk's setMessages calls). Without this,
  // a long answer would keep growing past the visible bottom of the chat
  // window with no indication anything new arrived unless you manually
  // scrolled down yourself.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    // FormData, not JSON — this mirrors exactly what the earlier
    // `curl -F "file=@..."` tests did: a multipart form body, which is
    // what FastAPI's UploadFile on the backend expects to parse.
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail ?? "Upload failed");
      }
      await fetchDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function handleDelete(documentId: string, filename: string) {
    // A plain browser confirm() dialog — deliberately the simplest
    // possible "are you sure?" step, appropriate for how low-stakes this
    // actually is (re-uploading a document is trivial if this were
    // clicked by mistake) rather than building a custom modal for a
    // single yes/no decision.
    if (!window.confirm(`Delete "${filename}"? This can't be undone.`)) {
      return;
    }

    setError(null);
    try {
      const response = await fetch(`${API_URL}/documents/${documentId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail ?? "Delete failed");
      }

      // If the document being deleted is also the one currently open in
      // the chat panel, clear that selection too — otherwise the UI would
      // keep showing a chat "with" a document that no longer exists.
      if (documentId === selectedDocumentId) {
        setSelectedDocumentId(null);
        setMessages([]);
      }

      await fetchDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  async function handleAsk() {
    if (!selectedDocumentId || !questionInput.trim()) return;

    const question = questionInput.trim();
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setQuestionInput("");
    setAsking(true);
    setError(null);

    // Add an empty assistant message up front, then fill its .text in as
    // tokens arrive — this is what makes the answer visibly "type itself
    // out" rather than appearing all at once. We track its position with
    // an index rather than searching for it later, since by the time
    // tokens start arriving, more messages could theoretically exist.
    const assistantMessageIndex = messages.length + 1; // +1: after the user message just added above
    setMessages((prev) => [...prev, { role: "assistant", text: "" }]);

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          document_id: selectedDocumentId,
          query: question,
          top_k: 5,
        }),
      });
      if (!response.ok || !response.body) {
        const data = await response.json();
        throw new Error(data.detail ?? "Chat request failed");
      }

      // The browser gives us the response body as a ReadableStream of raw
      // bytes — TextDecoder turns those bytes into text as they arrive.
      // SSE events are separated by a blank line ("\n\n"), so each loop
      // iteration reads whatever bytes have arrived so far, and we peel
      // off any COMPLETE events from the front of that buffer, since one
      // network chunk might contain multiple events, or a partial one.
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? ""; // last piece may be incomplete — keep it for next time

        for (const event of events) {
          const jsonText = event.replace(/^data: /, "");
          if (!jsonText) continue;
          const payload = JSON.parse(jsonText);

          if (payload.type === "token") {
            setMessages((prev) => {
              const next = [...prev];
              next[assistantMessageIndex] = {
                ...next[assistantMessageIndex],
                text: next[assistantMessageIndex].text + payload.text,
              };
              return next;
            });
          } else if (payload.type === "done") {
            setMessages((prev) => {
              const next = [...prev];
              next[assistantMessageIndex] = {
                ...next[assistantMessageIndex],
                sources: payload.sources,
              };
              return next;
            });
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat request failed");
    } finally {
      setAsking(false);
    }
  }

  const selectedDocument = documents.find(
    (doc) => doc.document_id === selectedDocumentId
  );

  // Nothing rendered yet until we've checked localStorage for an existing
  // token (see the first useEffect) — avoids a flash of the login screen
  // for someone who's actually already logged in, right before it's
  // replaced a split second later by the real app.
  if (!authChecked) {
    return null;
  }

  if (!token) {
    return (
      <div className="flex flex-1 items-center justify-center bg-gradient-to-b from-zinc-50 to-indigo-50/40 px-4">
        <div className="w-full max-w-sm rounded-2xl border border-zinc-200 bg-white p-8 shadow-xl shadow-indigo-100/70">
          <div className="mb-6 flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 shadow-sm shadow-indigo-300">
              <MessageCircle className="h-5 w-5 text-white" strokeWidth={2.25} />
            </div>
            <span className="text-lg font-semibold tracking-tight text-zinc-900">doc-chat</span>
          </div>

          <h1 className="mb-1 text-xl font-semibold text-zinc-900">
            {authMode === "login" ? "Welcome back" : "Create your account"}
          </h1>
          <p className="mb-6 text-sm text-zinc-500">
            {authMode === "login"
              ? "Log in to chat with your documents."
              : "Upload documents and chat with them, entirely offline."}
          </p>

          <div className="space-y-3">
            <input
              type="email"
              placeholder="Email"
              value={authEmail}
              onChange={(e) => setAuthEmail(e.target.value)}
              className="w-full rounded-lg border border-zinc-200 px-3.5 py-2.5 text-sm text-zinc-900 outline-none transition-shadow placeholder:text-zinc-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10"
            />
            <input
              type="password"
              placeholder="Password"
              value={authPassword}
              onChange={(e) => setAuthPassword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAuth()}
              className="w-full rounded-lg border border-zinc-200 px-3.5 py-2.5 text-sm text-zinc-900 outline-none transition-shadow placeholder:text-zinc-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10"
            />
            {authError && (
              <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{authError}</p>
            )}
            <button
              onClick={handleAuth}
              disabled={authLoading || !authEmail || !authPassword}
              className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white shadow-md shadow-indigo-200 transition-all hover:bg-indigo-700 hover:shadow-lg hover:shadow-indigo-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none disabled:active:scale-100"
            >
              {authLoading ? "Please wait..." : authMode === "login" ? "Log in" : "Sign up"}
            </button>
            <button
              onClick={() => {
                setAuthMode(authMode === "login" ? "signup" : "login");
                setAuthError(null);
              }}
              className="w-full text-center text-sm text-zinc-500 transition-colors hover:text-indigo-600"
            >
              {authMode === "login" ? (
                <>No account? <span className="font-medium text-indigo-600">Sign up instead</span></>
              ) : (
                <>Already have an account? <span className="font-medium text-indigo-600">Log in</span></>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const STATUS_STYLES: Record<string, string> = {
    pending: "bg-amber-50 text-amber-700",
    ready: "bg-emerald-50 text-emerald-700",
    failed: "bg-red-50 text-red-700",
  };

  return (
    <div className="flex flex-1 bg-gradient-to-br from-zinc-50 to-indigo-50/30">
      {/* Left sidebar: upload + document list */}
      <aside className="flex w-80 shrink-0 flex-col border-r border-zinc-200 bg-white p-4">
        <div className="mb-5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600">
              <MessageCircle className="h-4 w-4 text-white" strokeWidth={2.25} />
            </div>
            <span className="font-semibold tracking-tight text-zinc-900">doc-chat</span>
          </div>
          <button
            onClick={logout}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
          >
            <LogOut className="h-3.5 w-3.5" />
            Log out
          </button>
        </div>

        <label className="mb-5 flex cursor-pointer flex-col items-center gap-1.5 rounded-xl border border-dashed border-zinc-300 px-4 py-5 text-center text-sm text-zinc-500 transition-all hover:-translate-y-0.5 hover:border-indigo-400 hover:bg-indigo-50/40 hover:text-indigo-600 hover:shadow-sm">
          <Upload className="h-5 w-5" strokeWidth={1.75} />
          {uploading ? "Uploading..." : "Upload a PDF or DOCX"}
          <input
            type="file"
            accept=".pdf,.docx"
            className="hidden"
            onChange={handleUpload}
            disabled={uploading}
          />
        </label>

        <p className="mb-2 px-1 text-xs font-medium tracking-wide text-zinc-400 uppercase">
          Documents
        </p>

        {documents.length === 0 && (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 px-4 py-8 text-center">
            <FileText className="h-6 w-6 text-zinc-300" strokeWidth={1.5} />
            <p className="text-xs text-zinc-400">
              No documents yet — upload one above to get started.
            </p>
          </div>
        )}

        <ul className="space-y-0.5 overflow-y-auto">
          {documents.map((doc) => {
            const isSelected = doc.document_id === selectedDocumentId;

            return (
              <li key={doc.document_id} className="group flex items-center">
                <button
                  onClick={() => {
                    setSelectedDocumentId(doc.document_id);
                    setMessages([]);
                  }}
                  className={`flex min-w-0 flex-1 items-center justify-between gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                    isSelected
                      ? "bg-indigo-50 font-medium text-indigo-700"
                      : "text-zinc-700 hover:bg-zinc-100"
                  }`}
                >
                  <span className="truncate">{doc.filename}</span>
                  <span
                    className={`shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-medium ${
                      STATUS_STYLES[doc.status] ?? "bg-zinc-100 text-zinc-500"
                    }`}
                  >
                    {doc.status}
                  </span>
                </button>
                {/* opacity-0 + group-hover:opacity-100: only visible when
                    hovering the row, same "don't clutter the UI with
                    controls until they're relevant" instinct as most real
                    file-list UIs (Google Drive, Finder, etc.) — deleting a
                    document isn't an action anyone needs visible at all
                    times. */}
                <button
                  onClick={() => handleDelete(doc.document_id, doc.filename)}
                  className="ml-1 shrink-0 rounded-md p-2 text-zinc-400 opacity-0 transition-colors hover:bg-red-50 hover:text-red-600 group-hover:opacity-100"
                  title="Delete document"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </li>
            );
          })}
        </ul>
      </aside>

      {/* Right side: chat */}
      <main className="flex flex-1 flex-col">
        {!selectedDocument ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-zinc-400">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-100">
              <MessageCircle className="h-6 w-6" strokeWidth={1.5} />
            </div>
            <p className="text-sm">Select a document on the left to start chatting.</p>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2 border-b border-zinc-200 bg-white px-6 py-3.5 text-sm text-zinc-500">
              <span>Chatting with</span>
              <span className="font-medium text-zinc-900">{selectedDocument.filename}</span>
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto p-6">
              {messages.map((message, i) => {
                // Only the LAST message can possibly be the empty
                // placeholder handleAsk() creates before any tokens have
                // arrived (see handleAsk's assistantMessageIndex logic) —
                // checking i === messages.length - 1 as well as
                // text === "" avoids ever mistaking an OLDER, already-
                // finished empty response for one still in progress.
                const isWaitingForFirstToken =
                  message.role === "assistant" &&
                  message.text === "" &&
                  asking &&
                  i === messages.length - 1;
                const isUser = message.role === "user";

                return (
                  <div
                    key={i}
                    className={`animate-rise flex items-end gap-2 ${isUser ? "flex-row-reverse" : ""}`}
                  >
                    <div
                      className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-medium ${
                        isUser ? "bg-zinc-800 text-white" : "bg-indigo-100 text-indigo-600"
                      }`}
                    >
                      {isUser ? "U" : <Sparkles className="h-3.5 w-3.5" />}
                    </div>
                    <div
                      className={`max-w-xl rounded-2xl px-4 py-2.5 text-sm ${
                        isUser
                          ? "bg-indigo-600 text-white"
                          : "border border-zinc-200 bg-white text-zinc-900 shadow-sm"
                      }`}
                    >
                      {isWaitingForFirstToken ? (
                        <div className="flex items-center gap-2 py-1 text-zinc-400">
                          <span>{THINKING_PHRASES[thinkingPhraseIndex]}</span>
                          <div className="flex items-center gap-1">
                            <span
                              className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400"
                              style={{ animationDelay: "0ms" }}
                            />
                            <span
                              className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400"
                              style={{ animationDelay: "150ms" }}
                            />
                            <span
                              className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400"
                              style={{ animationDelay: "300ms" }}
                            />
                          </div>
                        </div>
                      ) : (
                        <p className="whitespace-pre-wrap">{message.text}</p>
                      )}
                      {message.sources && message.sources.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1 border-t border-zinc-100 pt-2">
                          {message.sources.map((s, j) => (
                            <span
                              key={j}
                              className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-500"
                            >
                              p.{s.page_number ?? "?"}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              {/* An empty, invisible anchor — scrollIntoView() on this
                  element (via messagesEndRef, see the useEffect above) is
                  what actually performs the auto-scroll. Placing it after
                  every real message means "scroll to the bottom" and
                  "scroll to this element" are the same operation. */}
              <div ref={messagesEndRef} />
            </div>

            <div className="border-t border-zinc-200 bg-white p-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={questionInput}
                  onChange={(e) => setQuestionInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                  placeholder="Ask a question about this document..."
                  className="flex-1 rounded-lg border border-zinc-200 px-3.5 py-2.5 text-sm text-zinc-900 outline-none transition-shadow placeholder:text-zinc-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10"
                  disabled={asking}
                />
                <button
                  onClick={handleAsk}
                  disabled={asking || !questionInput.trim()}
                  className="flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white shadow-md shadow-indigo-200 transition-all hover:bg-indigo-700 hover:shadow-lg hover:shadow-indigo-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none disabled:active:scale-100"
                >
                  <Send className="h-4 w-4" strokeWidth={2} />
                </button>
              </div>
            </div>
          </>
        )}

        {error && (
          <div className="flex items-center justify-between border-t border-red-200 bg-red-50 px-6 py-2 text-sm text-red-700">
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-4 shrink-0 rounded-md p-1 text-red-400 transition-colors hover:bg-red-100 hover:text-red-700"
              title="Dismiss"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
