import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ChatMessage } from "../types";
import { SupportContacts } from "./SupportContacts";


export function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);



  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // The backend keeps this session's turns and replays them into every prompt,
  // so after a reload the conversation has to be redrawn here too -- otherwise
  // the model is answering with context that is no longer on screen. Per-turn
  // metadata (emotions, XAI turn ids) is not restored: those explanations
  // expire server-side, so only the text comes back.
  useEffect(() => {
    let cancelled = false;
    api
      .chatHistory()
      .then((restored) => {
        if (!cancelled && restored.length > 0) {
          setMessages(restored);
        }
      })
      .catch(() => {
        // A fresh conversation is the right fallback if this fails.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSend() {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", text }]);
    setSending(true);
    try {
      const result = await api.sendChat(text);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: result.reply,
          meta: {
            support_contacts: result.support_contacts,
            input_text: text,
            turn_id: result.turn_id,
          },
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSending(false);
    }
  }

  async function handleNewChat() {
    if (sending) return;
    setError(null);
    try {
      await api.resetChat();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
    setMessages([]);
  }

  return (
    <div className="card chat-card">
      <div className="chat-header">
        <div className="chat-heading">
          <span className="companion-avatar" aria-hidden="true">♡</span>
          <div>
            <h2>SentiVeraAI</h2>
            <span className="online-status"><i /> Here with you</span>
          </div>
        </div>
        <button
          type="button"
          className="new-chat-button"
          onClick={handleNewChat}
          disabled={sending || messages.length === 0}
          title="Forget this conversation and start a new one"
        >
          New chat
        </button>
      </div>
      <div className="chat-log">
        {messages.length === 0 && (
          <div className="empty-state">
            <span className="empty-state-icon" aria-hidden="true">☀</span>
            <h3>How are you feeling right now?</h3>
            <p>There’s no right or wrong way to begin. Share as much or as little as you’d like.</p>
          </div>
        )}
        {messages.map((message, index) => (
          <div key={index} className={`bubble ${message.role}`}>
            <p>{message.text}</p>
            <SupportContacts contacts={message.meta?.support_contacts} />

          </div>
        ))}
        {sending && (
          <div className="bubble assistant thinking">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      {error && <p className="error">{error}</p>}
      <div className="chat-input">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              handleSend();
            }
          }}
          placeholder="Type how you're feeling…"
          aria-label="Share how you are feeling"
          disabled={sending}
          rows={2}
        />
        <button className="primary" onClick={handleSend} disabled={sending || !input.trim()}>
          Send <span aria-hidden="true">↑</span>
        </button>
      </div>
    </div>
  );
}
