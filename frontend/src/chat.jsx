
// src/components/Chat.jsx
import React, { useState } from 'react';
import { BASE_URL } from '../services/api';  // import the backend URL

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);

  async function send() {
    const msg = text.trim();
    if (!msg) return;

    // Add user message
    setMessages(prev => [...prev, { from: 'user', text: msg }]);
    setText('');
    setLoading(true);

    try {
      const res = await fetch(`${BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
      });

      if (!res.ok) throw new Error('Backend error');

      const j = await res.json();

      // Add bot response
      setMessages(prev => [...prev, { from: 'bot', text: j.answer }]);
    } catch (err) {
      setMessages(prev => [...prev, { from: 'bot', text: 'Error contacting backend' }]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 600, margin: '0 auto', padding: 16 }}>
      <div style={{ maxHeight: 320, overflowY: 'auto', padding: 8, border: '1px solid #f0f0f0', borderRadius: 8 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ textAlign: m.from === 'user' ? 'right' : 'left', margin: '8px 0' }}>
            <div
              style={{
                display: 'inline-block',
                padding: 8,
                borderRadius: 8,
                background: m.from === 'user' ? '#dcfce7' : '#f3f4f6'
              }}
            >
              {m.text}
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          style={{ flex: 1, padding: 8 }}
          placeholder="Type your question"
        />
        <button onClick={send} disabled={loading} style={{ padding: '8px 12px' }}>
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
