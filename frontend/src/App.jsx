import { useState } from "react";
import "./App.css";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import { queryRAG } from "./services/api";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  async function handleSubmit(event) {
  event.preventDefault();

  const query = input.trim();

  if (!query) {
    return;
  }

  const userMessage = {
    id: Date.now(),
    role: "user",
    content: query,
  };

  setMessages((currentMessages) => [
    ...currentMessages,
    userMessage,
  ]);

  setInput("");

  try {
    const result = await queryRAG(query);

    const assistantMessage = {
      id: Date.now() + 1,
      role: "assistant",
      content: result.answer,
    };

    setMessages((currentMessages) => [
      ...currentMessages,
      assistantMessage,
    ]);
  } catch (error) {
    const errorMessage = {
      id: Date.now() + 1,
      role: "assistant",
      content: `Error: ${error.message}`,
    };

    setMessages((currentMessages) => [
      ...currentMessages,
      errorMessage,
    ]);
  }
}

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Advanced RAG</h1>
          <p>
            Hybrid Search · Reranking · Grounded Generation
          </p>
        </div>
      </header>

      <main className="chat-container">
        <ChatWindow messages={messages} />

        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
        />
      </main>
    </div>
  );
}

export default App;