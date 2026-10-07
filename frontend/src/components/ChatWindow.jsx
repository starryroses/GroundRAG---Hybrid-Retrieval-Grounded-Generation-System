import MessageBubble from "./MessageBubble";

function ChatWindow({ messages }) {
  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <h2>Ask your documents</h2>
        <p>
          Ask a question about the indexed BBC News corpus.
        </p>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          role={message.role}
          content={message.content}
        />
      ))}
    </div>
  );
}

export default ChatWindow;