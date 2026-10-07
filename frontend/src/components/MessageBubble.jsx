function MessageBubble({ role, content }) {
  const isUser = role === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={`message-bubble ${isUser ? "user-message" : "assistant-message"}`}>
        <div className="message-role">
          {isUser ? "You" : "Assistant"}
        </div>

        <div className="message-content">
          {content}
        </div>
      </div>
    </div>
  );
}

export default MessageBubble;