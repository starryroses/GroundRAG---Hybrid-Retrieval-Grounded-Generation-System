function ChatInput({ value, onChange, onSubmit }) {
  return (
    <form className="input-area" onSubmit={onSubmit}>
      <input
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Ask a question..."
      />

      <button type="submit">
        Send
      </button>
    </form>
  );
}

export default ChatInput;