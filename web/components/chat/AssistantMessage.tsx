import type { Message } from "@/lib/types";
import { MarkdownWithCitations } from "./MarkdownWithCitations";

interface Props {
  message: Message;
  showTail?: boolean;
  showMetadata?: boolean;
}

export function AssistantMessage({
  message,
  showTail = true,
  showMetadata = true,
}: Props) {
  const radius = showTail
    ? "rounded-[20px_20px_20px_4px]"
    : "rounded-[20px]";

  if (message.error) {
    return (
      <div className="flex justify-start px-2">
        <div
          className={`imessage-bubble-enter max-w-[80%] border border-destructive/40 bg-destructive/10 px-[14px] py-[10px] text-[15px] text-destructive ${radius}`}
        >
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex flex-col items-start gap-1 px-2">
      <div
        className={`imessage-bubble-assistant imessage-bubble-enter max-w-[80%] px-[14px] py-[10px] text-[15px] leading-[1.4] ${radius}`}
      >
        <MarkdownWithCitations content={message.content} />
      </div>
      {showMetadata && message.metadata && (
        <div className="ml-3 flex gap-2 text-[10px] text-muted-foreground">
          <span>{message.metadata.model}</span>
          <span>·</span>
          <span>{(message.metadata.latency_ms / 1000).toFixed(1)}s</span>
          <span>·</span>
          <span>{message.metadata.token_usage.total ?? 0} tokens</span>
        </div>
      )}
    </div>
  );
}
