import type { Message } from "@/lib/types";

interface Props {
  message: Message;
  showTail?: boolean;
}

export function UserMessage({ message, showTail = true }: Props) {
  const radius = showTail
    ? "rounded-[20px_20px_4px_20px]"
    : "rounded-[20px]";
  return (
    <div className="flex justify-end px-2">
      <div
        className={`imessage-bubble-user imessage-bubble-enter max-w-[75%] whitespace-pre-wrap px-[14px] py-[8px] text-[15px] leading-[1.35] ${radius}`}
      >
        {message.content}
      </div>
    </div>
  );
}
