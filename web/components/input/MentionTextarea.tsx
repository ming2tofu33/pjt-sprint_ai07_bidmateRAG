"use client";

import { MentionsInput, Mention, SuggestionDataItem } from "react-mentions";
import { useStore } from "@/store/useStore";
import { getCachedCommands } from "@/lib/commands";

interface Props {
  value: string;
  onChange: (text: string) => void;
  onEnter: () => void;
  disabled?: boolean;
  inputRef?: React.RefObject<HTMLTextAreaElement | null>;
}

const mentionStyle = {
  control: {
    backgroundColor: "transparent",
    fontSize: 15,
    minHeight: 32,
  },
  input: {
    padding: "6px 4px",
    border: "none",
    borderRadius: 0,
    minHeight: 32,
    maxHeight: 140,
    overflow: "auto",
    outline: "none",
    color: "inherit",
    lineHeight: "1.4",
  },
  suggestions: {
    list: {
      backgroundColor: "hsl(var(--popover))",
      border: "1px solid hsl(var(--border))",
      borderRadius: 6,
      fontSize: 13,
      maxHeight: 240,
      overflowY: "auto" as const,
      zIndex: 50,
    },
    item: {
      padding: "6px 12px",
      "&focused": {
        backgroundColor: "hsl(var(--accent))",
      },
    },
  },
};

export function MentionTextarea({ value, onChange, onEnter, disabled, inputRef }: Props) {
  const documents = useStore((s) => s.documents);
  const pinDoc = useStore((s) => s.pinDoc);
  const setCommand = useStore((s) => s.setCommand);

  const documentSuggestions: SuggestionDataItem[] = documents.map((d) => ({
    id: d.id,
    display: d.title.length > 40 ? d.title.slice(0, 37) + "..." : d.title,
  }));

  const commandSuggestions: SuggestionDataItem[] = getCachedCommands().map(
    (c) => ({ id: c.id, display: `${c.icon} ${c.label} — ${c.description}` })
  );

  const handleDocSelect = (id: string | number) => {
    const doc = documents.find((d) => d.id === id);
    if (doc) pinDoc(doc);
  };

  const handleCommandSelect = (id: string | number) => {
    const cmd = getCachedCommands().find((c) => c.id === id);
    if (cmd) setCommand(cmd);
  };

  return (
    <MentionsInput
      value={value}
      onChange={(_event, newValue) => onChange(newValue)}
      onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement> | React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          onEnter();
        }
      }}
      placeholder="질문 입력. @로 문서 · /로 커맨드"
      style={mentionStyle}
      disabled={disabled}
      inputRef={inputRef as unknown as React.Ref<HTMLInputElement>}
      allowSuggestionsAboveCursor
    >
      <Mention
        trigger="@"
        data={documentSuggestions}
        onAdd={handleDocSelect}
        markup="@[__display__](__id__)"
        displayTransform={(_id, display) => `@${display}`}
        appendSpaceOnAdd
      />
      <Mention
        trigger="/"
        data={commandSuggestions}
        onAdd={handleCommandSelect}
        markup="/[__display__](__id__)"
        displayTransform={(_id, display) => `/${display}`}
        appendSpaceOnAdd
      />
    </MentionsInput>
  );
}
