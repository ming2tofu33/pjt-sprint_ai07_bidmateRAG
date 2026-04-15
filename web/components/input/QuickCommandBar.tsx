"use client";

import { useEffect, useState } from "react";
import { useStore } from "@/store/useStore";
import type { SlashCommandMeta } from "@/lib/types";
import { listCommands } from "@/lib/api";
import { setCachedCommands } from "@/lib/commands";

const PRIMARY_COMMAND_IDS = [
  "요약",
  "요구사항",
  "일정",
  "예산",
  "비교",
  "자격요건",
  "평가기준",
];

export function QuickCommandBar() {
  const [commands, setCommands] = useState<SlashCommandMeta[]>([]);
  const activeCommand = useStore((s) => s.activeCommand);
  const setCommand = useStore((s) => s.setCommand);

  useEffect(() => {
    listCommands()
      .then((data) => {
        setCommands(data.commands);
        setCachedCommands(data.commands);
      })
      .catch((err) => console.error("failed to load commands", err));
  }, []);

  const primary = commands.filter((c) => PRIMARY_COMMAND_IDS.includes(c.id));

  return (
    <div className="flex gap-1 overflow-x-auto px-4 pb-2">
      {primary.map((cmd) => {
        const isActive = activeCommand?.id === cmd.id;
        return (
          <button
            key={cmd.id}
            type="button"
            onClick={() => setCommand(isActive ? null : cmd)}
            className={`whitespace-nowrap rounded-full border px-3 py-1 text-xs transition-colors ${
              isActive
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border text-muted-foreground hover:bg-accent"
            }`}
            title={cmd.description}
          >
            {cmd.icon} {cmd.label}
          </button>
        );
      })}
    </div>
  );
}
