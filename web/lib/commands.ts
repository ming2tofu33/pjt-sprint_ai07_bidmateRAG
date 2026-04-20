import type { SlashCommandMeta } from "./types";

// 런타임에 /api/commands에서 로딩된 결과가 여기 저장됨
let cachedCommands: SlashCommandMeta[] = [];

export function setCachedCommands(commands: SlashCommandMeta[]): void {
  cachedCommands = commands;
}

export function getCachedCommands(): SlashCommandMeta[] {
  return cachedCommands;
}

export function findCommand(id: string): SlashCommandMeta | undefined {
  return cachedCommands.find((c) => c.id === id);
}
