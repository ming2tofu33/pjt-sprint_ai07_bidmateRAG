function formatKoreanTimestamp(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const time = new Intl.DateTimeFormat("ko-KR", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(d);

  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) return `오늘 ${time}`;

  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return `어제 ${time}`;

  return new Intl.DateTimeFormat("ko-KR", {
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(d);
}

export function TimestampDivider({ ts }: { ts: number }) {
  return (
    <div className="flex justify-center py-3 text-[11px] font-medium text-muted-foreground">
      {formatKoreanTimestamp(ts)}
    </div>
  );
}
