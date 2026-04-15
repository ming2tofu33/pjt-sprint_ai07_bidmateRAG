"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { ReactNode } from "react";

/** 답변 본문에서 [숫자] 패턴을 감지해 <a href="#cite-n">으로 치환. */
function renderTextWithCitations(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const regex = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let i = 0;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const num = match[1];
    parts.push(
      <a
        key={`cite-link-${i++}`}
        href={`#cite-${num}`}
        className="mx-0.5 inline-flex items-center rounded bg-[color-mix(in_oklab,var(--imessage-blue)_15%,transparent)] px-1 text-[11px] font-semibold text-[var(--imessage-blue)] hover:bg-[color-mix(in_oklab,var(--imessage-blue)_25%,transparent)]"
      >
        [{num}]
      </a>
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts.length > 0 ? parts : [text];
}

function traverseAndReplace(children: ReactNode): ReactNode {
  if (typeof children === "string") {
    return renderTextWithCitations(children);
  }
  if (Array.isArray(children)) {
    return children.map((child, idx) => (
      <span key={idx}>{traverseAndReplace(child)}</span>
    ));
  }
  return children;
}

const components: Components = {
  p: ({ children }) => <p className="mb-3 leading-relaxed">{traverseAndReplace(children)}</p>,
  li: ({ children }) => <li className="ml-5 list-disc">{traverseAndReplace(children)}</li>,
  td: ({ children }) => <td className="border border-border px-2 py-1">{traverseAndReplace(children)}</td>,
  th: ({ children }) => (
    <th className="border border-border bg-muted px-2 py-1 font-semibold">{children}</th>
  ),
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto">
      <table className="min-w-full border-collapse border border-border text-sm">
        {children}
      </table>
    </div>
  ),
};

export function MarkdownWithCitations({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
