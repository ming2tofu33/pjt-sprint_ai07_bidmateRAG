import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BidMate — RFP 질의응답",
  description: "공공입찰 RFP 분석 AI 어시스턴트",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" className="h-full antialiased">
      <body className="min-h-full bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
