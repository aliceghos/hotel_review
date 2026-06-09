import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "花园酒店住客评论",
  description: "花园酒店住客评论浏览系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col" style={{ background: '#faf8f5', color: '#2c2416' }}>
        {children}
      </body>
    </html>
  );
}
