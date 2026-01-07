import type { Metadata } from "next";
import { Archivo, DM_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const archivo = Archivo({
  subsets: ["latin"],
  variable: "--font-archivo",
  display: "swap",
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Kettlebell Counter | Competition Rep Tracking",
  description:
    "Accurate rep counting and form analysis for kettlebell sport athletes and coaches",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${archivo.variable} ${dmSans.variable} ${jetbrains.variable}`}
    >
      <body className="bg-kb-bg text-white font-body antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}

