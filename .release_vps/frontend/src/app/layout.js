import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { Sidebar } from "@/components/Sidebar";
import { MobileHeader } from "@/components/MobileHeader";
import { SystemStatusProvider } from "@/context/SystemStatusContext";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Trading Analytics Pro",
  description: "ML-Powered Trading Dashboard",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex flex-col md:flex-row min-h-screen">
            <SystemStatusProvider>
              <Sidebar />
              <div className="flex flex-col flex-1 min-h-screen">
                <MobileHeader />
                <main className="flex-1 bg-background text-foreground">
                  {children}
                </main>
              </div>
            </SystemStatusProvider>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
