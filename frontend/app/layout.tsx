import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { Toaster } from '@/components/ui/sonner'
import './globals.css'

const _inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: 'Splatt - Construction Site Memory',
  description: 'Query your helmet cam footage to find past events, locate objects, and recover lost project knowledge.',
}

export const viewport: Viewport = {
  themeColor: '#1e293b',
  userScalable: true,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        {children}
        <Toaster position="bottom-right" />
        <Analytics />
      </body>
    </html>
  )
}
