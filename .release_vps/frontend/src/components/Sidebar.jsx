"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, ShieldAlert, LineChart, Code2, History, Wifi, WifiOff, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";
import { useSystemStatus } from "@/context/SystemStatusContext";

export function Sidebar() {
    const pathname = usePathname();
    const { status } = useSystemStatus();

    const links = [
        { href: "/", label: "Dashboard", icon: LayoutDashboard },
        { href: "/risk", label: "Risk & Backtest", icon: ShieldAlert },
        { href: "/dashboard/test-manual", label: "Test Manual", icon: History },
    ];

    return (
        <aside className="w-64 border-r bg-card text-card-foreground hidden md:flex flex-col h-screen sticky top-0">
            <div className="p-6 border-b">
                <h1 className="text-xl font-bold flex items-center gap-2">
                    <LineChart className="w-6 h-6 text-primary" />
                    <span>TradeBot Pro</span>
                </h1>
            </div>

            <nav className="flex-1 p-4 space-y-2">
                {links.map((link) => {
                    const Icon = link.icon;
                    const isActive = pathname === link.href;
                    return (
                        <Link
                            key={link.href}
                            href={link.href}
                            className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors
                ${isActive
                                    ? 'bg-primary text-primary-foreground'
                                    : 'hover:bg-muted text-muted-foreground hover:text-foreground'
                                }`}
                        >
                            <Icon className="w-5 h-5" />
                            <span className="font-medium">{link.label}</span>
                        </Link>
                    );
                })}
            </nav>

            <div className="p-4 border-t">
                <div className="p-4 bg-muted/50 rounded-xl border border-border/50">
                    <p className="text-[10px] text-muted-foreground font-black uppercase tracking-widest mb-2">System Status</p>
                    <div className={`flex items-center gap-2 text-sm font-bold transition-colors
                        ${status === 'online' ? 'text-green-500' : status === 'disconnected' ? 'text-red-500' : 'text-yellow-500'}
                    `}>
                        {status === 'online' && (
                            <>
                                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                                <span>Online</span>
                            </>
                        )}
                        {status === 'disconnected' && (
                            <>
                                <WifiOff className="w-3 h-3" />
                                <span>Disconnected</span>
                            </>
                        )}
                        {status === 'connecting' && (
                            <>
                                <Loader2 className="w-3 h-3 animate-spin" />
                                <span>Connecting</span>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </aside>
    );
}
