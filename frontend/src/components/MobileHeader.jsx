"use client";
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X, LineChart, LayoutDashboard, ShieldAlert, History, WifiOff, Loader2, Brain } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';
import { useSystemStatus } from '@/context/SystemStatusContext';

export function MobileHeader() {
    const [isOpen, setIsOpen] = useState(false);
    const pathname = usePathname();
    const { status } = useSystemStatus();

    const links = [
        { href: "/", label: "Dashboard", icon: LayoutDashboard },
        { href: "/risk", label: "Risk & Backtest", icon: ShieldAlert },
        { href: "/dashboard/trainAlgo", label: "Train Neural Net", icon: Brain },
        { href: "/dashboard/test-manual", label: "Test Manual", icon: History },
    ];

    const toggleMenu = () => setIsOpen(!isOpen);

    return (
        <header className="md:hidden bg-card border-b sticky top-0 z-50 px-4 py-3">
            <div className="flex items-center justify-between">
                <Link href="/" className="flex items-center gap-2">
                    <LineChart className="w-6 h-6 text-primary" />
                    <span className="font-bold text-lg">TradeBot Pro</span>
                </Link>

                <div className="flex items-center gap-2">
                    <ThemeToggle />
                    <button
                        onClick={toggleMenu}
                        className="p-2 hover:bg-muted rounded-lg transition-colors"
                        aria-label="Toggle Menu"
                    >
                        {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                    </button>
                </div>
            </div>

            {/* Mobile Menu Overlay */}
            {isOpen && (
                <div className="fixed inset-0 top-[57px] bg-background z-40 animate-in fade-in slide-in-from-top-4 duration-200">
                    <nav className="p-4 space-y-2">
                        {links.map((link) => {
                            const Icon = link.icon;
                            const isActive = pathname === link.href;
                            return (
                                <Link
                                    key={link.href}
                                    href={link.href}
                                    onClick={() => setIsOpen(false)}
                                    className={`flex items-center gap-3 px-4 py-4 rounded-xl transition-colors
                                        ${isActive
                                            ? 'bg-primary text-primary-foreground'
                                            : 'hover:bg-muted text-muted-foreground hover:text-foreground'
                                        }`}
                                >
                                    <Icon className="w-5 h-5" />
                                    <span className="font-bold">{link.label}</span>
                                </Link>
                            );
                        })}
                    </nav>

                    <div className="absolute bottom-10 left-4 right-4 p-4 bg-muted/50 rounded-2xl border border-border/50">
                        <p className="text-[10px] text-muted-foreground font-black uppercase tracking-widest mb-2">System Status</p>
                        <div className={`flex items-center gap-2 text-sm font-bold transition-colors
                            ${status === 'online' ? 'text-green-500' : status === 'disconnected' ? 'text-red-500' : 'text-yellow-500'}
                        `}>
                            {status === 'online' && (
                                <>
                                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                                    <span>Online (Mobile)</span>
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
            )}
        </header>
    );
}
