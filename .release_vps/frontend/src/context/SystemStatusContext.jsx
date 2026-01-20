"use client";
import React, { createContext, useContext, useState, useEffect } from 'react';

const SystemStatusContext = createContext();

export function SystemStatusProvider({ children }) {
    const [status, setStatus] = useState('connecting');

    useEffect(() => {
        const checkStatus = async () => {
            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
                const res = await fetch(`${apiUrl}/api/health/`);
                if (res.ok) {
                    setStatus('online');
                } else {
                    setStatus('disconnected');
                }
            } catch (e) {
                setStatus('disconnected');
            }
        };

        // Initial check
        checkStatus();

        // Single interval for the entire application
        const interval = setInterval(checkStatus, 15000);
        return () => clearInterval(interval);
    }, []);

    return (
        <SystemStatusContext.Provider value={{ status }}>
            {children}
        </SystemStatusContext.Provider>
    );
}

export function useSystemStatus() {
    const context = useContext(SystemStatusContext);
    if (context === undefined) {
        throw new Error('useSystemStatus must be used within a SystemStatusProvider');
    }
    return context;
}
