"use client";
import React from 'react';

export default function StatCard({ title, value, icon: Icon, trend }) {
  return (
    <div className="rounded-xl border bg-card text-card-foreground shadow-sm p-6">
      <div className="flex flex-row items-center justify-between space-y-0 pb-2">
        <h3 className="tracking-tight text-sm font-medium">{title}</h3>
        {Icon && <Icon className="h-4 w-4 text-muted-foreground" />}
      </div>
      <div className="flex flex-col">
        <div className="text-2xl font-bold">{value}</div>
        {trend && (
          <p className={`text-xs ${trend > 0 ? 'text-green-500' : 'text-red-500'} mt-1`}>
            {trend > 0 ? '+' : ''}{trend}% from last session
          </p>
        )}
      </div>
    </div>
  );
}
