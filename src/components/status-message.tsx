"use client";

import { StatusMessage } from "@/lib/types";

interface StatusMessageProps {
  status: StatusMessage | null;
}

export function StatusMessageComponent({ status }: StatusMessageProps) {
  if (!status) return null;

  const getStatusStyles = (type: string) => {
    switch (type) {
      case "loading":
        return "bg-yellow-50 border-yellow-200 text-yellow-800";
      case "success":
        return "bg-green-50 border-green-200 text-green-800";
      case "error":
        return "bg-red-50 border-red-200 text-red-800";
      default:
        return "bg-gray-50 border-gray-200 text-gray-800";
    }
  };

  return (
    <div
      className={`text-center p-3 my-3 border rounded-md ${getStatusStyles(
        status.type
      )}`}
    >
      {status.message}
    </div>
  );
}
