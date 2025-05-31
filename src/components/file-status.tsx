"use client";

import { ModelFile } from "@/lib/types";

interface FileStatusProps {
  files: ModelFile[];
  fileStatuses: Record<
    string,
    { progress: number; size?: number; completed: boolean }
  >;
  isVisible: boolean;
}

export function FileStatus({
  files,
  fileStatuses,
  isVisible,
}: FileStatusProps) {
  if (!isVisible) return null;

  const formatFileSize = (bytes: number): string => {
    return (bytes / (1024 * 1024)).toFixed(1) + "MB";
  };

  return (
    <div>
      <h4 className="text-sm font-semibold mb-2">📋 ファイル状況:</h4>
      {files.map((file) => {
        const status = fileStatuses[file.name] || {
          progress: 0,
          completed: false,
        };

        return (
          <div
            key={file.name}
            className={`flex justify-between items-center p-2 mb-1 rounded text-sm ${
              status.completed ? "bg-green-100" : "bg-gray-100"
            }`}
          >
            <span>{file.name}</span>
            <span>
              {status.completed && status.size
                ? `✅ ${formatFileSize(status.size)}`
                : status.progress > 0
                ? `${Math.round(status.progress)}%`
                : "待機中..."}
            </span>
          </div>
        );
      })}
    </div>
  );
}
