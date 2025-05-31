"use client";

import { useState } from "react";
import { ModelInfo } from "@/lib/types";

interface CacheManagementProps {
  isVisible: boolean;
  onClearCache: () => Promise<void>;
  onShowCacheInfo: () => Promise<ModelInfo[]>;
}

export function CacheManagement({
  isVisible,
  onClearCache,
  onShowCacheInfo,
}: CacheManagementProps) {
  const [cacheInfo, setCacheInfo] = useState<ModelInfo[] | null>(null);
  const [showCacheInfo, setShowCacheInfo] = useState(false);

  if (!isVisible) return null;

  const handleClearCache = async () => {
    if (
      confirm(
        "キャッシュされたモデルファイルをすべて削除しますか？\n次回使用時に再ダウンロードが必要になります。"
      )
    ) {
      await onClearCache();
      setCacheInfo(null);
      setShowCacheInfo(false);
    }
  };

  const handleShowCacheInfo = async () => {
    if (showCacheInfo) {
      setShowCacheInfo(false);
    } else {
      const info = await onShowCacheInfo();
      setCacheInfo(info);
      setShowCacheInfo(true);
    }
  };

  const formatFileSize = (bytes: number): string => {
    return (bytes / (1024 * 1024)).toFixed(1) + "MB";
  };

  const formatDate = (timestamp: number): string => {
    return new Date(timestamp).toLocaleString("ja-JP");
  };

  const getTotalSize = (): number => {
    if (!cacheInfo) return 0;
    return cacheInfo.reduce((total, info) => total + info.size, 0);
  };

  return (
    <div>
      <h4 className="text-lg font-semibold mb-3">💾 キャッシュ管理:</h4>
      <div className="flex gap-3 mb-3">
        <button
          onClick={handleClearCache}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors text-sm"
        >
          🗑️ キャッシュクリア
        </button>
        <button
          onClick={handleShowCacheInfo}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors text-sm"
        >
          📊 キャッシュ情報
        </button>
      </div>

      {showCacheInfo && (
        <div className="bg-gray-50 p-4 rounded-md text-sm">
          {!cacheInfo || cacheInfo.length === 0 ? (
            <p>キャッシュされたモデルはありません。</p>
          ) : (
            <div>
              <p className="font-semibold mb-2">キャッシュされたファイル:</p>
              {cacheInfo.map((info) => (
                <div key={info.filename} className="mb-1">
                  • {info.filename}: {formatFileSize(info.size)} (
                  {formatDate(info.timestamp)})
                </div>
              ))}
              <div className="mt-3 pt-2 border-t border-gray-300">
                <strong>合計サイズ:</strong> {formatFileSize(getTotalSize())}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
