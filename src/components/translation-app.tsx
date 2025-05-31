"use client";

import { useState, useEffect, useRef } from "react";
import { TranslationSystem } from "@/lib/translation-system";
import { StatusMessage, ModelFile, FileStatus } from "@/lib/types";
import { FileStatus as FileStatusComponent } from "./file-status";
import { ProgressBar } from "./progress-bar";
import { StatusMessageComponent } from "./status-message";
import { CacheManagement } from "./cache-management";

export function TranslationApp() {
  const [japaneseText, setJapaneseText] = useState(
    "こんにちは、世界！今日はとても良い天気ですね。"
  );
  const [englishText, setEnglishText] = useState("");
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [progress, setProgress] = useState(0);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [showFileStatus, setShowFileStatus] = useState(false);
  const [showCacheManagement, setShowCacheManagement] = useState(false);
  const [fileStatuses, setFileStatuses] = useState<
    Record<string, { progress: number; size?: number; completed: boolean }>
  >({});

  const translationSystemRef = useRef<TranslationSystem | null>(null);
  const modelFiles: ModelFile[] = [
    { name: "model.onnx", size: "~150MB" },
    { name: "vocab.json", size: "~2MB" },
    { name: "sentencepiece.bpe.model", size: "~800KB" },
    { name: "config.json", size: "~1KB" },
  ];

  useEffect(() => {
    translationSystemRef.current = new TranslationSystem();

    // 初期化時にキャッシュ状況をチェック
    checkCacheStatus();
  }, []);

  const checkCacheStatus = async () => {
    if (!translationSystemRef.current) return;

    try {
      const cacheInfo = await translationSystemRef.current.getCacheInfo();
      if (cacheInfo.length > 0) {
        setShowCacheManagement(true);
      }
    } catch (error) {
      console.warn("キャッシュ初期化に失敗:", error);
    }
  };

  const handleDownloadModel = async () => {
    if (!translationSystemRef.current || isDownloading) return;

    setIsDownloading(true);
    setShowFileStatus(true);
    setProgress(0);

    try {
      console.log("[TranslationApp] handleDownloadModel - 開始");
      await translationSystemRef.current.downloadAndLoadModel(
        (statusMsg) => {
          // console.log("[TranslationApp] Status update:", statusMsg);
          setStatus(statusMsg);
        },
        (overallProgress) => {
          // console.log("[TranslationApp] Overall progress update:", overallProgress);
          setProgress(overallProgress);
        },
        (filename, fileSpecificProgress, loadedBytes, totalBytes) => {
          // console.log(`[TranslationApp] File progress: ${filename} - ${fileSpecificProgress}% (${loadedBytes}/${totalBytes})`);
          setFileStatuses((prev) => {
            const newStatuses = {
              ...prev,
              [filename]: {
                progress: fileSpecificProgress,
                loaded: loadedBytes,
                size: totalBytes, // totalBytes が0の場合もあるので注意 (content-lengthがない場合など)
                completed: fileSpecificProgress === 100,
              },
            };
            // console.log("[TranslationApp] Updated fileStatuses:", newStatuses);
            return newStatuses;
          });
        },
        (filename, size) => {
          console.log(
            `[TranslationApp] File complete: ${filename} - ${size} bytes`
          );
          setFileStatuses((prev) => {
            const newStatuses = {
              ...prev,
              [filename]: {
                progress: 100,
                loaded: size,
                size,
                completed: true,
              },
            };
            // console.log("[TranslationApp] Updated fileStatuses (complete):", newStatuses);
            return newStatuses;
          });
        }
      );
      console.log("[TranslationApp] handleDownloadModel - モデル読み込み成功");
      setIsModelLoaded(true);
      setShowCacheManagement(true);
    } catch (error) {
      console.error("[TranslationApp] モデル読み込みエラー:", error);
      setStatus({
        message: `モデル読み込みエラー: ${
          error instanceof Error ? error.message : String(error)
        }`,
        type: "error",
      });
    } finally {
      console.log("[TranslationApp] handleDownloadModel - 終了処理");
      setIsDownloading(false);
    }
  };

  const handleTranslate = async () => {
    if (!translationSystemRef.current || !isModelLoaded || isTranslating)
      return;

    const inputText = japaneseText.trim();
    if (!inputText) {
      alert("翻訳する日本語テキストを入力してください");
      return;
    }

    setIsTranslating(true);
    setStatus({ message: "翻訳中...", type: "loading" });

    try {
      const result = await translationSystemRef.current.translate(
        inputText,
        "jpn",
        "eng"
      );
      setEnglishText(result);
      setStatus({ message: "翻訳が完了しました！", type: "success" });
    } catch (error) {
      console.error("翻訳エラー:", error);
      setStatus({
        message: `翻訳エラー: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        type: "error",
      });
    } finally {
      setIsTranslating(false);
    }
  };

  const handleClearCache = async () => {
    if (!translationSystemRef.current) return;

    try {
      await translationSystemRef.current.clearCache();
      setStatus({ message: "キャッシュをクリアしました", type: "success" });
      setIsModelLoaded(false);
      setShowCacheManagement(false);
      setFileStatuses({});
    } catch (error) {
      console.error("キャッシュクリアエラー:", error);
      setStatus({ message: "キャッシュクリアに失敗しました", type: "error" });
    }
  };

  const handleShowCacheInfo = async (): Promise<FileStatus[]> => {
    if (!translationSystemRef.current) return [];
    return await translationSystemRef.current.getCacheInfo();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.ctrlKey && e.key === "Enter" && isModelLoaded && !isTranslating) {
      handleTranslate();
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white p-8 rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-8">
          🌐 完全ローカル日英翻訳システム (8bit量子化版)
        </h1>

        <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 rounded-md mb-6">
          <strong>⚠️ 注意:</strong>
          初回使用時は8bit量子化モデルファイル（約200MB）のダウンロードが必要です。高速なインターネット接続を推奨します。
        </div>

        <div className="bg-gray-100 p-6 rounded-md mb-6">
          <h3 className="text-xl font-semibold mb-4">
            📦 モデルの自動ダウンロード
          </h3>

          <div className="bg-gray-50 p-4 rounded-md mb-4">
            <strong>SMALL-100 多言語翻訳モデル (8bit量子化版)</strong>
            <br />
            リポジトリ: fukayatti0/small100-quantized-int8
            <br />
            サイズ: 約200MB（4ファイル合計）
            <br />
            <small className="text-gray-600">
              ※ 8bit量子化により、元モデルより高速で軽量になっています
            </small>
          </div>

          <div className="bg-blue-50 p-3 rounded-md mb-4">
            <strong>ステップ1:</strong>{" "}
            以下のボタンでモデルファイルを自動ダウンロード
          </div>

          <button
            onClick={handleDownloadModel}
            disabled={isDownloading || isModelLoaded}
            className={`w-full py-3 px-6 rounded-md text-white font-semibold transition-colors ${
              isModelLoaded
                ? "bg-green-500 cursor-not-allowed"
                : isDownloading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-500 hover:bg-blue-600"
            }`}
          >
            {isModelLoaded
              ? "✅ モデル読み込み完了"
              : isDownloading
              ? "⏳ ダウンロード中..."
              : "🚀 モデルを自動ダウンロード"}
          </button>

          <ProgressBar
            progress={progress}
            isVisible={isDownloading || progress > 0}
          />

          <FileStatusComponent
            files={modelFiles}
            fileStatuses={fileStatuses}
            isVisible={showFileStatus}
          />

          <div className="mt-6">
            <CacheManagement
              isVisible={showCacheManagement}
              onClearCache={handleClearCache}
              onShowCacheInfo={handleShowCacheInfo}
            />
          </div>
        </div>

        <div className="mb-6">
          <label
            htmlFor="japanese-text"
            className="block text-sm font-semibold text-gray-700 mb-2"
          >
            日本語テキスト:
          </label>
          <textarea
            id="japanese-text"
            value={japaneseText}
            onChange={(e) => setJapaneseText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="翻訳したい日本語文を入力してください..."
            className="w-full h-32 p-3 border-2 border-gray-300 rounded-md focus:border-blue-500 focus:outline-none resize-vertical"
          />
        </div>

        <button
          onClick={handleTranslate}
          disabled={!isModelLoaded || isTranslating}
          className={`w-full py-3 px-6 rounded-md text-white font-semibold mb-4 transition-colors ${
            !isModelLoaded || isTranslating
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-500 hover:bg-blue-600"
          }`}
        >
          {isTranslating ? "翻訳中..." : "翻訳する"}
        </button>

        <StatusMessageComponent status={status} />

        <div className="mb-6">
          <label
            htmlFor="english-text"
            className="block text-sm font-semibold text-gray-700 mb-2"
          >
            英語翻訳:
          </label>
          <textarea
            id="english-text"
            value={englishText}
            readOnly
            placeholder="翻訳結果がここに表示されます..."
            className="w-full h-32 p-3 border-2 border-gray-300 rounded-md bg-gray-50 resize-vertical"
          />
        </div>
      </div>
    </div>
  );
}
