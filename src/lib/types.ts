// 翻訳システムの型定義

export interface ModelFile {
  name: string;
  size: string;
}

export interface FileStatus {
  filename: string;
  size: number;
  timestamp: number;
}

export interface ModelInfo {
  filename: string;
  size: number;
  timestamp: number;
}

export interface DownloadProgress {
  progress: number;
  currentFile: string;
  fileProgress: number;
}

export interface TextSegment {
  text: string;
  type: "japanese" | "latin" | "other";
}

export interface TokenizeResult {
  input_ids: number[][];
}

export interface TranslationConfig {
  decoder_start_token_id: number;
  eos_token_id: number;
  pad_token_id?: number;
  unk_token_id?: number;
}

export type StatusType = "loading" | "success" | "error";

export interface StatusMessage {
  message: string;
  type: StatusType;
}
