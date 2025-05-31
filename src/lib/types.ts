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
  model_name: string;
  vocab_size: number;
  num_beams: number;
  max_length: number;
  architectures: string[];
}

export interface DownloadProgress {
  file: string; // currentFile -> file
  loaded: number;
  total: number;
  percentage: number; // progress -> percentage
  // fileProgress は削除 (percentageで代替)
}

export interface TextSegment {
  text: string;
  type: "japanese" | "latin" | "other";
}

export interface TokenizeResult {
  input_ids: number[][];
}

export interface TranslationConfig {
  _name_or_path?: string;
  architectures?: string[];
  decoder_start_token_id: number;
  eos_token_id: number;
  pad_token_id?: number;
  unk_token_id?: number;
  vocab_size: number;
  num_beams?: number;
  max_length?: number;
  // 他にもconfig.jsonに含まれる可能性のあるフィールドを追加
  model_type?: string;
  forced_bos_token_id?: number;
}

export type StatusType = "loading" | "success" | "error" | "info" | "progress";

export interface StatusMessage {
  message: string;
  type: StatusType;
  progress?: number; // progressタイプのメッセージ用にオプションで追加
}
