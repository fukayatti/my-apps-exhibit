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
  tokenizer?: TokenizerConfig; // 追加
}

export type StatusType = "loading" | "success" | "error" | "info" | "progress";

export interface StatusMessage {
  message: string;
  type: StatusType;
  progress?: number; // progressタイプのメッセージ用にオプションで追加
}

export interface AddedToken {
  id: number;
  content: string;
  single_word?: boolean;
  lstrip?: boolean;
  rstrip?: boolean;
  normalized?: boolean;
  special?: boolean;
}

export interface SpecialTokenMapValueObject {
  content: string;
  id?: number;
  single_word?: boolean;
  lstrip?: boolean;
  rstrip?: boolean;
  normalized?: boolean;
  special?: boolean;
}

export type SpecialTokenMapValue = string | SpecialTokenMapValueObject;

export interface TokenizerConfig {
  added_tokens?: AddedToken[];
  special_tokens_map?: Record<string, SpecialTokenMapValue>;
  unk_token?: SpecialTokenMapValue;
  bos_token?: SpecialTokenMapValue;
  eos_token?: SpecialTokenMapValue;
  pad_token?: SpecialTokenMapValue;
  cls_token?: SpecialTokenMapValue;
  sep_token?: SpecialTokenMapValue;
  mask_token?: SpecialTokenMapValue;
  model_max_length?: number;
  padding_side?: string;
  truncation_side?: string;
  clean_up_tokenization_spaces?: boolean;
}
