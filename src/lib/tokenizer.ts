import { TokenizeResult } from "./types";
import { ModelDownloader } from "./model-downloader";

// トークナイザーのインターフェース
interface TokenizerInterface {
  encode: (text: string) => number[];
  decode: (
    tokenIds: number[],
    options?: {
      skip_special_tokens?: boolean;
      clean_up_tokenization_spaces?: boolean;
    }
  ) => string;
  tokenize?: (text: string) => number[];
  decoderStartToken?: number;
  eosToken?: number;
  padToken?: number;
  unkToken?: number;
  getTokenString?: (id: number) => string | null;
}

export class HuggingFaceTokenizer {
  private tokenizer: TokenizerInterface | null = null;
  private isLoaded = false;
  // private transformers: TransformersModule | null = null; // Transformers.jsは使用しない

  async loadFromPretrained(
    repoId = "alirezamsh/small100",
    filename = "tokenizer.json"
  ): Promise<void> {
    try {
      console.log(`tokenizer.json を ${repoId} から読み込み中...`);

      const downloader = new ModelDownloader(
        `https://huggingface.co/${repoId}/resolve/main/`,
        [{ name: filename, size: "unknown" }]
      );

      const tokenizerBuffer = await downloader.downloadFile(
        filename,
        (loaded, total) => {
          const progress = total ? (loaded / total) * 100 : 0;
          console.log(`tokenizer.json ダウンロード中: ${progress.toFixed(2)}%`);
        }
      );

      if (!tokenizerBuffer) {
        throw new Error("tokenizer.json のダウンロードに失敗しました。");
      }

      // tokenizer.jsonの内容を解析
      const tokenizerConfig = JSON.parse(
        new TextDecoder().decode(tokenizerBuffer)
      );

      // tokenizer.json形式のトークナイザーを実装
      this.tokenizer = new TokenizerJsonProcessor(tokenizerConfig);
      this.isLoaded = true;
      console.log("✓ tokenizer.json からの読み込み完了");
    } catch (error) {
      console.error("Tokenizer読み込みエラー:", error);
      throw new Error(`Tokenizerの読み込みに失敗: ${error}`);
    }
  }

  // loadFromBuffer は tokenizer.json を直接読み込むため、用途に応じて残すか検討
  async loadFromBuffer(tokenizerBuffer: ArrayBuffer): Promise<void> {
    try {
      console.log("生成されたtokenizer.jsonから読み込み中...");

      // tokenizer.jsonの内容を解析
      const tokenizerConfig = JSON.parse(
        new TextDecoder().decode(tokenizerBuffer)
      );

      // tokenizer.json形式のトークナイザーを実装
      this.tokenizer = new TokenizerJsonProcessor(tokenizerConfig);
      this.isLoaded = true;
      console.log("✓ tokenizer.jsonからの読み込み完了");
    } catch (error) {
      console.error("Tokenizer読み込みエラー:", error);
      throw new Error(`tokenizer.jsonの読み込みに失敗: ${error}`);
    }
  }

  async loadFromHuggingFaceFiles(
    vocabBuffer: ArrayBuffer,
    spModelBuffer: ArrayBuffer, // 未使用だが互換性のために残す
    tokenizerConfigBuffer: ArrayBuffer
  ): Promise<void> {
    try {
      console.log(
        "HuggingFaceファイルから読み込み中 (TokenizerJsonProcessorを使用)..."
      );

      // vocab.jsonを解析
      const vocab = JSON.parse(new TextDecoder().decode(vocabBuffer));

      // tokenizer_config.jsonを解析
      const tokenizerConfigData = JSON.parse(
        new TextDecoder().decode(tokenizerConfigBuffer)
      );

      // TokenizerJsonProcessorが必要とする形式に変換
      const tokenizerJson = {
        model: { vocab },
        added_tokens: tokenizerConfigData.added_tokens || [],
        special_tokens_map: tokenizerConfigData.special_tokens_map || {},
        // 他の必要なフィールドもtokenizer_config.jsonからマッピングする
      };

      this.tokenizer = new TokenizerJsonProcessor(tokenizerJson);

      this.isLoaded = true;
      console.log(
        "✓ HuggingFaceファイルからの読み込み完了 (TokenizerJsonProcessor)"
      );
    } catch (error) {
      console.error("Tokenizer読み込みエラー:", error);
      throw new Error(`HuggingFaceファイルの読み込みに失敗: ${error}`);
    }
  }

  tokenize(text: string): TokenizeResult {
    if (!this.isLoaded || !this.tokenizer) {
      throw new Error("Tokenizerが読み込まれていません");
    }

    try {
      let encoded: number[];

      if (this.tokenizer.encode) {
        encoded = this.tokenizer.encode(text);
      } else if (this.tokenizer.tokenize) {
        encoded = this.tokenizer.tokenize(text);
      } else {
        throw new Error("トークナイザーメソッドが見つかりません");
      }

      return {
        input_ids: [Array.isArray(encoded) ? encoded : [encoded]],
      };
    } catch (error) {
      console.error("トークン化エラー:", error);
      throw new Error(`トークン化に失敗: ${error}`);
    }
  }

  decode(tokenIds: number[], skipSpecialTokens = true): string {
    if (!this.isLoaded || !this.tokenizer) {
      throw new Error("Tokenizerが読み込まれていません");
    }

    try {
      if (this.tokenizer.decode) {
        return this.tokenizer.decode(tokenIds, {
          skip_special_tokens: skipSpecialTokens,
        });
      }
      return "";
    } catch (error) {
      console.error("デコードエラー:", error);
      throw new Error(`デコードに失敗: ${error}`);
    }
  }

  get decoderStartToken(): number | null {
    if (!this.tokenizer) return null;
    return this.tokenizer.decoderStartToken || 2;
  }

  get eosToken(): number | null {
    if (!this.tokenizer) return null;
    return this.tokenizer.eosToken || 1;
  }

  get padToken(): number | null {
    if (!this.tokenizer) return null;
    return this.tokenizer.padToken || 0;
  }

  get unkToken(): number | null {
    if (!this.tokenizer) return null;
    return this.tokenizer.unkToken || 3;
  }

  setDecoderStartTokenId(id: number): void {
    if (this.tokenizer) {
      this.tokenizer.decoderStartToken = id;
    }
  }

  setEosTokenId(id: number): void {
    if (this.tokenizer) {
      this.tokenizer.eosToken = id;
    }
  }

  getTokenString(id: number): string | null {
    if (!this.tokenizer) return null;

    try {
      if (this.tokenizer.getTokenString) {
        return this.tokenizer.getTokenString(id);
      }
      return this.tokenizer.decode([id]);
    } catch {
      return null;
    }
  }

  getLangToken(langCode: string): string {
    return `__${langCode}__`;
  }

  getLangId(langCode: string): number | undefined {
    if (!this.tokenizer) return undefined;

    const langToken = this.getLangToken(langCode);
    if (this.tokenizer.encode) {
      const encoded = this.tokenizer.encode(langToken);
      return Array.isArray(encoded) ? encoded[0] : encoded;
    }
    return undefined;
  }
}

// tokenizer.json形式を処理するクラス
class TokenizerJsonProcessor implements TokenizerInterface {
  private vocab: Record<string, number> = {};
  private decoder: Record<number, string> = {};
  private specialTokens: Record<string, number> = {};
  private config: Record<string, unknown>;
  private addedTokens: Array<{
    id: number;
    content: string;
    special: boolean;
  }> = [];

  constructor(config: Record<string, unknown>) {
    this.config = config;
    this.processConfig();
  }

  private processConfig(): void {
    // modelセクションからvocabを取得
    if (
      this.config.model &&
      typeof this.config.model === "object" &&
      this.config.model !== null
    ) {
      const model = this.config.model as Record<string, unknown>;
      if (model.vocab && typeof model.vocab === "object") {
        this.vocab = model.vocab as Record<string, number>;

        // decoderを作成
        for (const [token, id] of Object.entries(this.vocab)) {
          this.decoder[id as number] = token;
        }
      }
    }

    // added_tokensから特殊トークンを取得
    if (this.config.added_tokens && Array.isArray(this.config.added_tokens)) {
      for (const tokenInfo of this.config.added_tokens) {
        if (typeof tokenInfo === "object" && tokenInfo !== null) {
          const token = tokenInfo as {
            id: number;
            content: string;
            special: boolean;
          };
          this.addedTokens.push(token);
          if (token.special) {
            this.specialTokens[token.content] = token.id;
          }
        }
      }
    }

    // special_tokens_mapからも取得 (フォールバックまたは補完)
    if (
      this.config.special_tokens_map &&
      typeof this.config.special_tokens_map === "object" &&
      this.config.special_tokens_map !== null
    ) {
      const specialTokensMap = this.config.special_tokens_map as Record<
        string,
        string
      >;
      for (const tokenName in specialTokensMap) {
        const tokenContent = specialTokensMap[tokenName];
        if (
          this.vocab[tokenContent] !== undefined &&
          this.specialTokens[tokenContent] === undefined
        ) {
          this.specialTokens[tokenContent] = this.vocab[tokenContent];
        }
      }
    }

    // デフォルトの特殊トークン（フォールバック）
    this.specialTokens["<pad>"] =
      this.specialTokens["<pad>"] ?? this.vocab["<pad>"] ?? 0;
    this.specialTokens["</s>"] =
      this.specialTokens["</s>"] ?? this.vocab["</s>"] ?? 1;
    this.specialTokens["<s>"] =
      this.specialTokens["<s>"] ?? this.vocab["<s>"] ?? 2;
    this.specialTokens["<unk>"] =
      this.specialTokens["<unk>"] ?? this.vocab["<unk>"] ?? 3;
  }

  encode(text: string): number[] {
    return this.tokenize(text);
  }

  tokenize(text: string): number[] {
    // 正規化とプリトークン化
    const normalizedText = this.normalize(text);
    const tokens = this.preTokenize(normalizedText);

    // 語彙との照合
    return tokens.map(
      (token) => this.vocab[token] || this.specialTokens["<unk>"] || 3
    );
  }

  private normalize(text: string): string {
    //  をプレフィックスとして追加し、スペースを に置換
    return " " + text.replace(/ /g, " ");
  }

  private preTokenize(text: string): string[] {
    // Metaspaceスタイルのプリトークン化
    const result: string[] = [];
    let i = 0;

    while (i < text.length) {
      let matched = false;

      // 最長マッチを探す
      for (let len = Math.min(text.length - i, 15); len >= 1; len--) {
        // 15は適当な最大トークン長
        const candidate = text.slice(i, i + len);
        if (this.vocab[candidate] !== undefined) {
          result.push(candidate);
          i += len;
          matched = true;
          break;
        }
      }

      if (!matched) {
        // 単一文字もなければ<unk>
        const char = text[i];
        if (this.vocab[char] !== undefined) {
          result.push(char);
        } else {
          result.push("<unk>");
        }
        i++;
      }
    }

    return result;
  }

  decode(
    tokenIds: number[],
    options?: { skip_special_tokens?: boolean }
  ): string {
    const skipSpecialTokens = options?.skip_special_tokens ?? true;

    const tokens = tokenIds
      .map((id) => {
        const token = this.decoder[id];
        if (!token) return "";

        if (
          skipSpecialTokens &&
          Object.values(this.specialTokens).includes(id)
        ) {
          return "";
        }

        return token;
      })
      .filter((token) => token !== "");

    //  をスペースに戻してデコード
    return tokens.join("").replace(/ /g, " ").trim();
  }

  get decoderStartToken(): number {
    return this.specialTokens["<s>"] ?? this.vocab["<s>"] ?? 2; // Ensure fallback
  }

  set decoderStartToken(id: number) {
    this.specialTokens["<s>"] = id;
  }

  get eosToken(): number {
    return this.specialTokens["</s>"] ?? this.vocab["</s>"] ?? 1; // Ensure fallback if not in specialTokens initially
  }

  set eosToken(id: number) {
    this.specialTokens["</s>"] = id;
  }

  get padToken(): number {
    return this.specialTokens["<pad>"] || 0;
  }

  get unkToken(): number {
    return this.specialTokens["<unk>"] || 3;
  }

  getTokenString(id: number): string | null {
    return this.decoder[id] || null;
  }
}

// 後方互換性のためのエイリアス
export const SMALL100TokenizerJS = HuggingFaceTokenizer;
