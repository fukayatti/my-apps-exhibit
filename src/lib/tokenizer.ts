import { TokenizeResult, TextSegment } from "./types";

export class SMALL100TokenizerJS {
  private encoder: Record<string, number> | null = null;
  private decoder: Record<number, string> | null = null;
  private decoderStartTokenId: number | null = null;
  private eosTokenId: number | null = null;
  private padTokenId: number | null = null;
  private unkTokenId: number | null = null;
  private langTokenToId: Record<string, number> = {};
  private idToLangToken: Record<number, string> = {};
  private langCodeToId: Record<string, number> = {};
  private encoderSize = 0;
  private tgtLang = "en";
  private curLangId: number | null = null;
  private numMadeupWords = 8;
  private japaneseSubwords: Record<string, boolean>;
  private spModel: {
    encode: (text: string) => string[];
    decode: (tokens: string[]) => string;
  } | null = null;

  // FAIRSEQ言語コード（簡略版）
  private fairseqLanguageCodes = [
    "af",
    "am",
    "ar",
    "ast",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "br",
    "bs",
    "ca",
    "ceb",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "ff",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "ilo",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "lb",
    "lg",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "ne",
    "nl",
    "no",
    "ns",
    "oc",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "ss",
    "su",
    "sv",
    "sw",
    "ta",
    "th",
    "tl",
    "tn",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "wo",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
  ];

  constructor() {
    this.japaneseSubwords = this.generateJapaneseSubwords();
  }

  private generateJapaneseSubwords(): Record<string, boolean> {
    const subwords: Record<string, boolean> = {};

    // ひらがな
    for (let i = 0x3041; i <= 0x3096; i++) {
      const char = String.fromCharCode(i);
      subwords[`▁${char}`] = true;
      subwords[char] = true;
    }

    // カタカナ
    for (let i = 0x30a1; i <= 0x30f6; i++) {
      const char = String.fromCharCode(i);
      subwords[`▁${char}`] = true;
      subwords[char] = true;
    }

    // 基本的な漢字（常用漢字の一部）
    const commonKanji =
      "日本語英語世界今日天気時間人間社会国家政治経済文化教育学校会社仕事生活家族友達愛情希望未来過去現在";
    for (const char of commonKanji) {
      subwords[`▁${char}`] = true;
      subwords[char] = true;
    }

    return subwords;
  }

  async loadVocab(vocabBuffer: ArrayBuffer): Promise<void> {
    const vocabText = new TextDecoder().decode(vocabBuffer);
    this.encoder = JSON.parse(vocabText);
    this.decoder = {};

    if (this.encoder) {
      for (const [token, id] of Object.entries(this.encoder)) {
        this.decoder[id as number] = token;
      }

      this.encoderSize = Object.keys(this.encoder).length;

      // 言語トークンマッピングを作成
      this.fairseqLanguageCodes.forEach((langCode, i) => {
        const langToken = `__${langCode}__`;
        const langId = this.encoderSize + i;
        this.langTokenToId[langToken] = langId;
        this.idToLangToken[langId] = langToken;
        this.langCodeToId[langCode] = langId;
      });

      this.curLangId = this.getLangId(this.tgtLang) ?? null;

      // 特殊トークンIDを設定
      this.padTokenId = this.encoder["<pad>"] ?? null;
      this.unkTokenId = this.encoder["<unk>"] ?? null;
      this.eosTokenId = this.encoder["</s>"] ?? null;
    }
  }

  async loadSentencePiece(): Promise<void> {
    // SentencePieceの簡易実装
    this.spModel = {
      encode: (text: string) => {
        return this.encodeBPE(text);
      },
      decode: (tokens: string[]) => {
        return tokens.join("").replace(/▁/g, " ").trim();
      },
    };
  }

  private encodeBPE(text: string): string[] {
    const tokens: string[] = [];
    const normalizedText = text.trim();

    // 文字種別ごとに分割
    const segments = this.segmentText(normalizedText);

    for (const segment of segments) {
      if (segment.type === "japanese") {
        tokens.push(...this.encodeJapanese(segment.text));
      } else if (segment.type === "latin") {
        tokens.push(...this.encodeLatin(segment.text));
      } else {
        tokens.push(...this.encodeDefault(segment.text));
      }
    }

    return tokens;
  }

  private segmentText(text: string): TextSegment[] {
    const segments: TextSegment[] = [];
    let current = "";
    let currentType: TextSegment["type"] | null = null;

    for (const char of text) {
      const type = this.getCharType(char);

      if (type !== currentType) {
        if (current) {
          segments.push({ text: current, type: currentType! });
        }
        current = char;
        currentType = type;
      } else {
        current += char;
      }
    }

    if (current) {
      segments.push({ text: current, type: currentType! });
    }

    return segments;
  }

  private getCharType(char: string): TextSegment["type"] {
    const code = char.charCodeAt(0);

    // ひらがな・カタカナ・漢字
    if (
      (code >= 0x3041 && code <= 0x3096) ||
      (code >= 0x30a1 && code <= 0x30f6) ||
      (code >= 0x4e00 && code <= 0x9faf)
    ) {
      return "japanese";
    }

    // 英数字
    if (
      (code >= 0x0041 && code <= 0x005a) ||
      (code >= 0x0061 && code <= 0x007a) ||
      (code >= 0x0030 && code <= 0x0039)
    ) {
      return "latin";
    }

    return "other";
  }

  private encodeJapanese(text: string): string[] {
    const tokens: string[] = [];
    let remaining = "▁" + text;

    while (remaining.length > 0) {
      let found = false;

      // 長いサブワードから順に探す
      for (let len = Math.min(remaining.length, 6); len >= 1; len--) {
        const subword = remaining.substring(0, len);
        if (this.encoder && this.encoder[subword] !== undefined) {
          tokens.push(subword);
          remaining = remaining.substring(len);
          found = true;
          break;
        }
      }

      if (!found) {
        const char = remaining[0];
        if (this.encoder && this.encoder[char] !== undefined) {
          tokens.push(char);
        } else {
          tokens.push("<unk>");
        }
        remaining = remaining.substring(1);
      }
    }

    return tokens;
  }

  private encodeLatin(text: string): string[] {
    const tokens: string[] = [];
    const words = text.toLowerCase().split(/\s+/);

    for (let i = 0; i < words.length; i++) {
      if (words[i].length === 0) continue;

      const wordWithPrefix = i === 0 ? "▁" + words[i] : "▁" + words[i];

      if (this.encoder && this.encoder[wordWithPrefix] !== undefined) {
        tokens.push(wordWithPrefix);
      } else {
        // サブワード分割
        tokens.push(...this.encodeWord(wordWithPrefix));
      }
    }

    return tokens;
  }

  private encodeWord(word: string): string[] {
    const tokens: string[] = [];
    let remaining = word;

    while (remaining.length > 0) {
      let found = false;

      for (let len = Math.min(remaining.length, 10); len >= 1; len--) {
        const subword = remaining.substring(0, len);
        if (this.encoder && this.encoder[subword] !== undefined) {
          tokens.push(subword);
          remaining = remaining.substring(len);
          found = true;
          break;
        }
      }

      if (!found) {
        const char = remaining[0];
        if (this.encoder && this.encoder[char] !== undefined) {
          tokens.push(char);
        } else {
          tokens.push("<unk>");
        }
        remaining = remaining.substring(1);
      }
    }

    return tokens;
  }

  private encodeDefault(text: string): string[] {
    return text.split("").map((char) => {
      return this.encoder && this.encoder[char] !== undefined ? char : "<unk>";
    });
  }

  tokenize(text: string): TokenizeResult {
    if (!this.spModel) {
      throw new Error("SentencePiece model not loaded");
    }

    const tokens = this.spModel.encode(text);
    const inputIds = tokens.map((token: string) => {
      if (token in this.langTokenToId) {
        return this.langTokenToId[token];
      }
      return (this.encoder && this.encoder[token]) || this.unkTokenId || 0;
    });

    return {
      input_ids: [inputIds], // バッチ次元を追加
    };
  }

  decode(tokenIds: number[], skipSpecialTokens = true): string {
    if (!this.spModel || !this.decoder) {
      throw new Error("Tokenizer not fully loaded");
    }

    const tokens = tokenIds
      .map((id) => {
        if (id in this.idToLangToken) {
          return skipSpecialTokens ? "" : this.idToLangToken[id];
        }
        return this.decoder![id] || "<unk>";
      })
      .filter((token) => token !== "");

    return this.spModel.decode(tokens);
  }

  getLangToken(langCode: string): string {
    return `__${langCode}__`;
  }

  getLangId(langCode: string): number | undefined {
    return this.langCodeToId[langCode];
  }

  // ゲッター
  get decoderStartToken(): number | null {
    return this.decoderStartTokenId;
  }

  get eosToken(): number | null {
    return this.eosTokenId;
  }

  get padToken(): number | null {
    return this.padTokenId;
  }

  get unkToken(): number | null {
    return this.unkTokenId;
  }

  // セッター
  setDecoderStartTokenId(id: number): void {
    this.decoderStartTokenId = id;
  }

  setEosTokenId(id: number): void {
    this.eosTokenId = id;
  }
}
