import { Hono, Context } from "hono";
import { handle } from "hono/vercel";
import { prettyJSON } from "hono/pretty-json";
import { load } from "cheerio";

export const runtime = "edge";

// 全角文字を半角に変換する
function fullwidthToHalfwidth(str: string): string {
  return str
    .split("")
    .map((c) => {
      if (c === "\u3000") return " ";
      const code = c.charCodeAt(0);
      if (code >= 0xff01 && code <= 0xff5e) {
        return String.fromCharCode(code - 0xfee0);
      }
      return c;
    })
    .join("");
}

// 改行・連続スペースを1スペースに統一する
function normalizeWhitespace(str: string): string {
  return str.replace(/\s+/g, " ").trim();
}

// シンボルのマッピング：◉→休講、◎→補講、◇→遠隔授業、☆→授業・教室変更
function symbolMap(symbol: string): string {
  switch (symbol) {
    case "◉":
      return "休講";
    case "◎":
      return "補講";
    case "◇":
      return "遠隔授業";
    case "☆":
      return "授業・教室変更";
    default:
      return "";
  }
}

// エントリー行解析用正規表現
const entryRegex =
  /^(?<symbol>[◉◎◇☆])\s*(?<class>[^\s]+)\s+(?<period>\d+・\d+限)\s+(?<rest>.+)$/;

// "⇒" 後が「数字・数字限」または「数字・数字限へ」かを判定する正規表現
const periodRegex = /^(?<period2>\d+・\d+限)(?:へ)?$/;

// 型定義
interface Entry {
  type: string;
  class: string;
  period: string;
  subject1: string | null;
  subject2: string | null;
  period2: string | null;
}

interface DateEntry {
  date: string;
  entries: Entry[];
}

interface FinalOutput {
  休講情報: DateEntry[];
}

// アプリ作成
const app = new Hono();

// prettyJSON ミドルウェアを全ルートに適用
app.use("*", prettyJSON());

// メインルート
app.get("/api/closed-lecture", async (c: Context) => {
  try {
    const url = "https://www.ibaraki-ct.ac.jp/info/archives/65544";
    const response = await fetch(url);
    const html: string = await response.text();

    const $ = load(html);
    const $postMain = $("#post_main");
    if (!$postMain.length) {
      return c.json({ error: "post_main が見つかりません" }, 404);
    }

    const results: { [date: string]: Entry[] } = {};
    let currentDate = "";

    $postMain.find("p").each((_, elem) => {
      const $p = $(elem);
      let text: string = $p.text();
      text = normalizeWhitespace(fullwidthToHalfwidth(text));

      if ($p.find("mark").length > 0) {
        currentDate = text;
        if (!results[currentDate]) {
          results[currentDate] = [];
        }
      } else {
        if (!currentDate) return;

        const match = entryRegex.exec(text);
        if (match && match.groups) {
          const { symbol, class: classText, period, rest } = match.groups;

          const typeField = symbolMap(symbol);
          let subject1: string | null = null;
          let subject2: string | null = null;
          let period2: string | null = null;

          if (rest.includes("⇒")) {
            const parts = rest.split("⇒", 2).map((s) => s.trim());
            subject1 = parts[0] === "授業なし" ? null : parts[0];

            const periodMatch = periodRegex.exec(parts[1]);
            if (periodMatch?.groups?.period2) {
              period2 = periodMatch.groups.period2.trim();
              subject2 = null;
            } else {
              subject2 = parts[1] === "授業なし" ? null : parts[1];
            }
          } else {
            subject1 = rest === "授業なし" ? null : rest;
          }

          const entry: Entry = {
            type: typeField,
            class: classText.trim(),
            period: period.trim(),
            subject1,
            subject2,
            period2,
          };

          if (!results[currentDate]) {
            results[currentDate] = [];
          }
          results[currentDate].push(entry);
        }
      }
    });

    const finalOutput: FinalOutput = {
      休講情報: Object.keys(results)
        .sort()
        .map((date) => ({
          date,
          entries: results[date],
        })),
    };

    return c.json(finalOutput);
  } catch (error: unknown) {
    console.error("Error:", error);
    const message = error instanceof Error ? error.message : String(error);
    return c.json({ error: message }, 500);
  }
});

// Edge Runtime 用の GET エクスポート
export const GET = handle(app);
