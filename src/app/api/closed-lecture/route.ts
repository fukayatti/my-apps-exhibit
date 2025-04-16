// src/app/api/closed-lecture/route.ts
import { Hono, Context } from "hono";
import { prettyJSON } from "hono/pretty-json";
import { load } from "cheerio";

// 全角文字を半角に変換する関数
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

// 空白（改行・複数スペース）を単一のスペースに統一する
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
// 例: "☆専攻科１年　５・６限　応用制御工学（平澤）⇒７・８限へ"
const entryRegex: RegExp =
  /^(?<symbol>[◉◎◇☆])\s*(?<class>[^\s]+)\s+(?<period>\d+・\d+限)\s+(?<rest>.+)$/;
// 「⇒」 後が「数字・数字限」または「数字・数字限へ」かチェックする正規表現
const periodRegex: RegExp = /^(?<period2>\d+・\d+限)(?:へ)?$/;

// 型定義
type Entry = {
  type: string;
  class: string;
  period1: string;
  period2: string;
  subject1: string | null;
  subject2: string | null;
};

type DateEntry = {
  date: string;
  entries: Entry[];
};

type FinalOutput = {
  休講情報: DateEntry[];
};

const app = new Hono();

// prettyJSON ミドルウェアで整形済みの JSON レスポンスを返す
app.use("*", prettyJSON());

app.get("/api/closed-lecture", async (c: Context) => {
  try {
    // 指定 URL から HTML を取得（グローバル fetch を使用）
    const url = "https://www.ibaraki-ct.ac.jp/info/archives/65544";
    const response = await fetch(url);
    const html: string = await response.text();

    // Cheerio により HTML をパース
    const $ = load(html);
    // 休講情報の本文は id="post_main" 内にあると仮定
    const $postMain = $("#post_main");
    if (!$postMain.length) {
      return c.json({ error: "post_main が見つかりません" }, 404);
    }

    // 結果を日付ごとに格納するオブジェクト
    const results: { [date: string]: Entry[] } = {};
    let currentDate = "";

    // $postMain 内の全ての <p> タグを走査
    $postMain.find("p").each((_, elem) => {
      const $p = $(elem);
      let text: string = $p.text();
      text = fullwidthToHalfwidth(text);
      text = normalizeWhitespace(text);

      // <mark> タグが含まれている段落は日付情報として扱う
      if ($p.find("mark").length > 0) {
        currentDate = text;
        if (!results[currentDate]) results[currentDate] = [];
      } else {
        if (!currentDate) return;
        const match = entryRegex.exec(text);
        if (match && match.groups) {
          const { symbol, class: classText, period, rest } = match.groups;
          const typeField: string = symbolMap(symbol);
          let subject1: string | null = null;
          let subject2: string | null = null;
          let period2: string | null = null;

          if (rest.includes("⇒")) {
            const parts: string[] = rest.split("⇒", 2).map((s) => s.trim());
            subject1 = parts[0] === "授業なし" ? null : parts[0];
            const periodMatch = periodRegex.exec(parts[1]);
            if (
              periodMatch &&
              periodMatch.groups &&
              periodMatch.groups.period2
            ) {
              period2 = periodMatch.groups.period2.trim();
              subject2 = null;
            } else {
              subject2 = parts[1] === "授業なし" ? null : parts[1];
            }
          } else {
            subject1 = rest === "授業なし" ? null : rest;
          }
          // period2 が抽出できなかった場合は、必ず period1 の値を代入する
          if (!period2 || period2 === "") {
            period2 = period.trim();
          }

          const entry: Entry = {
            type: typeField,
            class: classText.trim(),
            period1: period.trim(),
            period2,
            subject1,
            subject2,
          };
          results[currentDate].push(entry);
        }
      }
    });

    // 結果を "休講情報" キーの下に、日付順のオブジェクト配列に整形
    const finalOutput: FinalOutput = {
      休講情報: Object.keys(results)
        .sort()
        .map((date) => ({
          date,
          entries: results[date],
        })),
    };

    return c.json(finalOutput);
  } catch (error: any) {
    console.error("Error:", error);
    return c.json({ error: error.message }, 500);
  }
});

// HTTP GET メソッドとして名前付きエクスポート（Next.js App Router 用）
export const GET = app.fetch;
