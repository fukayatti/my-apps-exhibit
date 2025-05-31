// src/app/page.tsx
"use client";

import { useState, useEffect } from "react";
import { parse } from "mathjs";
import { BlockMath } from "react-katex";

export default function SigmaCalculatorPage() {
  const [from, setFrom] = useState<number>(1);
  const [toInput, setToInput] = useState<string>("10");
  const [expression, setExpression] = useState<string>("k");
  const [result, setResult] = useState<number | null>(0);
  const [error, setError] = useState<string>("");
  const [latex, setLatex] = useState<string>("");

  const isSymbolic = toInput.trim().toLowerCase() === "n";
  const toValue = isSymbolic ? null : parseInt(toInput, 10);

  useEffect(() => {
    try {
      if (isSymbolic) {
        // 上限が n の場合：一般式を表示
        setLatex(`\\displaystyle \\sum_{k=${from}}^{n} ${expression}`);
        setResult(null);
        setError("");
      } else {
        // 数値計算
        setError("");
        const node = parse(expression);
        let sum = 0;
        for (let k = from; k <= toValue!; k++) {
          const val = node.evaluate({ k });
          if (typeof val !== "number" || !isFinite(val)) {
            throw new Error(`k=${k} の評価が無効です: ${val}`);
          }
          sum += val;
        }
        setResult(sum);
        setLatex(`\\displaystyle \\sum_{k=${from}}^{${toValue}} ${expression}`);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "エラーが発生しました");
      setResult(null);
      setLatex("");
    }
  }, [from, toInput, expression, isSymbolic, toValue]);

  return (
    <div className="flex items-center justify-center py-12 px-4 bg-gray-100 min-h-screen">
      <div className="w-full max-w-md bg-white shadow-lg rounded-xl p-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Σ 計算機
        </h1>

        {/* 下限 */}
        <label className="block mb-4">
          <span className="text-gray-700">下限 (k の開始値)</span>
          <input
            type="number"
            value={from}
            onChange={(e) => setFrom(parseInt(e.target.value, 10) || 0)}
            className="mt-1 w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>

        {/* 上限 */}
        <label className="block mb-4">
          <span className="text-gray-700">上限 (k の終了値、数値 or n)</span>
          <input
            type="text"
            value={toInput}
            onChange={(e) => setToInput(e.target.value)}
            className="mt-1 w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>

        {/* 式 */}
        <label className="block mb-6">
          <span className="text-gray-700">項の式 (例: k, k^2+3)</span>
          <input
            type="text"
            value={expression}
            onChange={(e) => setExpression(e.target.value)}
            className="mt-1 w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>

        {/* 結果表示 */}
        <div className="mt-6 text-center">
          {error ? (
            <p className="text-red-600">{error}</p>
          ) : (
            <>
              <BlockMath math={latex} />
              {isSymbolic ? (
                <p className="mt-4 text-lg text-gray-600">
                  <em>一般式を表示しています</em>
                </p>
              ) : (
                <p className="mt-4 text-2xl font-semibold text-gray-800">
                  = {result}
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
