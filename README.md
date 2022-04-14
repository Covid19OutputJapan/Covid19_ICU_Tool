# Covid19_ICU_Tool
[入院患者数・重症患者数見通しツール](https://covid19-icu-tool.herokuapp.com/) で使用しているモデルのソースコードです。  
使用しているモデルの詳細・パラメターの推定方法に関してはこちらの[論文](https://covid19outputjapan.github.io/JP/files/FujiiNakata_20210811.pdf)をご覧ください。  
本ツールの目的・ツールの使い方・データの出典等にこちらの[参考資料](https://covid19outputjapan.github.io/JP/files/NakataOkamoto_Briefing_20220413.pdf)  
なお、研究チームでは毎週本ツールを使用して47都道府県の病床見通しを公表する予定です。[ウェブサイト](https://covid19outputjapan.github.io/JP/index.html)をご確認ください。

## 注意
- データの自動取得コード、およびウェブアプリ（Made with Streamlit）のコードは非公開とさせていただきます。
- コード、ウェブアプリに関するご意見・ご質問は covid19outputjapan [at] gmail.com までお願いします。
- 分析に関するご意見・ご要望・ご質問は taisuke.nakata [at] e.u-tokyo.ac.jp までお願いします。自治体関係者の皆様のご連絡も積極的にお待ちしております。

## 実行方法
1. [Google スプレッドシート](https://docs.google.com/spreadsheets/d/1OOwRFo5sh_kaDQF79BdpAHhI_WXXcXpV5tj4NXYQBHk/edit?usp=sharing)で最新の都道府県別データを確認する。
2. シート名 `data` のワークシートを csv として download して、ディレクトリ直下に保存する。
3. `main.py` を編集し、パラメータとシミュレーション対象の都道府県を設定する。
4. コードを実行する。
```
python main.py
```

## 実行環境
Python 3.7 以上および、以下モジュールの version で動作確認済みです。  
※これ以外の version では動作を保証できません。

- `japanize-matplotlib` >=1.1.3
- `matplotlib` >=3.5.1
- `numba` >=0.55.1
- `numpy` >=1.21.5
- `pandas` >=1.3.5

## 免責事項
https://covid19outputjapan.github.io/JP/disclaimer.html

## 開発チーム
https://covid19outputjapan.github.io/JP/team.html

## 引用規定
本コードは無償利用・改変・流用が可能ですが、本コードに基づいて外部に成果物を発表する際は、必ず引用をお願いします。（例を以下に示します。）

`仲田泰祐, 岡本亘 (東京大学) 「入院患者数・重症患者数見通しツール」`
