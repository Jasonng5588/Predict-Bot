import yfinance as yf

try:
    data = yf.download("NVDA", period="5d")
    print(data.tail())
    if data.empty:
        print("⚠️ 没有抓到任何资料，请检查网络或尝试 VPN。")
    else:
        print("✅ 成功抓到 NVDA 数据！")
except Exception as e:
    print("❌ 出错了：", e)
