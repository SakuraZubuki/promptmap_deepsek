
import json
import pandas as pd

# 读取 JSON 文件
with open("results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# 提取 rule_name 和 failed_result.response
rows = []
for rule_name, content in results.items():
    failed = content.get("failed_result", {})
    response = failed.get("response")
    if response:
        rows.append({"rule_name": rule_name, "response": response})

# 保存为 CSV
df = pd.DataFrame(rows)
df.to_csv("attacked_prompts.csv", index=False)

print("✅ 已将 results.json 转换为 attacked_prompts.csv")
