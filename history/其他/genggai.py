import json


def process_damage_data(data):
    for entry in data['data']:
        # 处理每个 damageType
        processed_damage_type = []
        processed_damage_data = {}

        for dtype, values in entry['damageData'].items():
            # 过滤出大于3毫米的值
            filtered_values = [v for v in values if v > 3.0]
            if filtered_values:
                processed_damage_data[dtype] = filtered_values
                processed_damage_type.append(dtype)

        # 更新 JSON 数据
        entry['damageData'] = processed_damage_data
        entry['damageType'] = ', '.join(processed_damage_type)

    return data


# 读取 JSON 文件
with open('D:/PythonCode/8号线-鼓楼大街-20240904/latest_data.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 处理数据
processed_data = process_damage_data(json_data)

# 保存处理后的 JSON 数据
with open('D:/PythonCode/8号线-鼓楼大街-20240904/latest_data.json', 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, indent=4, ensure_ascii=False)

print("处理完成，结果已保存到output.json")
