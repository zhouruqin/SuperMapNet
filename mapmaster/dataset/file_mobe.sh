#!/bin/bash

# 批量复制目录脚本
if [ $# -ne 3 ]; then
    echo "用法: $0 <文件夹列表文件> <目标目录> <root name>"
    exit 1
fi

list_file="$1"
dest_dir="$2"
root_name="$3"

# 检查列表文件是否存在
if [ ! -f "$list_file" ]; then
    echo "错误：列表文件 $list_file 不存在"
    exit 1
fi

# 创建目标目录（自动创建父目录）
mkdir -p "$dest_dir" || exit 1

# 设置内部字段分隔符为换行，处理含空格路径
while IFS= read -r folder; do
    # 跳过空行和注释行（以#开头）
    if [[ -z "$folder" || "$folder" =~ ^# ]]; then
        continue
    fi
    
    #root_name = "/zrq/SuperMapNet/data/av2/train"
    clean_root="${root_name}"
    clean_sub="${folder}"
    source_folder="$clean_root/$clean_sub"
    echo "$clean_root, $clean_sub"
    # 检查源目录是否存在
    if [ ! -d "$source_folder" ]; then
        echo "警告：源目录 '$source_folder' 不存在，跳过"
        continue
    fi

    # 获取目录基名

    folder_name=$(basename "$folder")
    target_path="$dest_dir/$folder_name"

    # 删除已存在的目标目录
    if [ -d "$target_path" ]; then
        echo "发现已存在目录，正在清理: $target_path"
        rm -rf "$target_path"
    fi

    # 执行复制操作
    echo "正在移动: $source_folder -> $dest_dir/"
    if mv "$source_folder" "$dest_dir/"; then
        echo "移动成功: $source_folder"
    else
        echo "错误：移动 $source_folder 失败"
    fi
done < "$list_file"

echo ""
echo "操作完成！共处理 $(grep -v '^$' "$list_file" | wc -l) 个目录"