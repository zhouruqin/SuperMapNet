import os
import os.path as osp
import shutil

def move_files_with_structure(root_path, txt_path, target_root):
    """
    根据txt文件中的相对路径，保留目录结构移动文件到目标根目录
    :param txt_path: 记录源文件夹相对路径的txt文件路径
    :param target_root: 目标根目录绝对路径
    """
    # 确保目标根目录存在
    os.makedirs(target_root, exist_ok=True)
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        src_relative_paths = [line.strip() for line in f.readlines()]
        print(src_relative_paths)

    for src_rel_path in src_relative_paths:
        # 构建源文件夹绝对路径（相对于脚本所在目录）
        src_abs_path = osp.join(root_path, src_rel_path)  
        
        # 验证源路径有效性
        if not os.path.exists(src_abs_path):
            print(f"警告：源路径不存在，已跳过 {src_abs_path}")
            continue
            
        if not os.path.isdir(src_abs_path):
            print(f"警告：路径不是文件夹，已跳过 {src_abs_path}")
            continue

        # 遍历源文件夹所有文件
        for root, _, files in os.walk(src_abs_path):
            for filename in files:
                src_file = os.path.join(root, filename)
                
                # 计算相对于源文件夹的相对路径
                relative_path = os.path.relpath(src_file, src_abs_path)
                
                # 构建目标路径
                dest_file = os.path.join(
                    target_root,
                    src_rel_path,  # 保留原始相对路径作为父目录
                    relative_path  # 保留内部目录结构
                )
                
                # 创建目标目录结构
                dest_dir = os.path.dirname(dest_file)
                os.makedirs(dest_dir, exist_ok=True)
                
                try:
                    shutil.move(src_file, dest_file)
                    print(f"移动成功: {src_file} -> {dest_file}")
                except Exception as e:
                    print(f"移动失败: {src_file} | 错误: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    root_path = "/zrq/SuperMapNet/data/av2/train"
    txt_file = "/zrq/SuperMapNet/assets/splits/av2/av2_val_split.txt"      # 记录文件夹路径的txt文件
    destination = "/zrq/SuperMapNet/data/av2/val"      # 目标文件夹路径
    
    move_files_with_structure(root_path, txt_file, destination)
  