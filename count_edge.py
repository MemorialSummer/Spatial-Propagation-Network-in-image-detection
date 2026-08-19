import torch
import os
import argparse

def count_edges_in_pt_file(file_path):
    """
    读取.pt文件并统计其中保存的边数量
    
    Args:
        file_path: .pt文件的路径
        
    Returns:
        int: 边的数量，如果读取失败返回None
    """
    try:
        # 加载.pt文件
        edge_index = torch.load(file_path)
        
        # 检查edge_index的维度
        if edge_index.dim() == 2:
            # 假设边是以2 x num_edges的形式存储
            num_edges = edge_index.size(1)
        elif edge_index.dim() == 1:
            # 如果是一维数组，可能存储的是边的数量
            num_edges = edge_index.size(0)
        else:
            print(f"警告：edge_index的维度为{edge_index.dim()}，无法正确解析")
            return None
            
        return num_edges
        
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：{e}")
        return None

def list_all_edge_files():
    """
    列出所有edge_index文件并统计每个文件中的边数量
    """
    edge_dir = "./outputs/edges"
    
    if not os.path.exists(edge_dir):
        print(f"目录 {edge_dir} 不存在")
        return
    
    # 查找所有edge_index文件
    edge_files = []
    for file in os.listdir(edge_dir):
        if file.startswith("edge_index_") and file.endswith(".pt"):
            edge_files.append(file)
    
    if not edge_files:
        print(f"在 {edge_dir} 中未找到任何edge_index文件")
        return
    
    # 按数字排序
    edge_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    print(f"找到 {len(edge_files)} 个edge_index文件：")
    print("-" * 50)
    
    total_edges = 0
    for file_name in edge_files:
        file_path = os.path.join(edge_dir, file_name)
        num_edges = count_edges_in_pt_file(file_path)
        
        if num_edges is not None:
            print(f"{file_name}: {num_edges:,} 条边")
            total_edges += num_edges
        else:
            print(f"{file_name}: 无法读取")
    
    print("-" * 50)
    print(f"总边数: {total_edges:,}")

def main():
    parser = argparse.ArgumentParser(description='统计edge_index.pt文件中保存的边数量')
    parser.add_argument('file_path', nargs='?', 
                       help='指定要查看的.pt文件路径（可选）')
    parser.add_argument('--list', '-l', action='store_true',
                       help='列出所有edge_index文件并统计')
    
    args = parser.parse_args()
    
    if args.list:
        list_all_edge_files()
    elif args.file_path:
        num_edges = count_edges_in_pt_file(args.file_path)
        if num_edges is not None:
            print(f"文件 {args.file_path} 中包含 {num_edges:,} 条边")
    else:
        print("请指定要查看的文件路径，或使用 --list 查看所有文件")

if __name__ == "__main__":
    main()