#!/bin/bash


# 定义源文件和复制次数
source_file="task_1.sh"
copy_count=35  # 需要的副本数量，可以根据需要调整

# 进行批量复制和重命名
for i in $(seq 2 $((copy_count + 1))); do
  cp "$source_file" "task_$i.sh"
done
