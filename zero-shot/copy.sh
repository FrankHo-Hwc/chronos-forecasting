#!/bin/bash


# ����Դ�ļ��͸��ƴ���
source_file="task_1.sh"
copy_count=35  # ��Ҫ�ĸ������������Ը�����Ҫ����

# �����������ƺ�������
for i in $(seq 2 $((copy_count + 1))); do
  cp "$source_file" "task_$i.sh"
done
